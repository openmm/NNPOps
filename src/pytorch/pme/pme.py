from ..neighbors import getNeighborPairs
import torch
import math

class PME:
    """This class implements the Particle Mesh Ewald algorithm (https://doi.org/10.1063/1.470117).

    This is a method of summing all the infinite pairwise electrostatic interactions in a periodic system.  It divides
    the energy into two parts: a short range term that can be computed efficiently in direct space, and a long range
    term that can be computed efficiently in reciprocal space.  The individual terms are not physical meaningful, only
    their sum.

    This class supports periodic boundary conditions with arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c`
    must satisfy certain requirements:

    `a[1] = a[2] = b[2] = 0`
    `a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff`
    `a[0] >= 2*b[0]`
    `a[0] >= 2*c[0]`
    `b[1] >= 2*c[1]`

    These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the
    requirement that the cutoff be no larger than half the box width.

    You can optionally specify that certain interactions should be omitted when computing the energy.  This is typically
    used for nearby atoms within the same molecule.  When two atoms are listed as an exclusion, only the interaction of
    each with the same periodic copy of the other (that is, not applying periodic boundary conditions) is excluded.
    Each atom still interacts with all the periodic copies of the other.

    Due to the way the reciprocal space term is calculated, it is impossible to prevent it from including excluded
    interactions.  The direct space term therefore compensates for it, subtracting off the energy that was incorrectly
    included in reciprocal space.  The sum of the two terms thus yields the correct energy with the interaction fully
    excluded.

    When performing backpropagation, this class computes derivatives with respect to atomic positions and charges, but
    not to any other parameters (box vectors, alpha, etc.).  In addition, it only computes first derivatives.
    Attempting to compute a second derivative will throw an exception.  This means that if you use PME during training,
    the loss function can only depend on energy, not forces.

    When you create an instance of this class, you must specify the value of Coulomb's constant 1/(4*pi*eps0).  Its
    value depends on the units used for energy and distance.  The value you specify thus sets the unit system.  Here are
    the values for some common units.

    kJ/mol, nm: 138.935457
    kJ/mol, A: 1389.35457
    kcal/mol, nm: 33.2063713
    kcal/mol, A: 332.063713
    eV, nm: 1.43996454
    eV, A: 14.3996454
    hartree, bohr: 1.0
    """
    def __init__(self, gridx: int, gridy: int, gridz: int, order: int, alpha: float, coulomb: float, exclusions: torch.Tensor):
        """Create an object for computing energies with PME.

        Parameters
        ----------
        gridx: int
            the size of the charge grid along the x axis
        gridy: int
            the size of the charge grid along the y axis
        gridz: int
            the size of the charge grid along the z axis
        order: int
            the B-spline order to use for charge spreading.  With CUDA, only order 4 and 5 are supported.
        alpha: float
            the coefficient of the erf() function used to separate the energy into direct and reciprocal space terms
        coulomb: float
            Coulomb's constant 1/(4*pi*eps0).  This sets the unit system.
        exclusions: torch.Tensor
            a tensor of shape `(atoms, max_exclusions)` containing excluded interactions, where `max_exclusions` is the
            maximum number of exclusions for any atom.  Row `i` lists the indices of all atoms with which atom `i` should
            not interact.  If an atom has less than `max_exclusions` excluded interactions, set the remaining elements
            in the row to -1.  The exclusions must be symmetric: if `j` appears in row `i`, then `i` must also appear in
            row `j`.  If you pass a tensor that does not satisfy that requirement, the results are undefined.
        """
        if gridx < 1 or gridy < 1 or gridz < 1:
            raise ValueError('The grid dimensions must be positive')
        if order < 1:
            raise ValueError('order must be positive')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if coulomb <= 0:
            raise ValueError('coulomb must be positive')
        if exclusions.dim() != 2:
            raise ValueError('exclusions must be 2D')
        self.gridx = gridx
        self.gridy = gridy
        self.gridz = gridz
        self.order = order
        self.alpha = alpha
        self.coulomb = coulomb
        self.exclusions, _ = torch.sort(exclusions.to(torch.int32), descending=True)

        # Initialize the bspline moduli.

        max_size = max(gridx, gridy, gridz)
        data = torch.zeros(order, dtype=torch.float32)
        ddata = torch.zeros(order, dtype=torch.float32)
        bsplines_data = torch.zeros(max_size, dtype=torch.float32)
        data[0] = 1
        for i in range(3, order):
            data[i-1] = 0
            for j in range(1, i-1):
                data[i-j-1] = (j*data[i-j-2]+(i-j)*data[i-j-1])/(i-1)
            data[0] /= i-1

        # Differentiate.

        ddata[0] = -data[0]
        ddata[1:order] = data[0:order-1]-data[1:order]
        for i in range(1, order-1):
            data[order-i-1] = (i*data[order-i-2]+(order-i)*data[order-i-1])/(order-1)
        data[0] /= order-1
        bsplines_data[1:order+1] = data

        # Evaluate the actual bspline moduli for X/Y/Z.

        self.moduli = []
        for ndata in (gridx, gridy, gridz):
            m = torch.zeros(ndata, dtype=torch.float32)
            for i in range(ndata):
                arg = (2*torch.pi*i/ndata)*torch.arange(ndata)
                sc = torch.sum(bsplines_data[:ndata]*torch.cos(arg))
                ss = torch.sum(bsplines_data[:ndata]*torch.sin(arg))
                m[i] = sc*sc + ss*ss
            for i in range(ndata):
                if m[i] < 1e-7:
                    m[i] = (m[(i-1+ndata)%ndata]+m[(i+1)%ndata])*0.5
            self.moduli.append(m)

    def compute_direct(self, positions: torch.Tensor, charges: torch.Tensor, cutoff: float, box_vectors: torch.Tensor, max_num_pairs: int = -1):
        """Compute the direct space energy.

        Parameters
        ----------
        positions: torch.Tensor
            a 2D tensor of shape `(atoms, 3)` containing the positions of the atoms
        charges: torch.Tensor
            a 1D tensor of length `atoms` containing the charge of each atom
        cutoff: float
            the cutoff distance to use when computing the direct space term
        box_vectors: torch.Tensor
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
        max_num_pairs: int, optional
            Maximum number of pairs (total number of neighbors). If set to `-1` (default),
            all possible combinations of atom pairs are included.

        Returns
        -------
        the energy of the direct space term
        """
        if positions.dim() != 2 or positions.shape[1] != 3:
            raise ValueError('positions must have shape (atoms, 3)')
        if charges.dim() != 1:
            raise ValueError('charges must be 1D')
        if positions.shape[0] != self.exclusions.shape[0] or charges.shape[0] != self.exclusions.shape[0]:
            raise ValueError('positions, charges, and exclusions must all have the same length')
        if box_vectors.dim() != 2 or box_vectors.shape[0] != 3 or box_vectors.shape[1] != 3:
            raise ValueError('box_vectors must have shape (3, 3)')
        if (cutoff <= 0):
            raise ValueError('cutoff must be positive')
        neighbors, deltas, distances, number_found_pairs = getNeighborPairs(positions, cutoff, max_num_pairs, box_vectors)
        self.exclusions = self.exclusions.to(positions.device)
        return torch.ops.pme.pme_direct(positions, charges, neighbors, deltas, distances, self.exclusions, self.alpha, self.coulomb)

    def compute_reciprocal(self, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor):
        """Compute the reciprocal space energy.

        Parameters
        ----------
        positions: torch.Tensor
            a 2D tensor of shape `(atoms, 3)` containing the positions of the atoms
        charges: torch.Tensor
            a 1D tensor of length `atoms` containing the charge of each atom
        box_vectors: torch.Tensor
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.

        Returns
        -------
        the energy of the reciprocal space term
        """
        if positions.dim() != 2 or positions.shape[1] != 3:
            raise ValueError('positions must have shape (atoms, 3)')
        if charges.dim() != 1:
            raise ValueError('charges must be 1D')
        if positions.shape[0] != self.exclusions.shape[0] or charges.shape[0] != self.exclusions.shape[0]:
            raise ValueError('positions, charges, and exclusions must all have the same length')
        if box_vectors.dim() != 2 or box_vectors.shape[0] != 3 or box_vectors.shape[1] != 3:
            raise ValueError('box_vectors must have shape (3, 3)')
        for i in range(3):
            self.moduli[i] = self.moduli[i].to(positions.device)
        self_energy = -torch.sum(charges**2)*self.coulomb*self.alpha/math.sqrt(torch.pi)
        return self_energy + torch.ops.pme.pme_reciprocal(positions, charges, box_vectors, self.gridx, self.gridy, self.gridz,
                                            self.order, self.alpha, self.coulomb, self.moduli[0], self.moduli[1], self.moduli[2])
