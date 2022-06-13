from torch import ops, Tensor
from typing import Tuple


def getNeighborPairs(positions: Tensor, cutoff: float, max_num_neighbors: int = -1) -> Tuple[Tensor, Tensor]:
    '''
    Returns indices and distances of atom pairs within a given cutoff distance.

    If `max_num_neighbors == -1` (default), all the atom pairs are returned,
    i.e. `num_pairs = num_atoms * (num_atoms + 1) / 2`. This is intended for
    the small molecules, where almost all the atoms are within the cutoff
    distance of each other.

    If `max_num_neighbors > 0`, a fixed number of the atom pair are returned,
    i.e. `num_pairs = num_atoms * max_num_neighbors`. This is indeded for large
    molecule, where most of the atoms are beyond the cutoff distance of each
    other.

    Parameters
    ----------
    positions: `torch.Tensor`
        Atomic positions. The tensor shape has to be `(num_atoms, 3)` and
        data type has to be`torch.float32` or `torch.float64`.
    cutoff: float
        Maximum distance between atom pairs.
    max_num_neighbors: int, optional
        Maximum number of neighbors per atom. If set to `-1` (default),
        all possible combinations of atom pairs are included.

    Returns
    -------
    neighbors: `torch.Tensor`
        Atom pair indices. The shape of the tensor is `(2, num_pairs)`.
        If an atom pair is separated by a larger distance than the cutoff,
        the indices are set to `-1`.

    distances: `torch.Tensor`
        Atom pair distances. The shape of the tensor is `(num_pairs)`.
        If an atom pair is separated by a larger distance than the cutoff,
        the distance is set to `NaN`.

    Exceptions
    ----------
    If `max_num_neighbors > 0` and too small, `RuntimeError` is raised.

    Note
    ----
    The operation is compatible with CUDA Grahps, i.e. the shapes of the output
    tensors are independed of the values of input tensors.

    The CUDA implementation returns the atom pairs in non-determinist order,
    if `max_num_neighbors > 0`.
    '''

    return ops.neighbors.getNeighborPairs(positions, cutoff, max_num_neighbors)