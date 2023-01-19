from torch import empty, ops, Tensor
from typing import Optional, Tuple


def getNeighborPairs(positions: Tensor, cutoff: float, max_num_neighbors: int = -1, box_vectors: Optional[Tensor] = None, check_errors: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
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

    This function optionally supports periodic boundary conditions with
    arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
    certain requirements:

    `a[1] = a[2] = b[2] = 0`
    `a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff`
    `a[0] >= 2*b[0]`
    `a[0] >= 2*c[0]`
    `b[1] >= 2*c[1]`

    These requirements correspond to a particular rotation of the system and
    reduced form of the vectors, as well as the requirement that the cutoff be
    no larger than half the box width.

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
    box_vectors: `torch.Tensor`, optional
        The vectors defining the periodic box.  This must have shape `(3, 3)`,
        where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
        If this is omitted, periodic boundary conditions are not applied.
    check_errors: bool, optional
        If set to True the function is guaranteed to throw if an error
        is encountered, synchronizing if necessary.
        If set to False, an error arising in this function might crash
        the code at some point after calling it.
        Defaults to False.

    Returns
    -------
    neighbors: `torch.Tensor`
        Atom pair indices. The shape of the tensor is `(2, num_pairs)`.
        If an atom pair is separated by a larger distance than the cutoff,
        the indices are set to `-1`.

    deltas: `torch.Tensor`
        Atom pair displacement vectors. The shape of the tensor is `(num_pairs, 3)`.
        The direction of the vectors are from `neighbors[1]` to `neighbors[0]`.
        If an atom pair is separated by a larger distance than the cutoff,
        the displacement vector is set to `[NaN, NaN, NaN]`.

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

    The check_errors argument is forced to False if CUDA graphs are used.

    Examples
    --------
    >>> import torch as pt
    >>> from NNPOps.neighbors import getNeighborPairs

    >>> positions = pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    >>> getNeighborPairs(positions, cutoff=3.0) # doctest: +NORMALIZE_WHITESPACE
    (tensor([[1, 2, 2],
             [0, 0, 1]], dtype=torch.int32),
     tensor([[1., 0., 0.],
             [2., 0., 0.],
             [1., 0., 0.]]),
     tensor([1., 2., 1.]))

    >>> getNeighborPairs(positions, cutoff=1.5) # doctest: +NORMALIZE_WHITESPACE
    (tensor([[ 1, -1,  2],
             [ 0, -1,  1]], dtype=torch.int32),
     tensor([[1., 0., 0.],
             [nan, nan, nan],
             [1., 0., 0.]]),
     tensor([1., nan, 1.]))

    >>> getNeighborPairs(positions, cutoff=3.0, max_num_neighbors=2) # doctest: +NORMALIZE_WHITESPACE
    (tensor([[ 1,  2,  2, -1, -1, -1],
             [ 0,  0,  1, -1, -1, -1]], dtype=torch.int32),
     tensor([[1., 0., 0.],
             [2., 0., 0.],
             [1., 0., 0.],
             [nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan]]),
     tensor([1., 2., 1., nan, nan, nan]))

    >>> getNeighborPairs(positions, cutoff=1.5, max_num_neighbors=2) # doctest: +NORMALIZE_WHITESPACE
    (tensor([[ 1,  2, -1, -1, -1, -1],
             [ 0,  1, -1, -1, -1, -1]], dtype=torch.int32),
     tensor([[1., 0., 0.],
             [1., 0., 0.],
             [nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan],
             [nan, nan, nan]]),
     tensor([1., 1., nan, nan, nan, nan]))

    '''

    if box_vectors is None:
        box_vectors = empty((0, 0), device=positions.device, dtype=positions.dtype)
    return ops.neighbors.getNeighborPairs(positions, cutoff, max_num_neighbors, box_vectors, check_errors)
