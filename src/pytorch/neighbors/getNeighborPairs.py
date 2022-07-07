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

    return ops.neighbors.getNeighborPairs(positions, cutoff, max_num_neighbors)