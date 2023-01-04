from torch import ops, Tensor


def passMessages(neighbors: Tensor, messages: Tensor, states: Tensor) -> Tensor:
    '''
    Pass messages between the neighbor atoms.

    Given a set of `num_atoms` atoms (each atom has a state with `num_features`
    features) and a set of `num_neighbors` neighbor atom pairs (each pair has a
    message with `num_features` features), the messages of the pairs are added
    to the corresponding atom states.

    Parameters
    ----------
    neighbors: `torch.Tensor`
        Atom pair indices. The shape of the tensor is `(2, num_pairs)`.
        The indices can be `[0, num_atoms)` or `-1` (ignored pairs).
        See for the documentation of `NNPOps.neighbors.getNeighborPairs` for
        details.
    messages: `torch.Tensor`
        Atom pair messages. The shape of the tensor is `(num_pairs, num_features)`.
        For efficient, `num_features` has to be a multiple of 32 and <= 1024.
    states: `torch.Tensor`
        Atom states. The shape of the tensor is `(num_atoms, num_features)`.

    Returns
    -------
    new_states: `torch.Tensor`
        Update atom states. The shape of the tensor is `(num_atoms, num_features)`.

    Note
    ----
    The operation is compatible with CUDA Grahps, i.e. the shapes of the output
    tensors are independed of the values of input tensors.

    Examples
    --------
    >>> import torch as pt
    >>> from NNPOps.messages import passMessages

    >>> num_atoms = 4
    >>> num_neigbors = 3
    >>> num_features = 32

    >>> neighbors = pt.tensor([[0, -1, 1], [0, -1, 3]], dtype=pt.int32)

    >>> messages = pt.ones((num_neigbors, 32)); messages[1] = 5
    >>> messages[:, 0]
    tensor([1., 5., 1.])

    >>> states = pt.zeros((num_atoms, num_features)); states[1] = 3
    >>> states[:, 0]
    tensor([0., 3., 0., 0.])

    >>> new_states = passMessages(neighbors, messages, states)
    >>> new_states[:, 0]
    tensor([2., 4., 0., 1.])
    '''

    return ops.messages.passMessages(neighbors, messages, states)