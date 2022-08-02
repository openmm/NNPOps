from torch import ops, Tensor


def passMessages(neighbors: Tensor, messages: Tensor, states: Tensor) -> Tensor:
    '''
    TODO
    '''

    return ops.messages.passMessages(neighbors, messages, states)