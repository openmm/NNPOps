'''
High-performance PyTorch operations for neural network potentials
'''

from NNPOps.OptimizedTorchANI import OptimizedTorchANI
from torch import ops, Tensor
from typing import Tuple


def getNeighborPairs(positions: Tensor, cutoff: float, max_num_neighbors: int = -1) -> Tuple[Tensor, Tensor]:
    '''
    TODO
    '''

    return ops.neighbors.getNeighborPairs(positions, cutoff, max_num_neighbors)