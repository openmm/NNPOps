'''
Neighbor operations
'''
import os.path
import torch

torch.ops.load_library(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libNNPOpsPyTorch.so'))

from NNPOps.neighbors.getNeighborPairs import getNeighborPairs
