'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))
torch.classes.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))

from NNPOps.OptimizedTorchANI import OptimizedTorchANI
