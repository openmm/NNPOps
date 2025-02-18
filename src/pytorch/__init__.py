'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("NNPOps")
except PackageNotFoundError:
    # package is not installed
    pass

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))


from NNPOps.OptimizedTorchANI import OptimizedTorchANI
