'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import platform
import torch
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("NNPOps")
except PackageNotFoundError:
    # package is not installed
    pass

if platform.system() == 'Darwin':
    libname = 'libNNPOpsPyTorch.dylib'
elif platform.system() == 'Windows':
    libname = 'NNPOpsPyTorch.dll'
else:
    libname = 'libNNPOpsPyTorch.so'
torch.ops.load_library(os.path.join(os.path.dirname(__file__), libname))


from NNPOps.OptimizedTorchANI import OptimizedTorchANI
