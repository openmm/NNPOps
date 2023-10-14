'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch
from torch.utils.cpp_extension import CLIB_PREFIX, IS_MACOS, LIB_EXT

_LIBRARY_EXT = LIB_EXT if not IS_MACOS else ".dylib"
_LIBRARY_NAME = f'{CLIB_PREFIX}NNPOpsPyTorch{_LIBRARY_EXT}'

torch.ops.load_library(os.path.join(os.path.dirname(__file__), _LIBRARY_NAME))


from NNPOps.OptimizedTorchANI import OptimizedTorchANI
