'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch
from torch.utils.cpp_extension import IS_WINDOWS, IS_MACOS, LIB_EXT

# by default torch assumes OSX has a .so ext, so we need to override
_LIBRARY_SUFFIX = LIB_EXT if not IS_MACOS else ".dylib"
_LIBRARY_PREFIX = '' if IS_WINDOWS else "lib"
_LIBRARY_NAME = f'{_LIBRARY_PREFIX}NNPOpsPyTorch{_LIBRARY_SUFFIX}'

torch.ops.load_library(os.path.join(os.path.dirname(__file__), _LIBRARY_NAME))


from NNPOps.OptimizedTorchANI import OptimizedTorchANI
