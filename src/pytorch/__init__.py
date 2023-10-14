'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import sys
import torch.utils.cpp_extension

_LIBRARY_NAME = f'NNPOpsPyTorch.{torch.utils.cpp_extension}'

if not sys.platform.lower().startswith("win"):
    _LIBRARY_NAME = f'lib{_LIBRARY_NAME}'

torch.ops.load_library(os.path.join(os.path.dirname(__file__), _LIBRARY_NAME))


from NNPOps.OptimizedTorchANI import OptimizedTorchANI
