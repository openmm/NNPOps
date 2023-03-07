'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import site
import torch
torch.ops.load_library(os.path.join(site.getsitepackages()[-1],"NNPOps", "libNNPOpsPyTorch.so"))
torch.classes.load_library(os.path.join(site.getsitepackages()[-1],"NNPOps", "libNNPOpsPyTorch.so"))

from NNPOps.OptimizedTorchANI import OptimizedTorchANI
