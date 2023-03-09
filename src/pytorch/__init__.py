'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch
import site

# look for NNPOps/libNNPOpsPyTorch.so in all the paths returned by site.getsitepackages()
for path in site.getsitepackages():
    if os.path.exists(os.path.join(path, 'NNPOps/libNNPOpsPyTorch.so')):
        torch.ops.load_library(os.path.join(path, 'NNPOps/libNNPOpsPyTorch.so'))
        break
else:
    # if we didn't find it, look for NNPOps/libNNPOpsPyTorch.so in the same directory as this file
    torch.ops.load_library(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libNNPOpsPyTorch.so'))


from NNPOps.OptimizedTorchANI import OptimizedTorchANI
