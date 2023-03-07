'''
Neighbor operations
'''
import site
import os
import torch

torch.ops.load_library(os.path.join(site.getsitepackages()[-1],"NNPOps", "libNNPOpsPyTorch.so"))

from NNPOps.neighbors.getNeighborPairs import getNeighborPairs
