#
# Copyright (c) 2020-2021 Acellera
# Authors: Raimondas Galvelis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os.path
import torch
from torch import Tensor

from NNPOps.CFConvNeighbors import CFConvNeighbors

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))
torch.classes.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))

class CFConv(torch.nn.Module):
    """
    Optimized continious-filter convolution layer (CFConv)

    CFConv is used in SchNet (https://arxiv.org/abs/1706.08566).

    Example::

        import torch
        from NNPOps.CFConvNeighbors import CFConvNeighbors
        from NNPOps.CFConv import CFConv

        # Set parameters
        numAtoms = 7
        numFilters = 5
        numGaussians = 3
        cutoff = 5.0
        gaussianWidth = 1.0
        activation = 'ssp'
        weights1 = torch.rand(numGaussians, numFilters)
        biases1 = torch.rand(numFilters)
        weights2 = torch.rand(numFilters, numFilters)
        biases2 = torch.rand(numFilters)

        # Generate random input
        positions = (10*torch.rand(numAtoms, 3) - 5).detach()
        positions.requires_grad = True
        input = torch.rand(numAtoms, numFilters)

        # Create objects
        neighbors = CFConvNeighbors(cutoff)
        conv = CFConv(gaussianWidth, activation, weights1, biases1, weights2, biases2)

        # Run forward and backward passes
        neighbors.build(positions)
        output = conv(neighbors, positions, input)
        total = torch.sum(output)
        total.backward()
        grad = positions.grad
    """

    Holder = torch.classes.NNPOpsCFConv.Holder
    operation = torch.ops.NNPOpsCFConv.operation

    def __init__(self, gaussianWidth: float, activation: str,
                 weights1: Tensor, biases1: Tensor,
                 weights2: Tensor, biases2: Tensor) -> None:

        super().__init__()

        self.holder = CFConv.Holder(gaussianWidth, activation, weights1, biases1, weights2, biases2)

    def forward(self, neighbors: CFConvNeighbors, positions: Tensor, input: Tensor) -> Tensor:

        return CFConv.operation(self.holder, neighbors.holder, positions, input)