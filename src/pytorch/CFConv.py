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
from typing import List, Optional, Tuple
import torch
from torch import Tensor

from NNPOps.CFConvNeighbors import CFConvNeighbors

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))
torch.classes.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))

Holder = torch.classes.NNPOpsCFConv.Holder
operation = torch.ops.NNPOpsCFConv.operation

class CFConv(torch.nn.Module):

    def __init__(self,
                 neighbors: CFConvNeighbors,
                 numAtoms: int,
                 numFilters: int,
                 numGaussians: int,
                 gaussianWidth: float,
                 activation: str,
                 weights1: Tensor,
                 biases1: Tensor,
                 weights2: Tensor,
                 biases2: Tensor) -> None:

        super().__init__()

        self.neighbors = neighbors
        self.numAtoms = numAtoms
        self.numFilters = numFilters
        self.numGaussians = numGaussians
        self.gaussianWidth = gaussianWidth
        self.activation = {'ssp': 0, 'tanh': 1}[activation]
        self.weights1 = weights1
        self.biases1 = biases1
        self.weights2 = weights2
        self.biases2 = biases2

        # Create an uninitialized holder
        self.holder = Holder(self.neighbors.holder, 0, 0, 0, 0.0, 0, Tensor(), Tensor(), Tensor(), Tensor(), Tensor())
        assert not self.holder.is_initialized()


    def forward(self, positions: Tensor, input: Tensor) -> Tensor:

        if not self.holder.is_initialized():
            self.holder = Holder(self.neighbors.holder,
                                 self.numAtoms,
                                 self.numFilters,
                                 self.numGaussians,
                                 self.gaussianWidth,
                                 self.activation,
                                 self.weights1,
                                 self.biases1,
                                 self.weights2,
                                 self.biases2,
                                 positions)
            assert self.holder.is_initialized()

        return operation(self.holder, positions, input)