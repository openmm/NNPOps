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

torch.classes.load_library(os.path.join(os.path.dirname(__file__), 'libNNPOpsPyTorch.so'))

Holder = torch.classes.NNPOpsCFConvNeighbors.Holder

class CFConvNeighbors(torch.nn.Module):

    def __init__(self, numAtoms: int, cutoff: float) -> None:

        super().__init__()

        self.numAtoms = numAtoms
        self.cutoff = cutoff

        # Create an uninitialized holder
        self.holder = Holder(0, 0.0, torch.device('cpu'))
        assert not self.holder.is_initialized()

    @torch.jit.export
    def build(self, positions: Tensor) -> None:

        if not self.holder.is_initialized():
            self.holder = Holder(self.numAtoms, self.cutoff, positions.device)
            assert self.holder.is_initialized()

        self.holder.build(positions)