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

class CFConvNeighbors(torch.nn.Module):
    """
    Optimized nearest-neighbor implementation for the continious-filter convolution (CFConf)

    CFConv is used in SchNet (https://arxiv.org/abs/1706.08566).

    See NNPOps.CFConv for more details.
    """

    def __init__(self, cutoff: float) -> None:

        super().__init__()

        self.holder = torch.classes.NNPOpsCFConvNeighbors.Holder(cutoff)

    @torch.jit.export
    def build(self, positions: Tensor) -> None:

        self.holder.build(positions)