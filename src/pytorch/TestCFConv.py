#
# Copyright (c) 2020 Acellera
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

import os
import pytest
import tempfile
import torch

molecules = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecules')

def test_import():
    import NNPOps
    import NNPOps.CFConvNeighbors
    import NNPOps.CFConv

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
def test_gradients(deviceString):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.CFConvNeighbors import CFConvNeighbors
    from NNPOps.CFConv import CFConv

    device = torch.device(deviceString)

    numAtoms = 3
    numFilters = 5
    numGaussians = 7
    cutoff = 5.0
    gaussianWidth = 1.0
    activation = 'ssp'
    weights1 = torch.rand(numGaussians, numFilters, dtype=torch.float32, device=device)
    biases1 = torch.rand(numFilters, dtype=torch.float32, device=device)
    weights2 = torch.rand(numFilters, numFilters, dtype=torch.float32, device=device)
    biases2 = torch.rand(numFilters, dtype=torch.float32, device=device)

    positions = (10*torch.rand(numAtoms, 3, dtype=torch.float32, device=device) - 5).detach()
    positions.requires_grad = True
    input = torch.rand(numAtoms, numFilters, dtype=torch.float32, device=device)

    neighbors = CFConvNeighbors(cutoff)
    conv = CFConv(neighbors, numGaussians, gaussianWidth, activation, weights1, biases1, weights2, biases2)

    neighbors.build(positions)
    output = conv(positions, input)
    total = torch.sum(output)
    total.backward()
    grad = positions.grad

    assert output.device == positions.device
    assert output.dtype == torch.float32
    assert output.shape == (numAtoms, numFilters)

    assert grad.device == positions.device
    assert grad.dtype == torch.float32
    assert grad.shape == (numAtoms, 3)

    # def func(pos):
    #     return torch.sum(conv(pos, input))
    # assert torch.autograd.gradcheck(func, positions)

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
def test_model_serialization(deviceString):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.CFConvNeighbors import CFConvNeighbors
    from NNPOps.CFConv import CFConv

    device = torch.device(deviceString)

    numAtoms = 3
    numFilters = 5
    numGaussians = 7
    cutoff = 5.0
    gaussianWidth = 1.0
    activation = 'ssp'
    weights1 = torch.rand(numGaussians, numFilters, dtype=torch.float32, device=device)
    biases1 = torch.rand(numFilters, dtype=torch.float32, device=device)
    weights2 = torch.rand(numFilters, numFilters, dtype=torch.float32, device=device)
    biases2 = torch.rand(numFilters, dtype=torch.float32, device=device)

    positions = (10*torch.rand(numAtoms, 3, dtype=torch.float32, device=device) - 5).detach()
    positions.requires_grad = True
    input = torch.rand(numAtoms, numFilters, dtype=torch.float32, device=device)

    neighbors_ref = CFConvNeighbors(cutoff)
    conv_ref = CFConv(neighbors_ref, numGaussians, gaussianWidth, activation, weights1, biases1, weights2, biases2)

    neighbors_ref.build(positions)
    output_ref = conv_ref(positions, input)
    total_ref = torch.sum(output_ref)
    total_ref.backward()
    grad_ref = positions.grad.clone()

    with tempfile.NamedTemporaryFile() as fd:

        torch.jit.script(conv_ref).save(fd.name)
        conv = torch.jit.load(fd.name).to(device)

        conv.neighbors.build(positions)
        output = conv(positions, input)
        total = torch.sum(output)
        positions.grad.zero_()
        total.backward()
        grad = positions.grad.clone()

    assert torch.allclose(output, output_ref, rtol=1e-07, atol=0)
    assert torch.allclose(grad, grad_ref, rtol=1e-07, atol=0)