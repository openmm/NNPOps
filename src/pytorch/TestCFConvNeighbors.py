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

import mdtraj
import os
import pytest
import tempfile
import torch

molecules = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecules')

def test_import():
    import NNPOps
    import NNPOps.CFConvNeighbors

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_build(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.CFConvNeighbors import CFConvNeighbors

    device = torch.device(deviceString)

    mol = mdtraj.load(os.path.join(molecules, f'{molFile}_ligand.mol2'))
    positions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)[0]

    neighbors = CFConvNeighbors(numAtoms=len(positions), cutoff=5)

    for _ in range(3):
        neighbors.build(positions)
        # TODO test the result


@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_model_serialization(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.CFConvNeighbors import CFConvNeighbors

    device = torch.device(deviceString)

    mol = mdtraj.load(os.path.join(molecules, f'{molFile}_ligand.mol2'))
    positions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)[0]

    neighbors_ref = CFConvNeighbors(numAtoms=len(positions), cutoff=5)

    for _ in range(3):
        neighbors_ref.build(positions)
        # TODO test the result

    with tempfile.NamedTemporaryFile() as fd:

        torch.jit.script(neighbors_ref).save(fd.name)
        neighbors = torch.jit.load(fd.name)

        for _ in range(3):
            neighbors.build(positions)
            # TODO test the result