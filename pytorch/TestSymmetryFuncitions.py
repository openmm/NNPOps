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
import pytest
import torch
import torchani

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
def test_compare_with_native(deviceString):

    import SymmetryFunctions

    device = torch.device(deviceString)

    mol = mdtraj.load('molecules/2iuz_ligand.mol2')
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz, dtype=torch.float32, requires_grad=True, device=device)

    nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
    energy_ref = nnp((atomicNumbers, atomicPositions)).energies
    energy_ref.backward()
    grad_ref = atomicPositions.grad.clone()

    nnp.aev_computer = SymmetryFunctions.TorchANISymmetryFunctions(nnp.aev_computer)
    energy = nnp((atomicNumbers, atomicPositions)).energies
    atomicPositions.grad.zero_()
    energy.backward()
    grad = atomicPositions.grad.clone()

    assert torch.abs((energy - energy_ref)/energy_ref) < 1e-7
    assert torch.max(torch.abs((grad - grad_ref)/grad_ref)) < 6e-4