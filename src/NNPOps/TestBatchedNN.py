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
import torchani

molecules = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecules')

def test_import():
    import NNPOps
    import NNPOps.BatchedNN

class DeterministicTorch:
    def __enter__(self):
        if torch.are_deterministic_algorithms_enabled():
            self._already_enabled = True
            return
        self._already_enabled = False
        torch.use_deterministic_algorithms(True)

    def __exit__(self, type, value, traceback):
        if not self._already_enabled:
            torch.use_deterministic_algorithms(False)

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_compare_with_native(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    with DeterministicTorch():
        from NNPOps.BatchedNN import TorchANIBatchedNN

        device = torch.device(deviceString)

        mol = mdtraj.load(os.path.join(molecules, f'{molFile}_ligand.mol2'))
        atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
        atomicPositions = torch.tensor(mol.xyz, dtype=torch.float32, requires_grad=True, device=device)

        nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
        energy_ref = nnp((atomicNumbers, atomicPositions)).energies
        energy_ref.backward()
        grad_ref = atomicPositions.grad.clone()

        nnp.neural_networks = TorchANIBatchedNN(nnp.species_converter, nnp.neural_networks, atomicNumbers).to(device)
        energy = nnp((atomicNumbers, atomicPositions)).energies
        atomicPositions.grad.zero_()
        energy.backward()
        grad = atomicPositions.grad.clone()

        energy_error = torch.abs((energy - energy_ref)/energy_ref)
        grad_error = torch.max(torch.abs((grad - grad_ref)/grad_ref))

        assert energy_error < 5e-7
        if molFile == '3o99':
            assert grad_error < 0.025 # Some numerical instability
        else:
            assert grad_error < 5e-3

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_model_serialization(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.BatchedNN import TorchANIBatchedNN

    device = torch.device(deviceString)

    mol = mdtraj.load(os.path.join(molecules, f'{molFile}_ligand.mol2'))
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz, dtype=torch.float32, requires_grad=True, device=device)

    nnp_ref = torchani.models.ANI2x(periodic_table_index=True).to(device)
    nnp_ref.neural_networks = TorchANIBatchedNN(nnp_ref.species_converter, nnp_ref.neural_networks, atomicNumbers).to(device)

    energy_ref = nnp_ref((atomicNumbers, atomicPositions)).energies
    energy_ref.backward()
    grad_ref = atomicPositions.grad.clone()

    with tempfile.NamedTemporaryFile() as fd:

        torch.jit.script(nnp_ref).save(fd.name)
        nnp = torch.jit.load(fd.name)

        energy = nnp((atomicNumbers, atomicPositions)).energies
        atomicPositions.grad.zero_()
        energy.backward()
        grad = atomicPositions.grad.clone()

    energy_error = torch.abs((energy - energy_ref)/energy_ref)
    grad_error = torch.max(torch.abs((grad - grad_ref)/grad_ref))

    assert energy_error < 5e-7
    if molFile == '3o99':
        assert grad_error < 0.05 # Some numerical instability
    else:
        assert grad_error < 5e-3
