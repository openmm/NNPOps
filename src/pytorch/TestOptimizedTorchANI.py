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

import mdtraj
import os
import pytest
import tempfile
import torch
import torchani

molecules = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecules')

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_compare_with_native(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps import OptimizedTorchANI

    device = torch.device(deviceString)

    mol = mdtraj.load(os.path.join(molecules, f'{molFile}_ligand.mol2'))
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

    nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
    energy_ref = nnp((atomicNumbers, atomicPositions)).energies
    energy_ref.backward()
    grad_ref = atomicPositions.grad.clone()

    nnp = OptimizedTorchANI(nnp, atomicNumbers).to(device)
    energy = nnp((atomicNumbers, atomicPositions)).energies
    atomicPositions.grad.zero_()
    energy.backward()
    grad = atomicPositions.grad.clone()

    energy_error = torch.abs((energy - energy_ref)/energy_ref)
    grad_error = torch.max(torch.abs((grad - grad_ref)/grad_ref))

    assert energy_error < 5e-7
    if molFile == '3o99':
        assert grad_error < 7e-3
    else:
        assert grad_error < 5e-3

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
def test_compare_waterbox_pbc_with_native(deviceString):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps import OptimizedTorchANI

    device = torch.device(deviceString)

    mol = mdtraj.load(os.path.join(molecules, 'water.pdb'))
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)
    cell = mol.unitcell_vectors[0]
    cell = torch.tensor(cell, dtype=torch.float32, device=device)*10.0
    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

    nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
    energy_ref = nnp((atomicNumbers, atomicPositions), cell=cell, pbc=pbc).energies
    energy_ref.backward()
    grad_ref = atomicPositions.grad.clone()

    nnp = OptimizedTorchANI(nnp, atomicNumbers).to(device)
    energy = nnp((atomicNumbers, atomicPositions), cell=cell, pbc=pbc).energies
    atomicPositions.grad.zero_()
    energy.backward()
    grad = atomicPositions.grad.clone()

    energy_error = torch.abs((energy - energy_ref)/energy_ref)
    grad_error = torch.max(torch.abs((grad - grad_ref)/grad_ref))

    assert energy_error < 5e-7
    assert grad_error < 8e-3

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_model_serialization(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps import OptimizedTorchANI

    device = torch.device(deviceString)

    mol = mdtraj.load(os.path.join(molecules, f'{molFile}_ligand.mol2'))
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

    nnp_ref = torchani.models.ANI2x(periodic_table_index=True).to(device)
    nnp_ref = OptimizedTorchANI(nnp_ref, atomicNumbers).to(device)

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
    assert grad_error < 5e-3

def test_device_change():
    # Give OptimizedTorchANI a chance to initialize using the gpu,
    # then move the positions to the cpu and call it again
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    device = torch.device('cuda')
    mol = mdtraj.load(os.path.join(molecules, f'1hvj_ligand.mol2'))
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)
    from NNPOps import OptimizedTorchANI
    nnp_ref = torchani.models.ANI2x(periodic_table_index=True).to(device)
    nnp_ref = OptimizedTorchANI(nnp_ref, atomicNumbers).to(device)
    energy_gpu = nnp_ref((atomicNumbers, atomicPositions)).energies
    atomicPositions = atomicPositions.to("cpu")
    energy_cpu = nnp_ref((atomicNumbers, atomicPositions)).energies
    assert torch.allclose(energy_gpu, energy_cpu.to(device))
