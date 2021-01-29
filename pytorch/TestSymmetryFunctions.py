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
import tempfile
import torch
from ase.io import read
import torchani

from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions


@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['3NIR.pdb'])
def test_compare_with_native_pbc_on(deviceString, molFile):

    device = torch.device(deviceString)

    mol = read(f'molecules/{molFile}')
    atomicNumbers = torch.tensor([mol.get_atomic_numbers()], device=device)
    atomicPositions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=True, device=device)
    cell = torch.tensor(mol.get_cell(complete=True), dtype=torch.float32, device=device)
    pbc = torch.tensor(mol.get_pbc(), dtype=torch.bool, device=device)
    if pbc[0] is False:
        pbc = None
        cell = None
    print(f'Mol: {molFile}, size: {atomicNumbers.shape[-1]} atoms')
    print(f'cell: {cell}')
    print(f'pbc: {pbc}')

    nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
    speciesPositions = nnp.species_converter((atomicNumbers, atomicPositions))
    aev_ref = nnp.aev_computer(speciesPositions, cell, pbc).aevs
    energy_ref = nnp((atomicNumbers, atomicPositions), cell, pbc).energies
    energy_ref.backward()
    grad_ref = atomicPositions.grad.clone()

    nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer)
    energy = nnp((atomicNumbers, atomicPositions), cell, pbc).energies
    aev = nnp.aev_computer(speciesPositions, cell, pbc).aevs
    aev_pbc_off = nnp.aev_computer(speciesPositions).aevs
    atomicPositions.grad.zero_()
    energy.backward()
    grad = atomicPositions.grad.clone()

    aev_error = torch.max(torch.abs((aev - aev_ref)))
    aev_diff_pbc_on_off = torch.max(torch.abs((aev - aev_pbc_off)))
    energy_error = torch.abs((energy - energy_ref)/energy_ref).item()
    grad_error = torch.max(torch.abs((grad - grad_ref)/grad_ref))

    print(f'AEV Error: {aev_error:.1e}')
    print(f'AEV difference between PBC ON and OFF: {aev_diff_pbc_on_off:.1e}')
    print(f'Energy Error: {energy_error:.1e}')
    print(f'Grad Error: {grad_error:.1e}')
    assert aev_error < 5e-5
    assert energy_error < 5e-5
    assert grad_error < 5e-3


@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_compare_with_native(deviceString, molFile):

    device = torch.device(deviceString)

    mol = mdtraj.load(f'molecules/{molFile}_ligand.mol2')
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

    nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
    energy_ref = nnp((atomicNumbers, atomicPositions)).energies
    energy_ref.backward()
    grad_ref = atomicPositions.grad.clone()

    nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer)
    energy = nnp((atomicNumbers, atomicPositions)).energies
    atomicPositions.grad.zero_()
    energy.backward()
    grad = atomicPositions.grad.clone()

    energy_error = torch.abs((energy - energy_ref)/energy_ref)
    grad_error = torch.max(torch.abs((grad - grad_ref)/grad_ref))

    assert energy_error < 5e-7
    assert grad_error < 5e-3

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99'])
def test_model_serialization(deviceString, molFile):

    device = torch.device(deviceString)

    mol = mdtraj.load(f'molecules/{molFile}_ligand.mol2')
    atomicNumbers = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
    atomicPositions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

    nnp_ref = torchani.models.ANI2x(periodic_table_index=True).to(device)
    nnp_ref.aev_computer = TorchANISymmetryFunctions(nnp_ref.aev_computer)

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