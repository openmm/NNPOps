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
import time
import torch
import torchani

# from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions
from NNPOps.BatchedNN import TorchANIBatchedNN

device = torch.device('cuda')

mol = mdtraj.load('molecules/2iuz_ligand.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
positions = torch.tensor(mol.xyz, dtype=torch.float32, requires_grad=True, device=device)

nnp = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
print(nnp)

energy_ref = nnp((species, positions)).energies
energy_ref.backward()
grad_ref = positions.grad.clone()

N = 2000
start = time.time()
for _ in range(N):
    energy_ref = nnp((species, positions)).energies
delta = time.time() - start
print(f'ANI-2x (forward pass)')
print(f'  Duration: {delta} s')
print(f'  Speed: {delta/N*1000} ms/it')

N = 1000
start = time.time()
for _ in range(N):
    energy_ref = nnp((species, positions)).energies
    positions.grad.zero_()
    energy_ref.backward()
delta = time.time() - start
print(f'ANI-2x (forward & backward pass)')
print(f'  Duration: {delta} s')
print(f'  Speed: {delta/N*1000} ms/it')

# nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer).to(device)
nnp.neural_networks = TorchANIBatchedNN(nnp.species_converter, nnp.neural_networks, species).to(device)
print(nnp)

# nnp = torch.jit.script(nnp)
# nnp.save('nnp.pt')
# npp = torch.jit.load('nnp.pt')

energy = nnp((species, positions)).energies
positions.grad.zero_()
energy.backward()
grad = positions.grad.clone()

N = 10000
start = time.time()
for _ in range(N):
    energy = nnp((species, positions)).energies
delta = time.time() - start
print(f'ANI-2x with BatchedNN (forward pass)')
print(f'  Duration: {delta} s')
print(f'  Speed: {delta/N*1000} ms/it')

N = 5000
start = time.time()
for _ in range(N):
    energy = nnp((species, positions)).energies
    positions.grad.zero_()
    energy.backward()
delta = time.time() - start
print(f'ANI-2x with BatchedNN (forward & backward pass)')
print(f'  Duration: {delta} s')
print(f'  Speed: {delta/N*1000} ms/it')

# print(float(energy_ref), float(energy), float(energy_ref - energy))
# print(float(torch.max(torch.abs((grad - grad_ref)/grad_ref))))