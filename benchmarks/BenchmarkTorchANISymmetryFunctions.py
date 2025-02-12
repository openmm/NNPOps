import mdtraj
import time
import torch
import torchani

from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

device = torch.device('cuda')

mol = mdtraj.load('./tests/molecules/2iuz_ligand.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
positions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

nnp = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
speciesPositions = nnp.species_converter((species, positions))
symmFuncRef = nnp.aev_computer
symmFunc = TorchANISymmetryFunctions(nnp.species_converter, nnp.aev_computer, species).to(device)

aev_ref = symmFuncRef(speciesPositions).aevs
sum_aev_ref = torch.sum(aev_ref)
sum_aev_ref.backward()
grad_ref = positions.grad.clone()

N = 10000
start = time.time()
for _ in range(N):
    aev_ref = symmFuncRef(speciesPositions).aevs
    sum_aev_ref = torch.sum(aev_ref)
    positions.grad.zero_()
    sum_aev_ref.backward()
delta = time.time() - start
grad_ref = positions.grad.clone()
print('Original TorchANI symmetry functions')
print(f'  Duration: {delta} s')
print(f'  Speed: {delta/N*1000} ms/it')

aev = symmFunc(speciesPositions).aevs
sum_aev = torch.sum(aev)
positions.grad.zero_()
sum_aev.backward()
grad = positions.grad.clone()

N = 100000
start = time.time()
for _ in range(N):
    aev = symmFunc(speciesPositions).aevs
    sum_aev = torch.sum(aev)
    positions.grad.zero_()
    sum_aev.backward()
delta = time.time() - start
grad = positions.grad.clone()
print('Optimized TorchANI symmetry functions')
print(f'  Duration: {delta} s')
print(f'  Speed: {delta/N*1000} ms/it')

aev_error = torch.max(torch.abs(aev - aev_ref))
grad_error = torch.max(torch.abs(grad - grad_ref))
assert aev_error < 0.0002
assert grad_error < 0.007