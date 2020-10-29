import mdtraj
import time
import torch
import torchani

from BatchedNN import TorchANIBatchedNNs
# from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

device = torch.device('cuda')

mol = mdtraj.load('../pytorch/molecules/2iuz_ligand.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
elements = [atom.element.symbol for atom in mol.top.atoms]
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
nnp.neural_networks = TorchANIBatchedNNs(nnp.neural_networks, elements).to(device)
print(nnp)

# torch.jit.script(nnp).save('nnp.pt')
# npp = torch.jit.load('nnp.pt')

energy = nnp((species, positions)).energies
positions.grad.zero_()
energy.backward()
grad = positions.grad.clone()

N = 20000
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