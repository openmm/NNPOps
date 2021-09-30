# PyTorch wrapper for NNPOps

*NNPOps* functionalities are available in [*PyTorch*](https://pytorch.org/).

## Optimized TorchANI operations

- [`torchani.AEVComputer`](https://aiqm.github.io/torchani/api.html?highlight=speciesaev#torchani.AEVComputer)
- [`torchani.neurochem.NeuralNetwork`](https://aiqm.github.io/torchani/api.html#module-torchani.neurochem)

## Example

```python
import mdtraj
import torch
import torchani

from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions
from NNPOps.BatchedNN import TorchANIBatchedNN

device = torch.device('cuda')

# Load a molecule
molecule = mdtraj.load('molecule.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in molecule.top.atoms]], device=device)
positions = torch.tensor(molecule.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

# Construct ANI-2x and replace its operations with the optimized ones
nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer).to(device)
nnp.neural_networks = TorchANIBatchedNN(nnp.species_converter, nnp.neural_networks, species).to(device)

# Compute energy and forces
energy = nnp((species, positions)).energies
energy.backward()
forces = -positions.grad.clone()

print(energy, forces)
```