# PyTorch wrapper for NNPOps

*NNPOps* functionalities are available in *PyTorch* (https://pytorch.org/).

## Optimized TorchANI symmetry functions

Optimized drop-in replacement for `torchani.AEVComputer` (https://aiqm.github.io/torchani/api.html?highlight=speciesaev#torchani.AEVComputer)

### Example

```python
import mdtraj
import torch
import torchani

from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

device = torch.device('cuda')

# Load a molecule
molecule = mdtraj.load('molecule.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in molecule.top.atoms]], device=device)
positions = torch.tensor(molecule.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

# Construct ANI-2x and replace its native featurizer with NNPOps implementation
nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer)

# Compute energy
energy = nnp((species, positions)).energies
energy.backward()
forces = -positions.grad.clone()

print(energy, forces)
```