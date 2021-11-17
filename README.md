# NNPOps

The goal of this repository is to promote the use of neural network potentials (NNPs)
by providing highly optimized, open source implementations of bottleneck operations
that appear in popular potentials.  These are the core design principles.

1. Each operation is entirely self contained, consisting of only a few source files
that can easily be incorporated into any code that needs to use it.

2. Each operation has a simple, clearly documented API allowing it to be easily
used in any context.

3. We provide both CPU (pure C++) and CUDA implementations of all operations.

4. The CUDA implementations are highly optimized.  The CPU implementations are written
in a generally efficient way, but no particular efforts have been made to tune them
for optimum performance.

5. This code is designed for inference (running simulations), not training (creating
new potential functions).  It computes gradients with respect to particle positions,
not model parameters.

## Installation

### From the source

#### Prerequisites

- *CUDA Toolkit* (https://developer.nvidia.com/cuda-downloads)
- *Miniconda* (https://docs.conda.io/en/latest/miniconda.html#linux-installers)

#### Build & install

- Get the source code
```bash
$ git clone https://github.com/openmm/NNPOps.git
```

- Set `CUDA_HOME`
```bash
$ export CUDA_HOME=/usr/local/cuda-11.2
```

- Crate and activate a *Conda* environment
```bash
$ cd NNPOps
$ conda env create -n nnpops -f environment.yml
$ conda activate nnpops
```

- Configure, build, and install
```bash
$ mkdir build && cd build
$ cmake .. \
        -DTorch_DIR=$CONDA_PREFIX/lib/python3.9/site-packages/torch/share/cmake/Torch \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ make install
```

- Run the tests
```bash
$ ctest --verbose
```

## Usage

Accelerated [*TorchANI*](https://aiqm.github.io/torchani/) operations:
- [`torchani.AEVComputer`](https://aiqm.github.io/torchani/api.html?highlight=speciesaev#torchani.AEVComputer)
- [`torchani.neurochem.NeuralNetwork`](https://aiqm.github.io/torchani/api.html#module-torchani.neurochem)

### Example

```python
import mdtraj
import torch
import torchani

from NNPOps.SpeciesConverter import TorchANISpeciesConverter
from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions
from NNPOps.BatchedNN import TorchANIBatchedNN
from NNPOps.EnergyShifter import TorchANIEnergyShifter

device = torch.device('cuda')

# Load a molecule
molecule = mdtraj.load('molecule.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in molecule.top.atoms]], device=device)
positions = torch.tensor(molecule.xyz * 10, dtype=torch.float32, requires_grad=True, device=device)

# Construct ANI-2x and replace its operations with the optimized ones
nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
nnp.species_converter = TorchANISpeciesConverter(nnp.species_converter, species).to(device)
nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer).to(device)
nnp.neural_networks = TorchANIBatchedNN(nnp.species_converter, nnp.neural_networks, species).to(device)
nnp.energy_shifter = TorchANIEnergyShifter(nnp.species_converter, nnp.energy_shifter, species).to(device)

# Compute energy and forces
energy = nnp((species, positions)).energies
energy.backward()
forces = -positions.grad.clone()

print(energy, forces)
```