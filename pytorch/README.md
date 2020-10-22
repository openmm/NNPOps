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
positions = torch.tensor(molecule.xyz, dtype=torch.float32, requires_grad=True, device=device)

# Construct ANI-2x and replace its native featurizer with NNPOps implementation
nnp = torchani.models.ANI2x(periodic_table_index=True).to(device)
nnp.aev_computer = TorchANISymmetryFunctions(nnp.aev_computer)

# Compute energy
energy = nnp((species, positions)).energies
energy.backward()
forces = -positions.grad.clone()

print(energy, forces)
```

## Installation

### Prerequisites

- *Linux*
- Complete *CUDA Toolkit* (https://developer.nvidia.com/cuda-downloads)
- *Miniconda* (https://docs.conda.io/en/latest/miniconda.html#linux-installers)

### Build & install

- Crate a *Conda* environment
```bash
$ conda create -n nnpops \
               -c pytorch \
               -c conda-forge \
               cmake \
               git \
               gxx_linux-64 \
               make \
               mdtraj \
               pytest \
               python=3.8 \
               pytorch=1.6 \
               torchani=2.2
$ conda activate nnpops
```
- Get the source code
```bash
$ git clone https://github.com/peastman/NNPOps.git
```
- Configure, build, and install
```bash
$ mkdir build
$ cd build
$ cmake ../NNPOps/pytorch \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_HOST_COMPILER=$CXX \
        -DTorch_DIR=$CONDA_PREFIX/lib/python3.8/site-packages/torch/share/cmake/Torch \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ make install
```
- Optional: run tests
```bash
$ cd ../NNPOps/pytorch
$ pytest TestSymmetryFunctions.py
```