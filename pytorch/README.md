# PyTorch wrapper for NNPOps

## Installation

### Prerequisites

- A *Linux* machine
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