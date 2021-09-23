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
$ ctest
```