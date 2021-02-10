# Conda package

## Build

```bash
export PATH=/usr/local/cuda-10.2/bin:$PATH
conda build nnpops-pytorch --channel conda-forge --python 3.7
```