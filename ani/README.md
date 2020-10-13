## ANI

This directory contains code to compute the symmetry functions that appear in the
[ANI](https://doi.org/10.1039/C6SC05720A) potential function.  It is organized as
follows.

`ANISymmetryFunctions` defines an abstract interface for computing symmetry
functions and their gradients.  It has two subclasses, `CpuANISymmetryFunctions`
and `CudaANISymmetryFunctions` that provide implementations.

Test suites are provided for both implementations.  You can compile them with

```
c++ --std=c++11 TestCpuANISymmetryFunctions.cpp CpuANISymmetryFunctions.cpp
```

and

```
nvcc --std=c++11 TestCudaANISymmetryFunctions.cpp CudaANISymmetryFunctions.cu
```

Also provided is a program for benchmarking the performance of the CUDA implementation.
Compile it with

```
/usr/local/cuda/bin/nvcc --std=c++11 BenchmarkCudaANISymmetryFunctions.cu CudaANISymmetryFunctions.cu
```

When running it, provide two command line arguments: the path to a PDB file and
a number of iterations to perform.  It constructs a representation of the molecules
in the file with the ANI-2 potential, then computes both the symmetry functions
and their gradients the specified number of times.
