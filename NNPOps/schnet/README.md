## SchNet

This directory contains code to compute the continuous filter convolution (cfconv)
function used in [SchNet](https://aip.scitation.org/doi/10.1063/1.5019779).  It is
organized as follows.

`CFConv` and `CFConvNeighbors` define an abstract interface for computing the
cfconv function and its gradients.  They each have two subclasses, for example
`CpuCFConv` and `CudaCFConv`, that provide implementations.

Test suites are provided for both implementations.  You can compile them with

```
c++ --std=c++11 TestCpuCFConv.cpp CpuCFConv.cpp
```

and

```
nvcc --std=c++11 TestCudaCFConv.cu CudaCFConv.cu
```

Also provided is a program for benchmarking the performance of the CUDA implementation.
Compile it with

```
nvcc --std=c++11 BenchmarkCudaCFConv.cu CudaCFConv.cu
```

When running it, provide two command line arguments: the path to a PDB file and
a number of iterations to perform.  It calculates the values and gradients of a
stack of cfconv layers similar to those found in a typical SchNet model.
