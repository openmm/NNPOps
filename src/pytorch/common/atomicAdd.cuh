#ifndef NNPOPS_ATOMICADD_H
#define NNPOPS_ATOMICADD_H

/*
Implement atomicAdd with double precision numbers for pre-Pascal GPUs.
Taken from https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
NOTE: remove when the support of CUDA 11 is dropped.
*/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif