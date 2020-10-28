#include "CpuCFConv.h"

CFConvNeighbors* createNeighbors(int numAtoms, float cutoff, bool periodic) {
    return new CpuCFConvNeighbors(numAtoms, cutoff, periodic);
}

CFConv* createConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth) {
    return new CpuCFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth);
}

#include "TestCFConv.h"
