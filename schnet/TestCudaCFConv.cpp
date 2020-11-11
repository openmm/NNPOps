#include "CudaCFConv.h"

CFConvNeighbors* createNeighbors(int numAtoms, float cutoff, bool periodic) {
    return new CudaCFConvNeighbors(numAtoms, cutoff, periodic);
}

CFConv* createConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth) {
    return new CudaCFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth);
}

#include "TestCFConv.h"
