#include "CudaCFConv.h"

CFConvNeighbors* createNeighbors(int numAtoms, float cutoff, bool periodic) {
    return new CudaCFConvNeighbors(numAtoms, cutoff, periodic);
}

CFConv* createConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth,
                   float* w1, float* b1, float* w2, float* b2) {
    return new CudaCFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth, w1, b1, w2, b2);
}

#include "TestCFConv.h"
