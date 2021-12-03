#include "CpuEquivariantTransformer.h"

EquivariantTransformerNeighbors* createNeighbors(int numAtoms, float cutoff, bool periodic) {
    return new CpuEquivariantTransformerNeighbors(numAtoms, cutoff, periodic);
}

EquivariantTransformerLayer* createLayer(int numAtoms, int width, int numHeads, int numRBF, const float* rbfMus, const float* rbfBetas,
              const float* qw, const float* qb, const float* kw, const float* kb, const float* vw, const float* vb,
              const float* ow, const float* ob, const float* uw, const float* ub, const float* dkw, const float* dkb,
              const float* dvw, const float* dvb) {
    return new CpuEquivariantTransformerLayer(numAtoms, width, numHeads, numRBF, rbfMus, rbfBetas, qw, qb, kw, kb, vw, vb, ow, ob, uw, ub, dkw, dkb, dvw, dvb);
}

#include "TestEquivariantTransformer.h"
