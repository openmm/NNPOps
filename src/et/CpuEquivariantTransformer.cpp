/**
 * Copyright (c) 2020-2021 Stanford University and the Authors
 * Authors: Peter Eastman
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "CpuEquivariantTransformer.h"
#include <cmath>
#include <cstring>

using namespace std;

template <bool PERIODIC, bool TRICLINIC>
static void computeDisplacement(const float* pos1, const float* pos2, float* delta, float& r2, const float* periodicBoxVectors, const float* invBoxSize) {
    delta[0] = pos2[0]-pos1[0];
    delta[1] = pos2[1]-pos1[1];
    delta[2] = pos2[2]-pos1[2];
    if (PERIODIC) {
        if (TRICLINIC) {
            float scale3 = round(delta[2]*invBoxSize[2]);
            delta[0] -= scale3*periodicBoxVectors[6];
            delta[1] -= scale3*periodicBoxVectors[7];
            delta[2] -= scale3*periodicBoxVectors[8];
            float scale2 = round(delta[1]*invBoxSize[1]);
            delta[0] -= scale2*periodicBoxVectors[3];
            delta[1] -= scale2*periodicBoxVectors[4];
            float scale1 = round(delta[0]*invBoxSize[0]);
            delta[0] -= scale1*periodicBoxVectors[0];
        }
        else {
            delta[0] -= round(delta[0]*invBoxSize[0])*periodicBoxVectors[0];
            delta[1] -= round(delta[1]*invBoxSize[1])*periodicBoxVectors[4];
            delta[2] -= round(delta[2]*invBoxSize[2])*periodicBoxVectors[8];
        }
    }
    r2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
}

CpuEquivariantTransformerNeighbors::CpuEquivariantTransformerNeighbors(int numAtoms, float lowerCutoff, float upperCutoff, bool periodic) :
            EquivariantTransformerNeighbors(numAtoms, lowerCutoff, upperCutoff, periodic) {
    neighbors.resize(numAtoms);
    neighborDistances.resize(numAtoms);
}

void CpuEquivariantTransformerNeighbors::build(const float* positions, const float* periodicBoxVectors) {
    // Determine whether we have a rectangular or triclinic periodic box.
    
    triclinic = false;
    if (getPeriodic())
        for (int i = 0 ; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j && periodicBoxVectors[3*i+j] != 0)
                    triclinic = true;

    // Clear the output arrays.

    for (int i = 0; i < getNumAtoms(); i++) {
        neighbors[i].clear();
        neighborDistances[i].clear();
    }

    // Build the neighbor list.

    if (getPeriodic()) {
        if (triclinic)
            findNeighbors<true, true>(positions, periodicBoxVectors);
        else
            findNeighbors<true, false>(positions, periodicBoxVectors);
    }
    else {
        findNeighbors<false, false>(positions, periodicBoxVectors);
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuEquivariantTransformerNeighbors::findNeighbors(const float* positions, const float* periodicBoxVectors) {
    float invBoxSize[3];
    if (getPeriodic()) {
        invBoxSize[0] = 1/periodicBoxVectors[0];
        invBoxSize[1] = 1/periodicBoxVectors[4];
        invBoxSize[2] = 1/periodicBoxVectors[8];
    }

    // Loop over all pairs of atoms using an O(N^2) algorithms.  This is efficient for the small molecules
    // we are currently targeting.  If we want to scale to larger systems, a voxel based algorithm would
    // be more efficient.

    float cutoff2 = getCutoff()*getCutoff();
    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        for (int atom2 = atom1+1; atom2 < getNumAtoms(); atom2++) {
            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2, periodicBoxVectors, invBoxSize);
            if (r2 < cutoff2) {
                neighbors[atom1].push_back(atom2);
                neighborDistances[atom1].push_back(sqrtf(r2));
            }
        }
    }
}

CpuEquivariantTransformerLayer::CpuEquivariantTransformerLayer(int numAtoms, int width, int numHeads, const vector<float>& rbfMus,
            const vector<float>& rbfBetas, const float* qw, const float* qb, const float* kw, const float* kb, const float* vw, const float* vb,
            const float* ow, const float* ob, const float* uw, const float* ub, const float* dkw, const float* dkb, const float* dvw, const float* dvb) :
            EquivariantTransformerLayer(numAtoms, width, numHeads, rbfMus, rbfBetas) {
    for (int i = 0; i < width*width; i++) {
        this->qw.push_back(qw[i]);
        this->kw.push_back(kw[i]);
    }
    for (int i = 0; i < 3*width*width; i++) {
        this->vw.push_back(vw[i]);
        this->ow.push_back(ow[i]);
        this->uw.push_back(uw[i]);
    }
    int numRBF = rbfMus.size();
    for (int i = 0; i < width*numRBF; i++)
        this->dkw.push_back(dkw[i]);
    for (int i = 0; i < 3*width*numRBF; i++)
        this->dvw.push_back(dvw[i]);
    for (int i = 0; i < width; i++) {
        this->qb.push_back(qb[i]);
        this->kb.push_back(kb[i]);
        this->dkb.push_back(dkb[i]);
    }
    for (int i = 0; i < 3*width; i++) {
        this->vb.push_back(vb[i]);
        this->ob.push_back(ob[i]);
        this->ub.push_back(ub[i]);
        this->dvb.push_back(dvb[i]);
    }
}

void CpuEquivariantTransformerLayer::compute(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                 const float* x, const float* vec, float* dx, float* dvec) {
    const CpuEquivariantTransformerNeighbors& cpuNeighbors = dynamic_cast<const CpuEquivariantTransformerNeighbors&>(neighbors);
    int numRBF = rbfMus.size();
    vector<float> rbf(numRBF), q(width), k(width), v(3*width), o(3*width), u(3*width), dk(width), dv(3*width);

    // Clear the output arrays.

    memset(dx, 0, getNumAtoms()*width*sizeof(float));
    memset(dvec, 0, 3*getNumAtoms()*width*sizeof(float));

    // Loop over pairs of atoms from the neighbor list.

    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        for (int neighborIndex = 0; neighborIndex < cpuNeighbors.getNeighbors()[atom1].size(); neighborIndex++) {
            int atom2 = cpuNeighbors.getNeighbors()[atom1][neighborIndex];
            float r = cpuNeighbors.getNeighborDistances()[atom1][neighborIndex];

            // Compute the radial basis functions.

            float alpha = 5.0f/(neighbors.getUpperCutoff()-neighbors.getLowerCutoff());
            float cutoffScale = cutoffFunction(r, neighbors.getUpperCutoff());
            for (int i = 0; i < numRBF; i++) {
                float expTerm = exp(-alpha*(r-neighbors.getLowerCutoff()));
                float expDiff = expTerm-rbfMus[i];
                rbf[i] = cutoffScale*exp(-rbfBetas[i]*expDiff*expDiff);
            }

            // Apply the various transforms.

            for (int i = 0; i < width; i++) {
                q[i] = qb[i];
                k[i] = kb[i];
                dk[i] = dkb[i];
                for (int j = 0; j < width; j++) {
                    q[i] += qw[i*width+j]*x[atom2*width+j];
                    k[i] += kw[i*width+j]*x[atom2*width+j];
                }
                for (int j = 0; j < numRBF; j++)
                    dk[i] += dkw[i*numRBF+j]*rbf[j];
                dk[i] = dk[i]/(1+exp(dk[i]));
            }
            for (int i = 0; i < 3*width; i++) {
                v[i] = vb[i];
                u[i] = ub[i];
                dv[i] = dvb[i];
                for (int j = 0; j < width; j++) {
                    v[i] += vw[i*width+j]*vec[atom2*3*width+j];
                    u[i] += uw[i*width+j]*vec[atom2*3*width+j];
                }
                for (int j = 0; j < numRBF; j++)
                    dv[i] += dvw[i*numRBF+j]*rbf[j];
                dv[i] = dv[i]/(1+exp(dv[i]));
            }

            // Apply the second dense layer.

            float cutoffScale = cutoffFunction(r);
            for (int i = 0; i < width; i++) {
                float sum = b2[i];
                for (int j = 0; j < width; j++)
                    sum += y1[j]*w2[i*width+j];
                y2[i] = cutoffScale*sum;
            }

            // Add it to the output.

            for (int i = 0; i < width; i++) {
                output[atom1*width+i] += y2[i]*input[atom2*width+i];
                output[atom2*width+i] += y2[i]*input[atom1*width+i];
            }
        }
    }
}

void CpuEquivariantTransformerLayer::backprop(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                         const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv) {
    // Clear the output array.

    memset(inputDeriv, 0, numAtoms*width*sizeof(float));
    memset(positionDeriv, 0, numAtoms*3*sizeof(float));

    // Backpropagate through the symmetry functions.

    const CpuEquivariantTransformerNeighbors& cpuNeighbors = dynamic_cast<const CpuEquivariantTransformerNeighbors&>(neighbors);
    if (neighbors.getPeriodic()) {
        if (neighbors.getTriclinic())
            backpropImpl<true, true>(cpuNeighbors, positions, periodicBoxVectors, input, outputDeriv, inputDeriv, positionDeriv);
        else
            backpropImpl<true, false>(cpuNeighbors, positions, periodicBoxVectors, input, outputDeriv, inputDeriv, positionDeriv);
    }
    else {
        backpropImpl<false, false>(cpuNeighbors, positions, periodicBoxVectors, input, outputDeriv, inputDeriv, positionDeriv);
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuEquivariantTransformerLayer::backpropImpl(const CpuEquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                             const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv) {
    int numRBF = rbfMus.size();
    vector<float> rbf(numRBF), dRBFdR(numRBF);
    vector<float> y1(width), dY1dR(width);
    vector<float> y2(width), dY2dR(width);
    float invBoxSize[3];
    if (PERIODIC) {
        invBoxSize[0] = 1/periodicBoxVectors[0];
        invBoxSize[1] = 1/periodicBoxVectors[4];
        invBoxSize[2] = 1/periodicBoxVectors[8];
    }

    // Clear the output arrays.

    memset(inputDeriv, 0, getNumAtoms()*width*sizeof(float));
    memset(positionDeriv, 0, getNumAtoms()*3*sizeof(float));

    // Loop over pairs of atoms from the neighbor list.

    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        for (int atom2 : neighbors.getNeighbors()[atom1]) {
            // Compute the displacement.

            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2, periodicBoxVectors, invBoxSize);
            float r = sqrtf(r2);
            float rInv = 1/r;

            // Compute the radial basis functions.

            float alpha = 5.0f/(neighbors.getUpperCutoff()-neighbors.getLowerCutoff());
            float cutoffScale = cutoffFunction(r, neighbors.getUpperCutoff());
            for (int i = 0; i < numRBF; i++) {
                float expTerm = exp(-alpha*(r-neighbors.getLowerCutoff());
                float expDiff = expTerm-rbfMus[i];
                rbf[i] = cutoffScale*exp(-rbfBetas[i]*expDiff*expDiff);
                dRBFdR[i] = rbf[i]*2*rbfBetas[i]*expDiff*expTerm*alpha;
            }

            // Apply the first dense layer.

            for (int i = 0; i < width; i++) {
                float sum = b1[i], dSumdR = 0;
                for (int j = 0; j < numRBF; j++) {
                    sum += rbf[j]*w1[i*numRBF+j];
                    dSumdR += dRBFdR[j]*w1[i*numRBF+j];
                }
                if (getActivation() == ShiftedSoftplus) {
                    float expSum = expf(sum);
                    y1[i] = logf(0.5f*expSum + 0.5f);
                    dY1dR[i] = dSumdR*expSum/(expSum + 1);
                }
                else {
                    float th = tanhf(sum);
                    y1[i] = th;
                    dY1dR[i] = dSumdR*(1-th*th);
                }
            }

            // Apply the second dense layer.

            float cutoffScale = cutoffFunction(r);
            float dCutoffdR = cutoffDeriv(r);
            for (int i = 0; i < width; i++) {
                float sum = b2[i], dSumdR = 0;
                for (int j = 0; j < width; j++) {
                    sum += y1[j]*w2[i*width+j];
                    dSumdR += dY1dR[j]*w2[i*width+j];
                }
                y2[i] = cutoffScale*sum;
                dY2dR[i] = dCutoffdR*sum + cutoffScale*dSumdR;
            }

            // Add it to the output.

            for (int i = 0; i < width; i++) {
                int index1 = atom1*width+i;
                int index2 = atom2*width+i;
                inputDeriv[index1] += y2[i]*outputDeriv[index2];
                inputDeriv[index2] += y2[i]*outputDeriv[index1];
                float scale = rInv*dY2dR[i]*(input[index2]*outputDeriv[index1] + input[index1]*outputDeriv[index2]);
                for (int j = 0; j < 3; j++) {
                    float dVdX = scale * delta[j];
                    positionDeriv[atom1*3+j] -= dVdX;
                    positionDeriv[atom2*3+j] += dVdX;
                }
            }
        }
    }
}

float CpuEquivariantTransformerLayer::cutoffFunction(float r, float cutoff) {
    return 0.5f * cosf(M_PI*r/cutoff) + 0.5f;
}

float CpuEquivariantTransformerLayer::cutoffDeriv(float r, float cutoff) {
    return -(0.5f*M_PI/cutoff) * sinf(M_PI*r/cutoff);
}
