/**
 * Copyright (c) 2020 Stanford University and the Authors
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

#include "CpuCFConv.h"
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

CpuCFConvNeighbors::CpuCFConvNeighbors(int numAtoms, float cutoff, bool periodic) : CFConvNeighbors(numAtoms, cutoff, periodic) {
    neighbors.resize(numAtoms);
    neighborDistances.resize(numAtoms);
}

void CpuCFConvNeighbors::build(const float* positions, const float* periodicBoxVectors) {
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
void CpuCFConvNeighbors::findNeighbors(const float* positions, const float* periodicBoxVectors) {
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

CpuCFConv::CpuCFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth,
                     ActivationFunction activation, const float* w1, const float* b1, const float* w2, const float* b2) :
            CFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth, activation) {
    for (int i = 0; i < numGaussians; i++)
        gaussianPos.push_back(i*cutoff/(numGaussians-1));
    for (int i = 0; i < numGaussians*width; i++)
        this->w1.push_back(w1[i]);
    for (int i = 0; i < width*width; i++)
        this->w2.push_back(w2[i]);
    for (int i = 0; i < width; i++) {
        this->b1.push_back(b1[i]);
        this->b2.push_back(b2[i]);
    }
}

void CpuCFConv::compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                 const float* input, float* output) {
    const CpuCFConvNeighbors& cpuNeighbors = dynamic_cast<const CpuCFConvNeighbors&>(neighbors);
    vector<float> gaussian(getNumGaussians());
    vector<float> y1(getWidth());
    vector<float> y2(getWidth());

    // Clear the output array.

    memset(output, 0, getNumAtoms()*getWidth()*sizeof(float));

    // Loop over pairs of atoms from the neighbor list.

    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        for (int neighborIndex = 0; neighborIndex < cpuNeighbors.getNeighbors()[atom1].size(); neighborIndex++) {
            int atom2 = cpuNeighbors.getNeighbors()[atom1][neighborIndex];
            float r = cpuNeighbors.getNeighborDistances()[atom1][neighborIndex];

            // Compute the Gaussian basis functions.

            for (int i = 0; i < getNumGaussians(); i++) {
                float x = (r-gaussianPos[i])/getGaussianWidth();
                gaussian[i] = exp(-0.5f*x*x);
            }

            // Apply the first dense layer.

            for (int i = 0; i < getWidth(); i++) {
                float sum = b1[i];
                for (int j = 0; j < getNumGaussians(); j++)
                    sum += gaussian[j]*w1[i*getNumGaussians()+j];
                if (getActivation() == ShiftedSoftplus)
                    y1[i] = logf(0.5f*expf(sum) + 0.5f);
                else
                    y1[i] = tanhf(sum);
            }

            // Apply the second dense layer.

            float cutoffScale = cutoffFunction(r);
            for (int i = 0; i < getWidth(); i++) {
                float sum = b2[i];
                for (int j = 0; j < getWidth(); j++)
                    sum += y1[j]*w2[i*getWidth()+j];
                y2[i] = cutoffScale*sum;
            }

            // Add it to the output.

            for (int i = 0; i < getWidth(); i++) {
                output[atom1*getWidth()+i] += y2[i]*input[atom2*getWidth()+i];
                output[atom2*getWidth()+i] += y2[i]*input[atom1*getWidth()+i];
            }
        }
    }
}

void CpuCFConv::backprop(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                         const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv) {
    // Clear the output array.

    memset(inputDeriv, 0, numAtoms*getWidth()*sizeof(float));
    memset(positionDeriv, 0, numAtoms*3*sizeof(float));

    // Backpropagate through the symmetry functions.

    const CpuCFConvNeighbors& cpuNeighbors = dynamic_cast<const CpuCFConvNeighbors&>(neighbors);
    if (getPeriodic()) {
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
void CpuCFConv::backpropImpl(const CpuCFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                             const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv) {
    vector<float> gaussian(getNumGaussians()), dGaussdR(getNumGaussians());
    vector<float> y1(getWidth()), dY1dR(getWidth());
    vector<float> y2(getWidth()), dY2dR(getWidth());
    float invBoxSize[3];
    if (PERIODIC) {
        invBoxSize[0] = 1/periodicBoxVectors[0];
        invBoxSize[1] = 1/periodicBoxVectors[4];
        invBoxSize[2] = 1/periodicBoxVectors[8];
    }

    // Clear the output arrays.

    memset(inputDeriv, 0, getNumAtoms()*getWidth()*sizeof(float));
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

            // Compute the Gaussian basis functions.

            for (int i = 0; i < getNumGaussians(); i++) {
                float x = (r-gaussianPos[i])/getGaussianWidth();
                gaussian[i] = exp(-0.5f*x*x);
                dGaussdR[i] = -x*gaussian[i]/getGaussianWidth();
            }

            // Apply the first dense layer.

            for (int i = 0; i < getWidth(); i++) {
                float sum = b1[i], dSumdR = 0;
                for (int j = 0; j < getNumGaussians(); j++) {
                    sum += gaussian[j]*w1[i*getNumGaussians()+j];
                    dSumdR += dGaussdR[j]*w1[i*getNumGaussians()+j];
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
            for (int i = 0; i < getWidth(); i++) {
                float sum = b2[i], dSumdR = 0;
                for (int j = 0; j < getWidth(); j++) {
                    sum += y1[j]*w2[i*getWidth()+j];
                    dSumdR += dY1dR[j]*w2[i*getWidth()+j];
                }
                y2[i] = cutoffScale*sum;
                dY2dR[i] = dCutoffdR*sum + cutoffScale*dSumdR;
            }

            // Add it to the output.

            for (int i = 0; i < getWidth(); i++) {
                int index1 = atom1*getWidth()+i;
                int index2 = atom2*getWidth()+i;
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

float CpuCFConv::cutoffFunction(float r) {
    return 0.5f * cosf(M_PI*r/getCutoff()) + 0.5f;
}

float CpuCFConv::cutoffDeriv(float r) {
    return -(0.5f*M_PI/getCutoff()) * sinf(M_PI*r/getCutoff());
}
