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
    if (periodic) {
        invBoxSize[0] = 1/periodicBoxVectors[0];
        invBoxSize[1] = 1/periodicBoxVectors[4];
        invBoxSize[2] = 1/periodicBoxVectors[8];
    }

    // Loop over all pairs of atoms using an O(N^2) algorithms.  This is efficient for the small molecules
    // we are currently targeting.  If we want to scale to larger systems, a voxel based algorithm would
    // be more efficient.

    float cutoff2 = getCutoff()*getCutoff();
    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        for (int atom2 = atom1+1; atom2 < getNumAtoms*(); atom2++) {
            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2, periodicBoxVectors, invBoxSize);
            if (r2 < cutoff2) {
                neighbors[atom1].push_back(atom2);
                neighbors[atom2].push_back(atom1);
                float r = sqrtf(r2);
                neighborDistances[atom1].push_back(r);
                neighborDistances[atom2].push_back(r);
            }
        }
    }
}

CpuCFConv::CpuCFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth) :
           CFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth) {
    for (int i = 0; i < numGaussians; i++)
        gaussianPos.push_back(i*cutoff/(numGaussians-1));
}

void CpuCFConv::compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                 float* input, float* output, const float* w1, const float* b1, const float* w2, const float* b2) {
    const CpuCFConvNeighbors& cpuNeighbors = dynamic_cast<const CpuCFConvNeighbors&>(neighbors);
    vector<float> gaussian(getNumGaussians());
    vector<float> y1(getWidth());
    vector<float> y2(getWidth());

    // Clear the output array.

    memset(output, 0, getNumAtoms()*getWidth()*sizeof(float));

    // Loop over pairs of atoms from the neighbor list.

    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        for (int atom2 : cpuNeighbors.getNeighbors()[atom1]) {
            float r = cpuNeighbors.getNeighborDistances()[atom1][atom2];

            // Compute the Gaussian basis functions.

            for (int i = 0; i < getNumGaussians(); i++) {
                float x = (r-gaussianPos[i])/getGaussianWidth();
                gaussian[i] = exp(-x*x);
            }

            // Apply the first dense layer.

            for (int i = 0; i < getWidth(); i++) {
                float sum = b1[i];
                for (int j = 0; j < getNumGaussians(); j++)
                    sum += gaussian[j]*w1[i*getNumGaussians()+j];
                y1[i] = logf(0.5f*expf(sum) + 0.5f);
            }

            // Apply the second dense layer.

            float cutoffScale = 1;
            if (getCutoff())
                cutoffScale = 0.5f * (cosf(r*M_PI/getCutoff()) + 1);
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

// void CpuCFConv::backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv) {
//     // Clear the output array.

//     memset(positionDeriv, 0, numAtoms*3*sizeof(float));

//     // Backpropagate through the symmetry functions.

//     if (periodic) {
//         if (triclinic) {
//             backpropRadialFunctions<true, true>(radialDeriv, positionDeriv);
//             if (torchani)
//                 backpropAngularFunctions<true, true, true>(angularDeriv, positionDeriv);
//             else
//                 backpropAngularFunctions<true, true, false>(angularDeriv, positionDeriv);
//         }
//         else {
//             backpropRadialFunctions<true, false>(radialDeriv, positionDeriv);
//             if (torchani)
//                 backpropAngularFunctions<true, false, true>(angularDeriv, positionDeriv);
//             else
//                 backpropAngularFunctions<true, false, false>(angularDeriv, positionDeriv);
//         }
//     }
//     else {
//         backpropRadialFunctions<false, false>(radialDeriv, positionDeriv);
//         if (torchani)
//             backpropAngularFunctions<false, false, true>(angularDeriv, positionDeriv);
//         else
//             backpropAngularFunctions<false, false, false>(angularDeriv, positionDeriv);
//     }
// }

// template <bool PERIODIC, bool TRICLINIC>
// void CpuCFConv::backpropRadialFunctions(const float* radialDeriv, float* positionDeriv) {
//     int c1 = radialFunctions.size();
//     int c2 = numSpecies*c1;
//     float radialCutoff2 = radialCutoff*radialCutoff;
//     float globalScale = (torchani ? 0.25f : 1.0f);
//     for (int atom1 = 0; atom1 < numAtoms; atom1++) {
//         for (int atom2 = atom1+1; atom2 < numAtoms; atom2++) {
//             float delta[3];
//             float r2;
//             computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2);
//             if (r2 < radialCutoff2) {

//                 // Compute the derivatives of the symmetry functions.

//                 float r = sqrtf(r2);
//                 float rInv = 1/r;
//                 float cutoff = cutoffFunction(r, radialCutoff);
//                 float dCdR = cutoffDeriv(r, radialCutoff);
//                 for (int i = 0; i < radialFunctions.size(); i++) {
//                     const RadialFunction& fn = radialFunctions[i];
//                     float shifted = r-fn.rs;
//                     float expTerm = expf(-fn.eta*shifted*shifted);
//                     float dVdR = dCdR*expTerm - cutoff*2*fn.eta*shifted*expTerm;
//                     float dEdV = radialDeriv[atom1*c2 + atomSpecies[atom2]*c1 + i] + radialDeriv[atom2*c2 + atomSpecies[atom1]*c1 + i];
//                     float scale = globalScale * dEdV * dVdR * rInv;
//                     for (int j = 0; j < 3; j++) {
//                         float dVdX = scale * delta[j];
//                         positionDeriv[3*atom1+j] -= dVdX;
//                         positionDeriv[3*atom2+j] += dVdX;
//                     }
//                 }
//             }
//         }
//     }
// }

// float CpuCFConv::cutoffFunction(float r, float rc) {
//     return 0.5f * cosf(M_PI*r/rc) + 0.5f;
// }

// float CpuCFConv::cutoffDeriv(float r, float rc) {
//     return -(0.5f*M_PI/rc) * sinf(M_PI*r/rc);
// }
