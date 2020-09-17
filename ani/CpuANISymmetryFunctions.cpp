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

#include "CpuANISymmetryFunctions.h"
#include <cmath>
#include <cstring>

using namespace std;

CpuANISymmetryFunctions::CpuANISymmetryFunctions(int numAtoms, int numSpecies, float radialCutoff, float angularCutoff, bool periodic, const std::vector<int>& atomSpecies,
        const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions, bool torchani) :
           ANISymmetryFunctions(numAtoms, numSpecies, radialCutoff, angularCutoff, periodic, atomSpecies, radialFunctions, angularFunctions, torchani) {
    positions.resize(3*numAtoms);
    neighbors.resize(numAtoms);

    // There are numSpecies*(numSpecies+1)/2 copies of each angular symmetry function.  Create a table mapping from
    // the species indices of two atoms to the corresponding symmetry function index.

    angularIndex.resize(numSpecies, vector<int>(numSpecies));
    int index = 0;
    for (int i = 0; i < numSpecies; i++)
        for (int j = i; j < numSpecies; j++)
            angularIndex[i][j] = angularIndex[j][i] = index++;
}

void CpuANISymmetryFunctions::computeSymmetryFunctions(const float* positions, const float* periodicBoxVectors, float* radial, float* angular) {
    // Record the positions and periodic box vectors.

    memcpy(this->positions.data(), positions, 3*numAtoms*sizeof(float));
    if (periodic) {
        memcpy(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float));
        invBoxSize[0] = 1/this->periodicBoxVectors[0][0];
        invBoxSize[1] = 1/this->periodicBoxVectors[1][1];
        invBoxSize[2] = 1/this->periodicBoxVectors[2][2];
    }

    // Determine whether we have a rectangular or triclinic periodic box.
    
    triclinic = false;
    if (periodic)
        for (int i = 0 ; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j && this->periodicBoxVectors[i][j] != 0)
                    triclinic = true;

    // Clear the output arrays.

    memset(radial, 0, numAtoms*numSpecies*radialFunctions.size()*sizeof(float));
    memset(angular, 0, numAtoms*(numSpecies*(numSpecies+1)/2)*angularFunctions.size()*sizeof(float));

    // Compute the symmetry functions.

    if (periodic) {
        if (triclinic) {
            computeRadialFunctions<true, true>(radial);
            if (torchani)
                computeAngularFunctions<true, true, true>(angular);
            else
                computeAngularFunctions<true, true, false>(angular);
        }
        else {
            computeRadialFunctions<true, false>(radial);
            if (torchani)
                computeAngularFunctions<true, false, true>(angular);
            else
                computeAngularFunctions<true, false, false>(angular);
        }
    }
    else {
        computeRadialFunctions<false, false>(radial);
        if (torchani)
            computeAngularFunctions<false, false, true>(angular);
        else
            computeAngularFunctions<false, false, false>(angular);
    }

    // Apply the overall scale factors to the symmetry functions.

    if (torchani) {
        int numRadial = numAtoms*numSpecies*radialFunctions.size();
        for (int i = 0; i < numRadial; i++)
            radial[i] *= 0.25f;
    }
    int numAngular = numAtoms*numSpecies*(numSpecies+1)*angularFunctions.size()/2;
    for (int i = 0; i < angularFunctions.size(); i++) {
        float scale = powf(2, 1-angularFunctions[i].zeta);
        for (int j = i; j < numAngular; j += angularFunctions.size())
            angular[j] *= scale;
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuANISymmetryFunctions::computeRadialFunctions(float* radial) {
    // Loop over all pairs of atoms using an O(N^2) algorithms.  This is efficient for the small molecules
    // we are currently targeting.  If we want to scale to larger systems, a voxel based algorithm would
    // be more efficient.

    int c1 = radialFunctions.size();
    int c2 = numSpecies*c1;
    float radialCutoff2 = radialCutoff*radialCutoff;
    float angularCutoff2 = angularCutoff*angularCutoff;
    for (int atom1 = 0; atom1 < numAtoms; atom1++)
        neighbors[atom1].clear();
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int atom2 = atom1+1; atom2 < numAtoms; atom2++) {
            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2);
            if (r2 < radialCutoff2) {
                // While we're at it, build the neighbor list for angular functions.

                if (r2 < angularCutoff2) {
                    neighbors[atom1].push_back(atom2);
                    neighbors[atom2].push_back(atom1);
                }

                // Compute the symmetry functions.

                float r = sqrtf(r2);
                float cutoff = cutoffFunction(r, radialCutoff);
                for (int i = 0; i < radialFunctions.size(); i++) {
                    const RadialFunction& fn = radialFunctions[i];
                    float shifted = r-fn.rs;
                    float value = cutoff * expf(-fn.eta*shifted*shifted);
                    radial[atom1*c2 + atomSpecies[atom2]*c1 + i] += value;
                    radial[atom2*c2 + atomSpecies[atom1]*c1 + i] += value;
                }
            }
        }
    }
}

template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
void CpuANISymmetryFunctions::computeAngularFunctions(float* angular) {
    // Loop over pairs of atoms.

    int c1 = angularFunctions.size();
    int c2 = numSpecies*(numSpecies+1)*c1/2;
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int i = 0; i < neighbors[atom1].size(); i++) {
            int atom2 = neighbors[atom1][i];
            float delta_12[3];
            float r2_12;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta_12, r2_12);
            float r_12 = sqrtf(r2_12);
            float cutoff_12 = cutoffFunction(r_12, angularCutoff);

            // Loop over third atoms to compute angles.
            
            for (int j = i+1; j < neighbors[atom1].size(); j++) {
                int atom3 = neighbors[atom1][j];
                float delta_13[3];
                float r2_13;
                computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom3], delta_13, r2_13);
                float r_13 = sqrtf(r2_13);
                float cutoff_13 = cutoffFunction(r_13, angularCutoff);
                float r_mean = 0.5f*(r_12+r_13);
                float theta = computeAngle<TORCHANI>(delta_12, delta_13, r_12, r_13);
                int index = angularIndex[atomSpecies[atom2]][atomSpecies[atom3]];

                // Compute the symmetry functions.

                for (int m = 0; m < angularFunctions.size(); m++) {
                    const AngularFunction& fn = angularFunctions[m];
                    float cosTerm = powf(1 + cosf(theta - fn.thetas), fn.zeta);
                    float shifted = r_mean-fn.rs;
                    float expTerm = expf(-fn.eta*shifted*shifted);
                    float value = cutoff_12 * cutoff_13 * cosTerm * expTerm;
                    angular[atom1*c2 + index*c1 + m] += value;
                }
            }
        }
    }
}

void CpuANISymmetryFunctions::backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv) {
    // Clear the output array.

    memset(positionDeriv, 0, numAtoms*3*sizeof(float));

    // Backpropagate through the symmetry functions.

    if (periodic) {
        if (triclinic) {
            backpropRadialFunctions<true, true>(radialDeriv, positionDeriv);
            if (torchani)
                backpropAngularFunctions<true, true, true>(angularDeriv, positionDeriv);
            else
                backpropAngularFunctions<true, true, false>(angularDeriv, positionDeriv);
        }
        else {
            backpropRadialFunctions<true, false>(radialDeriv, positionDeriv);
            if (torchani)
                backpropAngularFunctions<true, false, true>(angularDeriv, positionDeriv);
            else
                backpropAngularFunctions<true, false, false>(angularDeriv, positionDeriv);
        }
    }
    else {
        backpropRadialFunctions<false, false>(radialDeriv, positionDeriv);
        if (torchani)
            backpropAngularFunctions<false, false, true>(angularDeriv, positionDeriv);
        else
            backpropAngularFunctions<false, false, false>(angularDeriv, positionDeriv);
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuANISymmetryFunctions::backpropRadialFunctions(const float* radialDeriv, float* positionDeriv) {
    int c1 = radialFunctions.size();
    int c2 = numSpecies*c1;
    float radialCutoff2 = radialCutoff*radialCutoff;
    float globalScale = (torchani ? 0.25f : 1.0f);
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int atom2 = atom1+1; atom2 < numAtoms; atom2++) {
            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2);
            if (r2 < radialCutoff2) {

                // Compute the derivatives of the symmetry functions.

                float r = sqrtf(r2);
                float rInv = 1/r;
                float cutoff = cutoffFunction(r, radialCutoff);
                float dCdR = cutoffDeriv(r, radialCutoff);
                for (int i = 0; i < radialFunctions.size(); i++) {
                    const RadialFunction& fn = radialFunctions[i];
                    float shifted = r-fn.rs;
                    float expTerm = expf(-fn.eta*shifted*shifted);
                    float dVdR = dCdR*expTerm - cutoff*2*fn.eta*shifted*expTerm;
                    float dEdV = radialDeriv[atom1*c2 + atomSpecies[atom2]*c1 + i] + radialDeriv[atom2*c2 + atomSpecies[atom1]*c1 + i];
                    float scale = globalScale * dEdV * dVdR * rInv;
                    for (int j = 0; j < 3; j++) {
                        float dVdX = scale * delta[j];
                        positionDeriv[3*atom1+j] -= dVdX;
                        positionDeriv[3*atom2+j] += dVdX;
                    }
                }
            }
        }
    }
}

template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
void CpuANISymmetryFunctions::backpropAngularFunctions(const float* angularDeriv, float* positionDeriv) {
    // Loop over pairs of atoms.

    int c1 = angularFunctions.size();
    int c2 = numSpecies*(numSpecies+1)*c1/2;
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int i = 0; i < neighbors[atom1].size(); i++) {
            int atom2 = neighbors[atom1][i];
            float delta_12[3];
            float r2_12;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta_12, r2_12);
            float r_12 = sqrtf(r2_12);
            float rInv_12 = 1/r_12;
            float cutoff_12 = cutoffFunction(r_12, angularCutoff);
            float dC12dR = cutoffDeriv(r_12, angularCutoff);

            // Loop over third atoms to compute angles.

            for (int j = i+1; j < neighbors[atom1].size(); j++) {
                int atom3 = neighbors[atom1][j];
                float delta_13[3];
                float r2_13;
                computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom3], delta_13, r2_13);
                float r_13 = sqrtf(r2_13);
                float rInv_13 = 1/r_13;
                float cutoff_13 = cutoffFunction(r_13, angularCutoff);
                float dC13dR = cutoffDeriv(r_13, angularCutoff);
                float r_mean = 0.5f*(r_12+r_13);
                float theta = computeAngle<TORCHANI>(delta_12, delta_13, r_12, r_13);
                float grad2[3], grad3[3];
                computeAngleGradients<TORCHANI>(delta_12, delta_13, r_12, r_13, grad2, grad3);
                int index = angularIndex[atomSpecies[atom2]][atomSpecies[atom3]];

                // Compute the derivatives of the symmetry functions.

                for (int m = 0; m < angularFunctions.size(); m++) {
                    const AngularFunction& fn = angularFunctions[m];
                    float cosTerm = powf(1 + cosf(theta - fn.thetas), fn.zeta);
                    float shifted = r_mean-fn.rs;
                    float expTerm = expf(-fn.eta*shifted*shifted);
                    float dExpdR = -fn.eta*shifted*expTerm;
                    float dEdV = angularDeriv[atom1*c2 + index*c1 + m];
                    float zetaScale = powf(2, 1-fn.zeta);

                    // Derivatives based on the distance between atom1 and atom2.

                    {
                        float dVdR = dC12dR*cutoff_13*cosTerm*expTerm + cutoff_12*cutoff_13*cosTerm*dExpdR;
                        float scale = zetaScale * dEdV * dVdR * rInv_12;
                        for (int k = 0; k < 3; k++) {
                            float dVdX = scale * delta_12[k];
                            positionDeriv[3*atom1+k] -= dVdX;
                            positionDeriv[3*atom2+k] += dVdX;
                        }
                    }

                    // Derivatives based on the distance between atom1 and atom3.

                    {
                        float dVdR = cutoff_12*dC13dR*cosTerm*expTerm + cutoff_12*cutoff_13*cosTerm*dExpdR;
                        float scale = zetaScale * dEdV * dVdR * rInv_13;
                        for (int k = 0; k < 3; k++) {
                            float dVdX = scale * delta_13[k];
                            positionDeriv[3*atom1+k] -= dVdX;
                            positionDeriv[3*atom3+k] += dVdX;
                        }
                    }

                    // Derivatives based on the angle.

                    {
                        float dCosdA = -fn.zeta * powf(1+cosf(theta-fn.thetas), fn.zeta-1) * sinf(theta-fn.thetas);
                        float dVdA = cutoff_12*cutoff_13*dCosdA*expTerm;
                        float scale2 = zetaScale * dEdV * dVdA;
                        float scale3 = zetaScale * dEdV * dVdA;
                        for (int k = 0; k < 3; k++) {
                            float dVdX2 = scale2*grad2[k];
                            float dVdX3 = scale3*grad3[k];
                            positionDeriv[3*atom2+k] += dVdX2;
                            positionDeriv[3*atom3+k] += dVdX3;
                            positionDeriv[3*atom1+k] -= dVdX2 + dVdX3;
                        }
                    }
                }
            }
        }
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuANISymmetryFunctions::computeDisplacement(const float* pos1, const float* pos2, float* delta, float& r2) {
    delta[0] = pos2[0]-pos1[0];
    delta[1] = pos2[1]-pos1[1];
    delta[2] = pos2[2]-pos1[2];
    if (PERIODIC) {
        if (TRICLINIC) {
            float scale3 = round(delta[2]*invBoxSize[2]);
            delta[0] -= scale3*periodicBoxVectors[2][0];
            delta[1] -= scale3*periodicBoxVectors[2][1];
            delta[2] -= scale3*periodicBoxVectors[2][2];
            float scale2 = round(delta[1]*invBoxSize[1]);
            delta[0] -= scale2*periodicBoxVectors[1][0];
            delta[1] -= scale2*periodicBoxVectors[1][1];
            float scale1 = round(delta[0]*invBoxSize[0]);
            delta[0] -= scale1*periodicBoxVectors[0][0];
        }
        else {
            delta[0] -= round(delta[0]*invBoxSize[0])*periodicBoxVectors[0][0];
            delta[1] -= round(delta[1]*invBoxSize[1])*periodicBoxVectors[1][1];
            delta[2] -= round(delta[2]*invBoxSize[2])*periodicBoxVectors[2][2];
        }
    }
    r2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
}

float CpuANISymmetryFunctions::cutoffFunction(float r, float rc) {
    return 0.5f * cosf(M_PI*r/rc) + 0.5f;
}

float CpuANISymmetryFunctions::cutoffDeriv(float r, float rc) {
    return -(0.5f*M_PI/rc) * sinf(M_PI*r/rc);
}

template <bool TORCHANI>
float CpuANISymmetryFunctions::computeAngle(const float* vec1, const float* vec2, float r1, float r2) {
    float dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    if (TORCHANI)
        dot *= 0.95f;
    float cosine = dot/(r1*r2);
    float angle;
    if (!TORCHANI && (cosine > 0.99f || cosine < -0.99f)) {
        // We're close to the singularity in acos(), so take the cross product and use asin() instead.

        float c[3];
        computeCross(vec1, vec2, c);
        angle = asinf(sqrtf(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])/(r1*r2));
        if (cosine < 0)
            angle = M_PI-angle;
    }
    else
       angle = acosf(cosine);
    return angle;
}

template <bool TORCHANI>
void CpuANISymmetryFunctions::computeAngleGradients(const float* vec1, const float* vec2, float r1, float r2, float* grad1, float* grad2) {
    float dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    float rInv1 = 1/r1;
    float rInv2 = 1/r2;
    float rInvProd = rInv1*rInv2;
    float rInv1_2 = rInv1*rInv1;
    float rInv2_2 = rInv2*rInv2;
    float dAngledDot;
    if (TORCHANI) {
        float scaledDot = 0.95f*dot*rInvProd;
        dAngledDot = -0.95f/sqrt(1-scaledDot*scaledDot);
    }
    else {
        float scaledDot = dot*rInvProd;
        dAngledDot = -1/sqrt(1-scaledDot*scaledDot);
    }
    grad1[0] = dAngledDot * rInvProd * (vec2[0] - dot*rInv1_2*vec1[0]);
    grad1[1] = dAngledDot * rInvProd * (vec2[1] - dot*rInv1_2*vec1[1]);
    grad1[2] = dAngledDot * rInvProd * (vec2[2] - dot*rInv1_2*vec1[2]);
    grad2[0] = dAngledDot * rInvProd * (vec1[0] - dot*rInv2_2*vec2[0]);
    grad2[1] = dAngledDot * rInvProd * (vec1[1] - dot*rInv2_2*vec2[1]);
    grad2[2] = dAngledDot * rInvProd * (vec1[2] - dot*rInv2_2*vec2[2]);
}

void CpuANISymmetryFunctions::computeCross(const float* vec1, const float* vec2, float* c) {
    c[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
    c[1] = vec1[2]*vec2[0]-vec1[0]*vec2[2];
    c[2] = vec1[0]*vec2[1]-vec1[1]*vec2[0];
}
