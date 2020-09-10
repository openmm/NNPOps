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
        const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions) :
           ANISymmetryFunctions(numAtoms, numSpecies, radialCutoff, angularCutoff, periodic, atomSpecies, radialFunctions, angularFunctions) {
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
    if (periodic)
        memcpy(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float));
    invBoxSize[0] = 1/this->periodicBoxVectors[0][0];
    invBoxSize[1] = 1/this->periodicBoxVectors[1][1];
    invBoxSize[2] = 1/this->periodicBoxVectors[2][2];

    // Determine whether we have a rectangular or triclinic periodic box.
    
    bool triclinic = false;
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
            computeAngularFunctions<true, true>(angular);
        }
        else {
            computeRadialFunctions<true, false>(radial);
            computeAngularFunctions<true, false>(angular);
        }
    }
    else {
        computeRadialFunctions<false, false>(radial);
        computeAngularFunctions<false, false>(angular);
    }

    // Apply the overall scale factors to the symmetry functions.

    int numRadial = numAtoms*numSpecies*radialFunctions.size();
    for (int i = 0; i < numRadial; i++)
        radial[i] *= 0.25f;
    int numAngular = numAtoms*numSpecies*(numSpecies+1)*angularFunctions.size()/2;
    for (int i = 0; i < angularFunctions.size(); i++) {
        float scale = powf(2, 1-angularFunctions[i].zeta);
        for (int j = i; j < numAngular; j += angularFunctions.size())
            angular[j] *= scale;
    }
}

void CpuANISymmetryFunctions::backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv) {

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
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        neighbors[atom1].clear();
        for (int atom2 = atom1+1; atom2 < numAtoms; atom2++) {
            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2);
            if (r2 < radialCutoff2) {
                // While we're at it, build the neighbor list for angular functions.  The neighbor list for atom1
                // only includes atoms with index > atom1, and lists them in order.

                if (r2 < angularCutoff2)
                    neighbors[atom1].push_back(atom2);

                // Compute the symmetry functions.

                for (int i = 0; i < radialFunctions.size(); i++) {
                    const RadialFunction& fn = radialFunctions[i];
                    float r = sqrtf(r2);
                    float shifted = r-fn.rs;
                    float value = cutoffFunction(r, radialCutoff) * expf(-fn.eta*shifted*shifted);
                    radial[atom1*c2 + atomSpecies[atom2]*c1 + i] += value;
                    radial[atom2*c2 + atomSpecies[atom1]*c1 + i] += value;
                }
            }
        }
    }
}

template <bool PERIODIC, bool TRICLINIC>
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
                float theta = computeAngle(delta_12, delta_13, r_12, r_13);
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

float CpuANISymmetryFunctions::computeAngle(const float* vec1, const float* vec2, float r1, float r2) {
    float dotProduct = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    float cosine = dotProduct/(r1*r2);
    float angle;
    if (cosine > 0.99f || cosine < -0.99f) {
        // We're close to the singularity in acos(), so take the cross product and use asin() instead.

        float cx = vec1[1]*vec2[2]-vec1[2]*vec2[1];
        float cy = vec1[2]*vec2[0]-vec1[0]*vec2[2];
        float cz = vec1[0]*vec2[1]-vec1[1]*vec2[0];
        angle = asinf(sqrtf(cx*cx+cy*cy+cz*cz)/(r1*r2));
        if (cosine < 0)
            angle = M_PI-angle;
    }
    else
       angle = acosf(cosine);
    return angle;
}
