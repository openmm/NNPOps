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

#include "CudaANISymmetryFunctions.h"
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace std;

#define CHECK_RESULT(result) \
    if (result != cudaSuccess) { \
        throw runtime_error(string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+to_string(__LINE__));\
    }

CudaANISymmetryFunctions::CudaANISymmetryFunctions(int numAtoms, int numSpecies, float radialCutoff, float angularCutoff, bool periodic, const std::vector<int>& atomSpecies,
        const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions, bool torchani) :
           ANISymmetryFunctions(numAtoms, numSpecies, radialCutoff, angularCutoff, periodic, atomSpecies, radialFunctions, angularFunctions, torchani),
           positions(0), neighbors(0), neighborCount(0), periodicBoxVectors(0), angularIndex(0), atomSpeciesArray(0), radialFunctionArray(0), angularFunctionArray(0), radialValues(0), angularValues(0) {
    CHECK_RESULT(cudaMallocManaged(&positions, numAtoms*sizeof(float3)));
    CHECK_RESULT(cudaMallocManaged(&neighbors, numAtoms*numAtoms*sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&neighborCount, numAtoms*sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&periodicBoxVectors, 9*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&angularIndex, numSpecies*numSpecies*sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&atomSpeciesArray, numAtoms*sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&radialFunctionArray, radialFunctions.size()*sizeof(RadialFunction)));
    CHECK_RESULT(cudaMallocManaged(&angularFunctionArray, angularFunctions.size()*sizeof(AngularFunction)));
    CHECK_RESULT(cudaMallocManaged(&radialValues, numAtoms*numSpecies*radialFunctions.size()*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&angularValues, numAtoms*(numSpecies*(numSpecies+1))*angularFunctions.size()*sizeof(float)/2));
    CHECK_RESULT(cudaMemcpyAsync(atomSpeciesArray, atomSpecies.data(), atomSpecies.size()*sizeof(int), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(radialFunctionArray, radialFunctions.data(), radialFunctions.size()*sizeof(RadialFunction), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(angularFunctionArray, angularFunctions.data(), angularFunctions.size()*sizeof(AngularFunction), cudaMemcpyDefault));

    // There are numSpecies*(numSpecies+1)/2 copies of each angular symmetry function.  Create a table mapping from
    // the species indices of two atoms to the corresponding symmetry function index.

    int index = 0;
    for (int i = 0; i < numSpecies; i++)
        for (int j = i; j < numSpecies; j++)
            angularIndex[i*numSpecies+j] = angularIndex[j*numSpecies+i] = index++;
}

CudaANISymmetryFunctions::~CudaANISymmetryFunctions() {
    if (positions != 0)
        cudaFree(positions);
    if (neighbors != 0)
        cudaFree(neighbors);
    if (periodicBoxVectors != 0)
        cudaFree(periodicBoxVectors);
    if (angularIndex != 0)
        cudaFree(angularIndex);
    if (atomSpeciesArray != 0)
        cudaFree(atomSpeciesArray);
    if (radialFunctionArray != 0)
        cudaFree(radialFunctionArray);
    if (angularFunctionArray != 0)
        cudaFree(angularFunctionArray);
    if (radialValues != 0)
        cudaFree(radialValues);
    if (angularValues != 0)
        cudaFree(angularValues);
}

template <bool PERIODIC, bool TRICLINIC>
__device__ void computeDisplacement(const float3 pos1, const float3 pos2, float3& delta, float& r2, const float* periodicBoxVectors, float3 invBoxSize) {
    delta.x = pos2.x-pos1.x;
    delta.y = pos2.y-pos1.y;
    delta.z = pos2.z-pos1.z;
    if (PERIODIC) {
        if (TRICLINIC) {
            float scale3 = roundf(delta.z*invBoxSize.z);
            delta.x -= scale3*periodicBoxVectors[2*3+0];
            delta.y -= scale3*periodicBoxVectors[2*3+1];
            delta.z -= scale3*periodicBoxVectors[2*3+2];
            float scale2 = roundf(delta.y*invBoxSize.y);
            delta.x -= scale2*periodicBoxVectors[1*3+0];
            delta.y -= scale2*periodicBoxVectors[1*3+1];
            float scale1 = roundf(delta.x*invBoxSize.x);
            delta.x -= scale1*periodicBoxVectors[0*3+0];
        }
        else {
            delta.x -= roundf(delta.x*invBoxSize.x)*periodicBoxVectors[0*3+0];
            delta.y -= roundf(delta.y*invBoxSize.y)*periodicBoxVectors[1*3+1];
            delta.z -= roundf(delta.z*invBoxSize.z)*periodicBoxVectors[2*3+2];
        }
    }
    r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
}

__device__ float cutoffFunction(float r, float rc) {
    return 0.5f * cosf(M_PI*r/rc) + 0.5f;
}

__device__ float cutoffDeriv(float r, float rc) {
    return -(0.5f*M_PI/rc) * sinf(M_PI*r/rc);
}

template <bool PERIODIC, bool TRICLINIC>
__global__ void computeRadialFunctions(int numAtoms, int numSpecies, int numRadial,
            float radialCutoff, float angularCutoff, float* radial, int* neighbors,
            int* neighborCount, const float3* positions, const float* periodicBoxVectors,
            const RadialFunction* radialFunctions, const int* atomSpecies) {
    const int warp = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int indexInWarp = threadIdx.x%32;
    const int numWarps = (gridDim.x*blockDim.x)/32;
    const int warpMask = (1<<indexInWarp)-1;
    const float3 invBoxSize = (PERIODIC ? make_float3(1/periodicBoxVectors[0], 1/periodicBoxVectors[4], 1/periodicBoxVectors[8]) : make_float3(0, 0, 0));
    const int c1 = numRadial;
    const int c2 = numSpecies*c1;
    const float radialCutoff2 = radialCutoff*radialCutoff;
    const float angularCutoff2 = angularCutoff*angularCutoff;

    // Each warp loops over atoms.

    for (int atom1 = warp; atom1 < numAtoms; atom1 += numWarps) {
        int numNeighbors = 0;
        float3 pos1 = positions[atom1];
        for (int atom2 = indexInWarp; atom2 < numAtoms; atom2 += 32) {
            float3 pos2 = positions[atom2];
            float3 delta;
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos2, delta, r2, periodicBoxVectors, invBoxSize);
            if (r2 < radialCutoff2 && atom1 != atom2) {
                // Compute the symmetry functions.

                float r = sqrtf(r2);
                float cutoff = cutoffFunction(r, radialCutoff);
                for (int i = 0; i < numRadial; i++) {
                    const RadialFunction fn = radialFunctions[i];
                    float shifted = r-fn.rs;
                    float value = cutoff * expf(-fn.eta*shifted*shifted);
                    atomicAdd(&radial[atom1*c2 + atomSpecies[atom2]*c1 + i], value);
                }
            }

            // While we're at it, build the neighbor list for angular functions.

            bool isNeighbor = (r2 < angularCutoff2 && atom1 != atom2);
            int neighborFlags = __ballot_sync(0xFFFFFFFF, isNeighbor);
            if (isNeighbor) {
                int index = numNeighbors + __popc(neighborFlags&warpMask);
                neighbors[atom1*numAtoms + index] = atom2;
            }
            numNeighbors += __popc(neighborFlags);
            if (indexInWarp == 0)
                neighborCount[atom1] = numNeighbors;
        }
    }
}

__global__ void scaleSymmetryFunctions(int numAtoms, int numSpecies, int numRadial, int numAngular, bool torchani,
            float* radial, float* angular, const AngularFunction* angularFunctions) {
    if (torchani) {
        int numRadialValues = numAtoms*numSpecies*numRadial;
        for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numRadialValues; i += gridDim.x*blockDim.x)
            radial[i] *= 0.25f;
    }
//    int numAngularValues = numAtoms*numSpecies*(numSpecies+1)*angularFunctions.size()/2;
//    for (int i = 0; i < angularFunctions.size(); i++) {
//        float scale = powf(2, 1-angularFunctions[i].zeta);
//        for (int j = i; j < numAngularValues; j += angularFunctions.size())
//            angular[j] *= scale;
//    }
}

void CudaANISymmetryFunctions::computeSymmetryFunctions(const float* positions, const float* periodicBoxVectors, float* radial, float* angular) {
    // Record the positions and periodic box vectors.

    CHECK_RESULT(cudaMemcpyAsync(this->positions, positions, 3*numAtoms*sizeof(float), cudaMemcpyDefault));
    if (periodic)
        CHECK_RESULT(cudaMemcpyAsync(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float), cudaMemcpyDefault));

    // Determine whether we have a rectangular or triclinic periodic box.
    
    triclinic = false;
    if (periodic)
        for (int i = 0 ; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j && periodicBoxVectors[3*i+j] != 0)
                    triclinic = true;

    // Clear the output arrays.

    CHECK_RESULT(cudaMemsetAsync(radialValues, 0, numAtoms*numSpecies*radialFunctions.size()*sizeof(float)));
    CHECK_RESULT(cudaMemsetAsync(angularValues, 0, numAtoms*(numSpecies*(numSpecies+1)/2)*angularFunctions.size()*sizeof(float)));

    // Compute the symmetry functions.

    int blockSize = 128;
    int numBlocks = (int) ceil(numAtoms/4.0);
    int numRadial = radialFunctions.size();
    int numAngular = angularFunctions.size();
    if (periodic) {
        if (triclinic) {
            computeRadialFunctions<true, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, angularCutoff, radialValues, neighbors, neighborCount, (float3*) this->positions, periodicBoxVectors, radialFunctionArray, atomSpeciesArray);
//            if (torchani)
//                computeAngularFunctions<true, true, true>(angular);
//            else
//                computeAngularFunctions<true, true, false>(angular);
        }
        else {
            computeRadialFunctions<true, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, angularCutoff, radialValues, neighbors, neighborCount, (float3*) this->positions, periodicBoxVectors, radialFunctionArray, atomSpeciesArray);
//            if (torchani)
//                computeAngularFunctions<true, false, true>(angular);
//            else
//                computeAngularFunctions<true, false, false>(angular);
        }
    }
    else {
        computeRadialFunctions<false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, angularCutoff, radialValues, neighbors, neighborCount, (float3*) this->positions, periodicBoxVectors, radialFunctionArray, atomSpeciesArray);
//        if (torchani)
//            computeAngularFunctions<false, false, true>(angular);
//        else
//            computeAngularFunctions<false, false, false>(angular);
    }
    //CHECK_RESULT(cudaDeviceSynchronize());

    // Apply the overall scale factors to the symmetry functions.
/*
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
    }*/
    scaleSymmetryFunctions<<<(int) ceil(numAtoms/128.0), 128>>>(numAtoms, numSpecies, numRadial, numAngular, torchani, radialValues, angularValues, angularFunctionArray);
    //CHECK_RESULT(cudaDeviceSynchronize());

    // Copy the final values to the destination memory.

    CHECK_RESULT(cudaMemcpyAsync(radial, radialValues, numAtoms*numSpecies*radialFunctions.size()*sizeof(float), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(angular, angularValues, numAtoms*(numSpecies*(numSpecies+1))*angularFunctions.size()*sizeof(float)/2, cudaMemcpyDefault));
}
/*
template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
void CudaANISymmetryFunctions::computeAngularFunctions(float* angular) {
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
*/

void CudaANISymmetryFunctions::backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv) {
/*
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
    }*/
}

/*
template <bool PERIODIC, bool TRICLINIC>
void CudaANISymmetryFunctions::backpropRadialFunctions(const float* radialDeriv, float* positionDeriv) {
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
void CudaANISymmetryFunctions::backpropAngularFunctions(const float* angularDeriv, float* positionDeriv) {
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

template <bool TORCHANI>
float CudaANISymmetryFunctions::computeAngle(const float* vec1, const float* vec2, float r1, float r2) {
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
void CudaANISymmetryFunctions::computeAngleGradients(const float* vec1, const float* vec2, float r1, float r2, float* grad1, float* grad2) {
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

void CudaANISymmetryFunctions::computeCross(const float* vec1, const float* vec2, float* c) {
    c[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
    c[1] = vec1[2]*vec2[0]-vec1[0]*vec2[2];
    c[2] = vec1[0]*vec2[1]-vec1[1]*vec2[0];
}
*/
