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

const float Pi = (float) M_PI;

CudaANISymmetryFunctions::CudaANISymmetryFunctions(int numAtoms, int numSpecies, float radialCutoff, float angularCutoff, bool periodic, const std::vector<int>& atomSpecies,
        const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions, bool torchani) :
           ANISymmetryFunctions(numAtoms, numSpecies, radialCutoff, angularCutoff, periodic, atomSpecies, radialFunctions, angularFunctions, torchani),
           positions(0), neighbors(0), neighborCount(0), periodicBoxVectors(0), angularIndex(0), atomSpeciesArray(0), radialFunctionArray(0), angularFunctionArray(0),
           radialValues(0), angularValues(0), positionDerivValues(0) {
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
    CHECK_RESULT(cudaMallocManaged(&positionDerivValues, numAtoms*sizeof(float3)));
    CHECK_RESULT(cudaMemcpyAsync(atomSpeciesArray, atomSpecies.data(), atomSpecies.size()*sizeof(int), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(radialFunctionArray, radialFunctions.data(), radialFunctions.size()*sizeof(RadialFunction), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(angularFunctionArray, angularFunctions.data(), angularFunctions.size()*sizeof(AngularFunction), cudaMemcpyDefault));

    // There are numSpecies*(numSpecies+1)/2 copies of each angular symmetry function.  Create a table mapping from
    // the species indices of two atoms to the corresponding symmetry function index.

    int index = 0;
    for (int i = 0; i < numSpecies; i++)
        for (int j = i; j < numSpecies; j++)
            angularIndex[i*numSpecies+j] = angularIndex[j*numSpecies+i] = index++;

    // Set an upper limit on how many thread blocks we try to launch based on the size of the GPU.

    int device, numMultiprocessors;
    CHECK_RESULT(cudaGetDevice(&device));
    CHECK_RESULT(cudaDeviceGetAttribute(&numMultiprocessors, cudaDevAttrMultiProcessorCount, device));
    maxBlocks = numMultiprocessors*4;
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
    if (positionDerivValues != 0)
        cudaFree(positionDerivValues);
}

template <bool PERIODIC, bool TRICLINIC>
__device__ void computeDisplacement(float3 pos1, float3 pos2, float3& delta, float& r2, const float* periodicBoxVectors, float3 invBoxSize) {
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
    return 0.5f * cosf(Pi*r/rc) + 0.5f;
}

__device__ float cutoffDeriv(float r, float rc) {
    return -(0.5f*Pi/rc) * sinf(Pi*r/rc);
}

template <bool TORCHANI>
__device__ float computeAngle(float3 vec1, float3 vec2, float r1, float r2) {
    float dot = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
    if (TORCHANI)
        dot *= 0.95f;
    float cosine = dot/(r1*r2);
    float angle;
    if (!TORCHANI && (cosine > 0.99f || cosine < -0.99f)) {
        // We're close to the singularity in acos(), so take the cross product and use asin() instead.

        float3 c = make_float3(vec1.y*vec2.z-vec1.z*vec2.y, vec1.z*vec2.x-vec1.x*vec2.z, vec1.x*vec2.y-vec1.y*vec2.x);
        angle = asinf(sqrtf(c.x*c.x+c.y*c.y+c.z*c.z)/(r1*r2));
        if (cosine < 0)
            angle = Pi-angle;
    }
    else
       angle = acosf(cosine);
    return angle;
}

template <bool TORCHANI>
__device__ void computeAngleGradients(float3 vec1, float3 vec2, float r1, float r2, float3& grad1, float3& grad2) {
    float dot = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
    float rInv1 = 1/r1;
    float rInv2 = 1/r2;
    float rInvProd = rInv1*rInv2;
    float rInv1_2 = rInv1*rInv1;
    float rInv2_2 = rInv2*rInv2;
    float dAngledDot;
    if (TORCHANI) {
        float scaledDot = 0.95f*dot*rInvProd;
        dAngledDot = -0.95f/sqrtf(1-scaledDot*scaledDot);
    }
    else {
        float scaledDot = dot*rInvProd;
        dAngledDot = -1/sqrtf(1-scaledDot*scaledDot);
    }
    grad1.x = dAngledDot * rInvProd * (vec2.x - dot*rInv1_2*vec1.x);
    grad1.y = dAngledDot * rInvProd * (vec2.y - dot*rInv1_2*vec1.y);
    grad1.z = dAngledDot * rInvProd * (vec2.z - dot*rInv1_2*vec1.z);
    grad2.x = dAngledDot * rInvProd * (vec1.x - dot*rInv2_2*vec2.x);
    grad2.y = dAngledDot * rInvProd * (vec1.y - dot*rInv2_2*vec2.y);
    grad2.z = dAngledDot * rInvProd * (vec1.z - dot*rInv2_2*vec2.z);
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

        // The threads in the warp loop over second atoms.

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
        }
        if (indexInWarp == 0)
            neighborCount[atom1] = numNeighbors;
    }
}

template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
__global__ void computeAngularFunctions(int numAtoms, int numSpecies, int numAngular, float angularCutoff, float* angular,
            int* neighbors, int* neighborCount, const float3* positions, const float* periodicBoxVectors,
            const AngularFunction* angularFunctions, const int* atomSpecies, const int* angularIndex) {
    const float3 invBoxSize = (PERIODIC ? make_float3(1/periodicBoxVectors[0], 1/periodicBoxVectors[4], 1/periodicBoxVectors[8]) : make_float3(0, 0, 0));
    const int c1 = numAngular;
    const int c2 = numSpecies*(numSpecies+1)*c1/2;

    // Each thread block loops over atoms.

    for (int atom1 = blockIdx.x; atom1 < numAtoms; atom1 += gridDim.x) {
        float3 pos1 = positions[atom1];
        int numNeighbors = neighborCount[atom1];

        // The threads in the block loop over pairs of atoms from the neighbor list.

        int numPairs = numNeighbors*(numNeighbors-1)/2;
        for (int pair = threadIdx.x; pair < numPairs; pair += blockDim.x) {
            int i = (int) floorf(numNeighbors-0.5f-sqrtf((numNeighbors-0.5f)*(numNeighbors-0.5f)-2*pair));
            int j = pair - i*numNeighbors + (i+1)*(i+2)/2;
            int atom2 = neighbors[atom1*numAtoms + i];
            int atom3 = neighbors[atom1*numAtoms + j];
            float3 pos2 = positions[atom2];
            float3 pos3 = positions[atom3];
            float3 delta_12, delta_13;
            float r2_12, r2_13;
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos2, delta_12, r2_12, periodicBoxVectors, invBoxSize);
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos3, delta_13, r2_13, periodicBoxVectors, invBoxSize);
            float r_12 = sqrtf(r2_12);
            float r_13 = sqrtf(r2_13);
            float cutoff_12 = cutoffFunction(r_12, angularCutoff);
            float cutoff_13 = cutoffFunction(r_13, angularCutoff);
            float r_mean = 0.5f*(r_12+r_13);
            float theta = computeAngle<TORCHANI>(delta_12, delta_13, r_12, r_13);
            int index = angularIndex[atomSpecies[atom2]*numSpecies + atomSpecies[atom3]];

            // Compute the symmetry functions.

            for (int m = 0; m < numAngular; m++) {
                const AngularFunction fn = angularFunctions[m];
                float cosTerm = powf(1 + cosf(theta - fn.thetas), fn.zeta);
                float shifted = r_mean-fn.rs;
                float expTerm = expf(-fn.eta*shifted*shifted);
                float value = cutoff_12 * cutoff_13 * cosTerm * expTerm;
                atomicAdd(&angular[atom1*c2 + index*c1 + m], value);
            }
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
    int numAngularValues = numAtoms*numSpecies*(numSpecies+1)*numAngular/2;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numAngularValues; i += gridDim.x*blockDim.x) {
        float scale = powf(2, 1-angularFunctions[i%numAngular].zeta);
        angular[i] *= scale;
    }
}

void CudaANISymmetryFunctions::computeSymmetryFunctions(const float* positions, const float* periodicBoxVectors, float* radial, float* angular) {
    // Record the positions and periodic box vectors.

    CHECK_RESULT(cudaMemcpyAsync(this->positions, positions, 3*numAtoms*sizeof(float), cudaMemcpyDefault));
    float* hostBoxVectors;
    if (periodic) {
        // We'll need to access the box vectors on both host and device.  Figure out the most
        // efficient way of doing that.

        cudaPointerAttributes attrib;
        cudaError_t result = cudaPointerGetAttributes(&attrib, periodicBoxVectors);
        if (result != cudaSuccess || attrib.hostPointer == 0) {
            CHECK_RESULT(cudaMemcpy(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float), cudaMemcpyDefault));
            hostBoxVectors = this->periodicBoxVectors;
        }
        else {
            CHECK_RESULT(cudaMemcpyAsync(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float), cudaMemcpyDefault));
            hostBoxVectors = (float*) attrib.hostPointer;
        }
    }

    // Determine whether we have a rectangular or triclinic periodic box.
    
    triclinic = false;
    if (periodic)
        for (int i = 0 ; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j && hostBoxVectors[3*i+j] != 0)
                    triclinic = true;

    // Clear the output arrays.

    CHECK_RESULT(cudaMemsetAsync(radialValues, 0, numAtoms*numSpecies*radialFunctions.size()*sizeof(float)));
    CHECK_RESULT(cudaMemsetAsync(angularValues, 0, numAtoms*(numSpecies*(numSpecies+1)/2)*angularFunctions.size()*sizeof(float)));

    // Compute the symmetry functions.

    int blockSize = 128;
    int numBlocks = min(maxBlocks, numAtoms);
    int numRadial = radialFunctions.size();
    int numAngular = angularFunctions.size();
    if (periodic) {
        if (triclinic) {
            computeRadialFunctions<true, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, angularCutoff, radialValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, radialFunctionArray, atomSpeciesArray);
            if (torchani)
                computeAngularFunctions<true, true, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
            else
                computeAngularFunctions<true, true, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
        }
        else {
            computeRadialFunctions<true, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, angularCutoff, radialValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, radialFunctionArray, atomSpeciesArray);
            if (torchani)
                computeAngularFunctions<true, false, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
            else
                computeAngularFunctions<true, false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
        }
    }
    else {
        computeRadialFunctions<false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, angularCutoff, radialValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, radialFunctionArray, atomSpeciesArray);
        if (torchani)
            computeAngularFunctions<false, false, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
        else
            computeAngularFunctions<false, false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
    }

    // Apply the overall scale factors to the symmetry functions.

    scaleSymmetryFunctions<<<maxBlocks, 128>>>(numAtoms, numSpecies, numRadial, numAngular, torchani, radialValues, angularValues, angularFunctionArray);

    // Copy the final values to the destination memory.

    CHECK_RESULT(cudaMemcpyAsync(radial, radialValues, numAtoms*numSpecies*radialFunctions.size()*sizeof(float), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(angular, angularValues, numAtoms*(numSpecies*(numSpecies+1))*angularFunctions.size()*sizeof(float)/2, cudaMemcpyDefault));
}

template <bool PERIODIC, bool TRICLINIC>
__global__ void backpropRadialFunctions(int numAtoms, int numSpecies, int numRadial,
            float radialCutoff, const float* radialDeriv, float* positionDeriv,
            const float3* positions, const float* periodicBoxVectors,
            const RadialFunction* radialFunctions, const int* atomSpecies, bool torchani) {
    const float3 invBoxSize = (PERIODIC ? make_float3(1/periodicBoxVectors[0], 1/periodicBoxVectors[4], 1/periodicBoxVectors[8]) : make_float3(0, 0, 0));
    const int c1 = numRadial;
    const int c2 = numSpecies*c1;
    const float radialCutoff2 = radialCutoff*radialCutoff;
    const float globalScale = (torchani ? 0.25f : 1.0f);

    // Each thread block loops over atoms.

    for (int atom1 = blockIdx.x; atom1 < numAtoms; atom1 += gridDim.x) {
        float3 pos1 = positions[atom1];
        float3 posDeriv1 = make_float3(0, 0, 0);

        // The threads in the block loop over second atoms.

        for (int atom2 = atom1+1+threadIdx.x; atom2 < numAtoms; atom2 += blockDim.x) {
            float3 pos2 = positions[atom2];
            float3 delta;
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos2, delta, r2, periodicBoxVectors, invBoxSize);

            // Compute the derivatives of the symmetry functions.

            if (r2 < radialCutoff2) {
                float r = sqrtf(r2);
                float rInv = 1/r;
                float cutoff = cutoffFunction(r, radialCutoff);
                float dCdR = cutoffDeriv(r, radialCutoff);
                float3 posDeriv2 = make_float3(0, 0, 0);
                for (int i = 0; i < numRadial; i++) {
                    const RadialFunction fn = radialFunctions[i];
                    float shifted = r-fn.rs;
                    float expTerm = expf(-fn.eta*shifted*shifted);
                    float dVdR = dCdR*expTerm - cutoff*2*fn.eta*shifted*expTerm;
                    float dEdV = radialDeriv[atom1*c2 + atomSpecies[atom2]*c1 + i] + radialDeriv[atom2*c2 + atomSpecies[atom1]*c1 + i];
                    float scale = globalScale * dEdV * dVdR * rInv;
                    posDeriv1.x -= scale*delta.x;
                    posDeriv1.y -= scale*delta.y;
                    posDeriv1.z -= scale*delta.z;
                    posDeriv2.x += scale*delta.x;
                    posDeriv2.y += scale*delta.y;
                    posDeriv2.z += scale*delta.z;
                }
                atomicAdd(&positionDeriv[3*atom2], posDeriv2.x);
                atomicAdd(&positionDeriv[3*atom2+1], posDeriv2.y);
                atomicAdd(&positionDeriv[3*atom2+2], posDeriv2.z);
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            posDeriv1.x += __shfl_down_sync(0xFFFFFFFF, posDeriv1.x, offset);
            posDeriv1.y += __shfl_down_sync(0xFFFFFFFF, posDeriv1.y, offset);
            posDeriv1.z += __shfl_down_sync(0xFFFFFFFF, posDeriv1.z, offset);
        }
        if (threadIdx.x%32 == 0) {
            atomicAdd(&positionDeriv[3*atom1], posDeriv1.x);
            atomicAdd(&positionDeriv[3*atom1+1], posDeriv1.y);
            atomicAdd(&positionDeriv[3*atom1+2], posDeriv1.z);
        }
    }
}

template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
__global__ void backpropAngularFunctions(int numAtoms, int numSpecies, int numAngular, float angularCutoff,
            const float* angularDeriv, float* positionDeriv, int* neighbors, int* neighborCount,
            const float3* positions, const float* periodicBoxVectors, const AngularFunction* angularFunctions,
            const int* atomSpecies, const int* angularIndex) {
    const float3 invBoxSize = (PERIODIC ? make_float3(1/periodicBoxVectors[0], 1/periodicBoxVectors[4], 1/periodicBoxVectors[8]) : make_float3(0, 0, 0));
    const int c1 = numAngular;
    const int c2 = numSpecies*(numSpecies+1)*c1/2;

    // Each thread block loops over atoms.

    for (int atom1 = blockIdx.x; atom1 < numAtoms; atom1 += gridDim.x) {
        float3 pos1 = positions[atom1];
        float3 posDeriv1 = make_float3(0, 0, 0);
        int numNeighbors = neighborCount[atom1];

        // The threads in the block loop over pairs of atoms from the neighbor list.

        int numPairs = numNeighbors*(numNeighbors-1)/2;
        for (int pair = threadIdx.x; pair < numPairs; pair += blockDim.x) {
            int i = (int) floorf(numNeighbors-0.5f-sqrtf((numNeighbors-0.5f)*(numNeighbors-0.5f)-2*pair));
            int j = pair - i*numNeighbors + (i+1)*(i+2)/2;
            int atom2 = neighbors[atom1*numAtoms + i];
            int atom3 = neighbors[atom1*numAtoms + j];
            float3 pos2 = positions[atom2];
            float3 pos3 = positions[atom3];
            float3 posDeriv2 = make_float3(0, 0, 0);
            float3 posDeriv3 = make_float3(0, 0, 0);
            float3 delta_12, delta_13;
            float r2_12, r2_13;
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos2, delta_12, r2_12, periodicBoxVectors, invBoxSize);
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos3, delta_13, r2_13, periodicBoxVectors, invBoxSize);
            float r_12 = sqrtf(r2_12);
            float r_13 = sqrtf(r2_13);
            float rInv_12 = 1/r_12;
            float rInv_13 = 1/r_13;
            float cutoff_12 = cutoffFunction(r_12, angularCutoff);
            float cutoff_13 = cutoffFunction(r_13, angularCutoff);
            float dC12dR = cutoffDeriv(r_12, angularCutoff);
            float dC13dR = cutoffDeriv(r_13, angularCutoff);
            float r_mean = 0.5f*(r_12+r_13);
            float theta = computeAngle<TORCHANI>(delta_12, delta_13, r_12, r_13);
            float3 grad2, grad3;
            computeAngleGradients<TORCHANI>(delta_12, delta_13, r_12, r_13, grad2, grad3);
            int index = angularIndex[atomSpecies[atom2]*numSpecies + atomSpecies[atom3]];

            // Compute the derivatives of the symmetry functions.

            for (int m = 0; m < numAngular; m++) {
                const AngularFunction fn = angularFunctions[m];
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
                    float3 dVdX = make_float3(scale*delta_12.x, scale*delta_12.y, scale*delta_12.z);
                    posDeriv1.x -= dVdX.x;
                    posDeriv1.y -= dVdX.y;
                    posDeriv1.z -= dVdX.z;
                    posDeriv2.x += dVdX.x;
                    posDeriv2.y += dVdX.y;
                    posDeriv2.z += dVdX.z;
                }

                // Derivatives based on the distance between atom1 and atom3.

                {
                    float dVdR = cutoff_12*dC13dR*cosTerm*expTerm + cutoff_12*cutoff_13*cosTerm*dExpdR;
                    float scale = zetaScale * dEdV * dVdR * rInv_13;
                    float3 dVdX = make_float3(scale*delta_13.x, scale*delta_13.y, scale*delta_13.z);
                    posDeriv1.x -= dVdX.x;
                    posDeriv1.y -= dVdX.y;
                    posDeriv1.z -= dVdX.z;
                    posDeriv3.x += dVdX.x;
                    posDeriv3.y += dVdX.y;
                    posDeriv3.z += dVdX.z;
                }

                // Derivatives based on the angle.

                {
                    float dCosdA = -fn.zeta * powf(1+cosf(theta-fn.thetas), fn.zeta-1) * sinf(theta-fn.thetas);
                    float dVdA = cutoff_12*cutoff_13*dCosdA*expTerm;
                    float scale2 = zetaScale * dEdV * dVdA;
                    float scale3 = zetaScale * dEdV * dVdA;
                    float3 dVdX2 = make_float3(scale2*grad2.x, scale2*grad2.y, scale2*grad2.z);
                    float3 dVdX3 = make_float3(scale3*grad3.x, scale3*grad3.y, scale3*grad3.z);
                    posDeriv2.x += dVdX2.x;
                    posDeriv2.y += dVdX2.y;
                    posDeriv2.z += dVdX2.z;
                    posDeriv3.x += dVdX3.x;
                    posDeriv3.y += dVdX3.y;
                    posDeriv3.z += dVdX3.z;
                    posDeriv1.x -= dVdX2.x + dVdX3.x;
                    posDeriv1.y -= dVdX2.y + dVdX3.y;
                    posDeriv1.z -= dVdX2.z + dVdX3.z;
                }
            }
            atomicAdd(&positionDeriv[3*atom2], posDeriv2.x);
            atomicAdd(&positionDeriv[3*atom2+1], posDeriv2.y);
            atomicAdd(&positionDeriv[3*atom2+2], posDeriv2.z);
            atomicAdd(&positionDeriv[3*atom3], posDeriv3.x);
            atomicAdd(&positionDeriv[3*atom3+1], posDeriv3.y);
            atomicAdd(&positionDeriv[3*atom3+2], posDeriv3.z);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            posDeriv1.x += __shfl_down_sync(0xFFFFFFFF, posDeriv1.x, offset);
            posDeriv1.y += __shfl_down_sync(0xFFFFFFFF, posDeriv1.y, offset);
            posDeriv1.z += __shfl_down_sync(0xFFFFFFFF, posDeriv1.z, offset);
        }
        if (threadIdx.x%32 == 0) {
            atomicAdd(&positionDeriv[3*atom1], posDeriv1.x);
            atomicAdd(&positionDeriv[3*atom1+1], posDeriv1.y);
            atomicAdd(&positionDeriv[3*atom1+2], posDeriv1.z);
        }
    }
}

void CudaANISymmetryFunctions::backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv) {
    // Clear the output array.

    CHECK_RESULT(cudaMemcpyAsync(radialValues, radialDeriv, numAtoms*numSpecies*radialFunctions.size()*sizeof(float), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(angularValues, angularDeriv, numAtoms*(numSpecies*(numSpecies+1))*angularFunctions.size()*sizeof(float)/2, cudaMemcpyDefault));
    CHECK_RESULT(cudaMemsetAsync(positionDerivValues, 0, numAtoms*sizeof(float3)));

    // Backpropagate through the symmetry functions.

    int blockSize = 128;
    int numBlocks = min(maxBlocks, numAtoms);
    int numRadial = radialFunctions.size();
    int numAngular = angularFunctions.size();
    if (periodic) {
        if (triclinic) {
            backpropRadialFunctions<true, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, radialValues, positionDerivValues, (float3*) this->positions, this->periodicBoxVectors, radialFunctionArray, atomSpeciesArray, torchani);
            if (torchani)
                backpropAngularFunctions<true, true, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, positionDerivValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
            else
                backpropAngularFunctions<true, true, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, positionDerivValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
        }
        else {
            backpropRadialFunctions<true, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, radialValues, positionDerivValues, (float3*) this->positions, this->periodicBoxVectors, radialFunctionArray, atomSpeciesArray, torchani);
            if (torchani)
                backpropAngularFunctions<true, false, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, positionDerivValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
            else
                backpropAngularFunctions<true, false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, positionDerivValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
        }
    }
    else {
        backpropRadialFunctions<false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numRadial, radialCutoff, radialValues, positionDerivValues, (float3*) this->positions, this->periodicBoxVectors, radialFunctionArray, atomSpeciesArray, torchani);
        if (torchani)
            backpropAngularFunctions<false, false, true><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, positionDerivValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
        else
            backpropAngularFunctions<false, false, false><<<numBlocks, blockSize>>>(numAtoms, numSpecies, numAngular, angularCutoff, angularValues, positionDerivValues, neighbors, neighborCount, (float3*) this->positions, this->periodicBoxVectors, angularFunctionArray, atomSpeciesArray, angularIndex);
    }

    // Copy the final values to the destination memory.

    CHECK_RESULT(cudaMemcpyAsync(positionDeriv, positionDerivValues, numAtoms*sizeof(float3), cudaMemcpyDefault));
}

