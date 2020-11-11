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

#include "CudaCFConv.h"
#include <cstring>
#include <stdexcept>

using namespace std;

#define CHECK_RESULT(result) \
if (result != cudaSuccess) { \
    throw runtime_error(string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+to_string(__LINE__));\
}

const float Pi = (float) M_PI;

CudaCFConvNeighbors::CudaCFConvNeighbors(int numAtoms, float cutoff, bool periodic) : CFConvNeighbors(numAtoms, cutoff, periodic),
                                         positions(0), periodicBoxVectors(0), neighbors(0), neighborCount(0), neighborDistances(0) {
    CHECK_RESULT(cudaMallocManaged(&positions, numAtoms*sizeof(float3)));
    CHECK_RESULT(cudaMallocManaged(&periodicBoxVectors, 9*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&neighbors, numAtoms*numAtoms*sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&neighborCount, numAtoms*sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&neighborDistances, numAtoms*numAtoms*sizeof(float)));

    // Set an upper limit on how many thread blocks we try to launch based on the size of the GPU.

    int device, numMultiprocessors;
    CHECK_RESULT(cudaGetDevice(&device));
    CHECK_RESULT(cudaDeviceGetAttribute(&numMultiprocessors, cudaDevAttrMultiProcessorCount, device));
    maxBlocks = numMultiprocessors*4;
}

CudaCFConvNeighbors::~CudaCFConvNeighbors() {
    if (positions != 0)
        cudaFree(positions);
    if (periodicBoxVectors != 0)
        cudaFree(periodicBoxVectors);
    if (neighbors != 0)
        cudaFree(neighbors);
    if (neighborCount != 0)
        cudaFree(neighborCount);
    if (neighborDistances != 0)
        cudaFree(neighborDistances);
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

template <bool PERIODIC, bool TRICLINIC>
__global__ void buildNeighborList(int numAtoms, float cutoff, int* neighbors, int* neighborCount, float* neighborDistances,
            const float3* positions, const float* periodicBoxVectors) {
    const int warp = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int indexInWarp = threadIdx.x%32;
    const int numWarps = (gridDim.x*blockDim.x)/32;
    const int warpMask = (1<<indexInWarp)-1;
    const float3 invBoxSize = (PERIODIC ? make_float3(1/periodicBoxVectors[0], 1/periodicBoxVectors[4], 1/periodicBoxVectors[8]) : make_float3(0, 0, 0));
    const float cutoff2 = cutoff*cutoff;

    // Each warp loops over atoms.

    for (int atom1 = warp; atom1 < numAtoms; atom1 += numWarps) {
        int numNeighbors = 0;
        float3 pos1 = positions[atom1];

        // The threads in the warp loop over second atoms.

        for (int atom2 = atom1+indexInWarp; atom2 < numAtoms; atom2 += 32) {
            float3 pos2 = positions[atom2];
            float3 delta;
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos2, delta, r2, periodicBoxVectors, invBoxSize);
            bool isNeighbor = (r2 < cutoff2 && atom1 != atom2);
            int neighborFlags = __ballot_sync(0xFFFFFFFF, isNeighbor);
            if (isNeighbor) {
                int index = numNeighbors + __popc(neighborFlags&warpMask);
                neighbors[atom1*numAtoms + index] = atom2;
                neighborDistances[atom1*numAtoms + index] = sqrtf(r2);
            }
            numNeighbors += __popc(neighborFlags);
        }
        if (indexInWarp == 0)
            neighborCount[atom1] = numNeighbors;
    }
}

void CudaCFConvNeighbors::build(const float* positions, const float* periodicBoxVectors) {
    // If necessary, copy the positions to the device.

    cudaPointerAttributes attrib;
    cudaError_t result = cudaPointerGetAttributes(&attrib, positions);
    if (result != cudaSuccess || attrib.devicePointer == 0) {
        CHECK_RESULT(cudaMemcpyAsync(this->positions, positions, 3*getNumAtoms()*sizeof(float), cudaMemcpyDefault));
        devicePositions = this->positions;
    }
    else
        devicePositions = (float*) attrib.devicePointer;

    // We'll need to access the box vectors on both host and device.  Figure out the most
    // efficient way of doing that.

   const float* hostBoxVectors;
    if (getPeriodic()) {
        result = cudaPointerGetAttributes(&attrib, periodicBoxVectors);
        if (result != cudaSuccess || attrib.devicePointer == 0) {
            CHECK_RESULT(cudaMemcpyAsync(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float), cudaMemcpyDefault));
            hostBoxVectors = periodicBoxVectors;
            deviceBoxVectors = this->periodicBoxVectors;
        }
        else {
            if (attrib.hostPointer == 0) {
                CHECK_RESULT(cudaMemcpy(this->periodicBoxVectors, periodicBoxVectors, 9*sizeof(float), cudaMemcpyDefault));
                hostBoxVectors = this->periodicBoxVectors;
            }
            else
                hostBoxVectors = periodicBoxVectors;
            deviceBoxVectors = (float*) attrib.devicePointer;
        }
    }

    // Determine whether we have a rectangular or triclinic periodic box.
    
    triclinic = false;
    if (getPeriodic())
        for (int i = 0 ; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j && hostBoxVectors[3*i+j] != 0)
                    triclinic = true;

    // Build the neighbor list.

    int blockSize = 192;
    int numBlocks = min(maxBlocks, getNumAtoms());
    if (getPeriodic()) {
        if (triclinic)
            buildNeighborList<true, true><<<numBlocks, blockSize>>>(getNumAtoms(), getCutoff(), neighbors, neighborCount, neighborDistances, (float3*) devicePositions, deviceBoxVectors);
        else
            buildNeighborList<true, false><<<numBlocks, blockSize>>>(getNumAtoms(), getCutoff(), neighbors, neighborCount, neighborDistances, (float3*) devicePositions, deviceBoxVectors);
    }
    else
        buildNeighborList<false, false><<<numBlocks, blockSize>>>(getNumAtoms(), getCutoff(), neighbors, neighborCount, neighborDistances, (float3*) devicePositions, deviceBoxVectors);
}

CudaCFConv::CudaCFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth) :
                       CFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth), input(0), output(0), inputDeriv(0),
                       positionDeriv(0), w1(0), b1(0), w2(0), b2(0) {
}

CudaCFConv::~CudaCFConv() {
    if (input != 0)
        cudaFree(input);
    if (output != 0)
        cudaFree(output);
    if (inputDeriv != 0)
        cudaFree(inputDeriv);
    if (positionDeriv != 0)
        cudaFree(positionDeriv);
    if (w1 != 0)
        cudaFree(w1);
    if (b1 != 0)
        cudaFree(b1);
    if (w2 != 0)
        cudaFree(w2);
    if (b2 != 0)
        cudaFree(b2);
}

float* CudaCFConv::ensureOnDevice(float* arg, float*& deviceMemory, int size) {
    cudaPointerAttributes attrib;
    cudaError_t result = cudaPointerGetAttributes(&attrib, arg);
    if (result != cudaSuccess || attrib.devicePointer == 0) {
        if (deviceMemory == 0)
            CHECK_RESULT(cudaMallocManaged(&deviceMemory, size));
        CHECK_RESULT(cudaMemcpyAsync(deviceMemory, arg, size, cudaMemcpyDefault));
        return deviceMemory;
    }
    return (float*) attrib.devicePointer;
}

const float* CudaCFConv::ensureOnDevice(const float* arg, float*& deviceMemory, int size) {
    cudaPointerAttributes attrib;
    cudaError_t result = cudaPointerGetAttributes(&attrib, arg);
    if (result != cudaSuccess || attrib.devicePointer == 0) {
        if (deviceMemory == 0)
            CHECK_RESULT(cudaMallocManaged(&deviceMemory, size));
        CHECK_RESULT(cudaMemcpyAsync(deviceMemory, arg, size, cudaMemcpyDefault));
        return deviceMemory;
    }
    return (const float*) attrib.devicePointer;
}

__device__ float cutoffFunction(float r, float rc) {
    return 0.5f * cosf(Pi*r/rc) + 0.5f;
}

__device__ float cutoffDeriv(float r, float rc) {
    return -(0.5f*Pi/rc) * sinf(Pi*r/rc);
}

__global__ void computeCFConv(int numAtoms, int numGaussians, int width, float cutoff, float gaussianWidth,
            const int* neighbors, const int* neighborCount, const float* neighborDistance, const float* input,
            float* output, const float* w1, const float* b1, const float* w2, const float* b2) {
    const int warp = threadIdx.x/32;
    const int indexInWarp = threadIdx.x%32;
    const int numWarps = blockDim.x/32;
    const int tempSize = max(numGaussians, width);
    extern __shared__ float tempArrays[];
    float* temp1 = &tempArrays[tempSize*warp];
    float* temp2 = &tempArrays[tempSize*(numWarps+warp)];

    // Each thread block loops over atoms.

    for (int atom1 = blockIdx.x; atom1 < numAtoms; atom1 += gridDim.x) {
        int numNeighbors = neighborCount[atom1];

        // The warps in the block loop over atoms from the neighbor list.

        for (int neighborIndex = warp; neighborIndex < numNeighbors; neighborIndex += numWarps) {
            int atom2 = neighbors[atom1*numAtoms + neighborIndex];
            float r = neighborDistance[atom1*numAtoms + neighborIndex];

            // Compute the Gaussian basis functions and store them in temp1.

            for (int i = indexInWarp; i < numGaussians; i += 32) {
                float gaussianPos = i*cutoff/(numGaussians-1);
                float x = (r-gaussianPos)/gaussianWidth;
                temp1[i] = expf(-0.5f*x*x);
            }
            __syncwarp();

            // Apply the first dense layer, storing the result in temp2.

            for (int i = indexInWarp; i < width; i += 32) {
                float sum = b1[i];
                for (int j = 0; j < numGaussians; j++)
                    sum += temp1[j]*w1[i*numGaussians+j];
                temp2[i] = logf(0.5f*expf(sum) + 0.5f);
            }
            __syncwarp();

            // Apply the second dense layer, storing the result in temp1.

            float cutoffScale = cutoffFunction(r, cutoff);
            for (int i = indexInWarp; i < width; i += 32) {
                float sum = b2[i];
                for (int j = 0; j < width; j++)
                    sum += temp2[j]*w2[i*width+j];
                temp1[i] = cutoffScale*sum;
            }
            __syncwarp();

            // Add it to the output.

            for (int i = indexInWarp; i < width; i += 32) {
                atomicAdd(&output[atom1*width+i], temp1[i]*input[atom2*width+i]);
                atomicAdd(&output[atom2*width+i], temp1[i]*input[atom1*width+i]);
            }
        }
    }
}

void CudaCFConv::compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                const float* input, float* output, const float* w1, const float* b1, const float* w2, const float* b2) {
    // Get device pointers to all the data we need, copying it if necessary.

    const float* deviceInput = ensureOnDevice(input, this->input, getNumAtoms()*getWidth()*sizeof(float));
    float* deviceOutput = ensureOnDevice(output, this->output, getNumAtoms()*getWidth()*sizeof(float));
    const float* deviceW1 = ensureOnDevice(w1, this->w1, getNumGaussians()*getWidth()*sizeof(float));
    const float* deviceB1 = ensureOnDevice(b1, this->b1, getWidth()*sizeof(float));
    const float* deviceW2 = ensureOnDevice(w2, this->w2, getWidth()*getWidth()*sizeof(float));
    const float* deviceB2 = ensureOnDevice(b2, this->b2, getWidth()*sizeof(float));

    // Invoke the kernel.

    const int blockSize = 192;
    const CudaCFConvNeighbors& cudaNeighbors = dynamic_cast<const CudaCFConvNeighbors&>(neighbors);
    const int numBlocks = min(cudaNeighbors.getMaxBlocks(), getNumAtoms());
    const int tempSize = max(getNumGaussians(), getWidth())*(blockSize/32);
    computeCFConv<<<numBlocks, blockSize, 2*tempSize*sizeof(float)>>>(getNumAtoms(), getNumGaussians(),
        getWidth(), getCutoff(), getGaussianWidth(), cudaNeighbors.getNeighbors(), cudaNeighbors.getNeighborCount(),
        cudaNeighbors.getNeighborDistances(), deviceInput, deviceOutput, deviceW1, deviceB1, deviceW2, deviceB2);

    // If necessary, copy the output.

    if (deviceOutput == this->output)
        CHECK_RESULT(cudaMemcpy(output, deviceOutput, getNumAtoms()*getWidth()*sizeof(float), cudaMemcpyDefault));
}

void CudaCFConv::backprop(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                        const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv,
                        const float* w1, const float* b1, const float* w2, const float* b2) {
//     // Clear the output array.

//     memset(inputDeriv, 0, numAtoms*getWidth()*sizeof(float));
//     memset(positionDeriv, 0, numAtoms*3*sizeof(float));

//     // Backpropagate through the symmetry functions.

//     const CudaCFConvNeighbors& CudaNeighbors = dynamic_cast<const CudaCFConvNeighbors&>(neighbors);
//     if (getPeriodic()) {
//         if (neighbors.getTriclinic())
//             backpropImpl<true, true>(CudaNeighbors, positions, periodicBoxVectors, input, outputDeriv, inputDeriv, positionDeriv, w1, b1, w2, b2);
//         else
//             backpropImpl<true, false>(CudaNeighbors, positions, periodicBoxVectors, input, outputDeriv, inputDeriv, positionDeriv, w1, b1, w2, b2);
//     }
//     else {
//         backpropImpl<false, false>(CudaNeighbors, positions, periodicBoxVectors, input, outputDeriv, inputDeriv, positionDeriv, w1, b1, w2, b2);
//     }
}

// template <bool PERIODIC, bool TRICLINIC>
// void CudaCFConv::backpropImpl(const CudaCFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
//                             const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv,
//                             const float* w1, const float* b1, const float* w2, const float* b2) {
//     vector<float> gaussian(getNumGaussians()), dGaussdR(getNumGaussians());
//     vector<float> y1(getWidth()), dY1dR(getWidth());
//     vector<float> y2(getWidth()), dY2dR(getWidth());
//     float invBoxSize[3];
//     if (PERIODIC) {
//         invBoxSize[0] = 1/periodicBoxVectors[0];
//         invBoxSize[1] = 1/periodicBoxVectors[4];
//         invBoxSize[2] = 1/periodicBoxVectors[8];
//     }

//     // Clear the output arrays.

//     memset(inputDeriv, 0, getNumAtoms()*getWidth()*sizeof(float));
//     memset(positionDeriv, 0, getNumAtoms()*3*sizeof(float));

//     // Loop over pairs of atoms from the neighbor list.

//     for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
//         for (int atom2 : neighbors.getNeighbors()[atom1]) {
//             // Compute the displacement.

//             float delta[3];
//             float r2;
//             computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2, periodicBoxVectors, invBoxSize);
//             float r = sqrtf(r2);
//             float rInv = 1/r;

//             // Compute the Gaussian basis functions.

//             for (int i = 0; i < getNumGaussians(); i++) {
//                 float x = (r-gaussianPos[i])/getGaussianWidth();
//                 gaussian[i] = exp(-0.5f*x*x);
//                 dGaussdR[i] = -x*gaussian[i]/getGaussianWidth();
//             }

//             // Apply the first dense layer.

//             for (int i = 0; i < getWidth(); i++) {
//                 float sum = b1[i], dSumdR = 0;
//                 for (int j = 0; j < getNumGaussians(); j++) {
//                     sum += gaussian[j]*w1[i*getNumGaussians()+j];
//                     dSumdR += dGaussdR[j]*w1[i*getNumGaussians()+j];
//                 }
//                 float expSum = expf(sum);
//                 y1[i] = logf(0.5f*expSum + 0.5f);
//                 dY1dR[i] = dSumdR*expSum/(expSum + 1);
//             }

//             // Apply the second dense layer.

//             float cutoffScale = cutoffFunction(r);
//             float dCutoffdR = cutoffDeriv(r);
//             for (int i = 0; i < getWidth(); i++) {
//                 float sum = b2[i], dSumdR = 0;
//                 for (int j = 0; j < getWidth(); j++) {
//                     sum += y1[j]*w2[i*getWidth()+j];
//                     dSumdR += dY1dR[j]*w2[i*getWidth()+j];
//                 }
//                 y2[i] = cutoffScale*sum;
//                 dY2dR[i] = dCutoffdR*sum + cutoffScale*dSumdR;
//             }

//             // Add it to the output.

//             for (int i = 0; i < getWidth(); i++) {
//                 int index1 = atom1*getWidth()+i;
//                 int index2 = atom2*getWidth()+i;
//                 inputDeriv[index1] += y2[i]*outputDeriv[index2];
//                 inputDeriv[index2] += y2[i]*outputDeriv[index1];
//                 float scale = rInv*dY2dR[i]*(input[index2]*outputDeriv[index1] + input[index1]*outputDeriv[index2]);
//                 for (int j = 0; j < 3; j++) {
//                     float dVdX = scale * delta[j];
//                     positionDeriv[atom1*3+j] -= dVdX;
//                     positionDeriv[atom2*3+j] += dVdX;
//                 }
//             }
//         }
//     }
// }
