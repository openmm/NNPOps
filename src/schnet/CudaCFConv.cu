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
    CHECK_RESULT(cudaMallocManaged(&neighbors, numAtoms*numAtoms*sizeof(int2)));
    CHECK_RESULT(cudaMallocManaged(&neighborCount, sizeof(int)));
    CHECK_RESULT(cudaMallocManaged(&neighborDistances, numAtoms*numAtoms*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&neighborDeltas, numAtoms*numAtoms*sizeof(float3)));

    // We will limit how many thread blocks we try to launch based on the size of the GPU.

    int device;
    CHECK_RESULT(cudaGetDevice(&device));
    CHECK_RESULT(cudaDeviceGetAttribute(&numMultiprocessors, cudaDevAttrMultiProcessorCount, device));
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
    if (neighborDeltas != 0)
        cudaFree(neighborDeltas);
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
__global__ void buildNeighborList(int numAtoms, float cutoff, int2* neighbors, int* neighborCount, float* neighborDistances,
            float3* neighborDeltas, const float3* positions, const float* periodicBoxVectors) {
    const int warp = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int indexInWarp = threadIdx.x%32;
    const int numWarps = (gridDim.x*blockDim.x)/32;
    const int warpMask = (1<<indexInWarp)-1;
    const float3 invBoxSize = (PERIODIC ? make_float3(1/periodicBoxVectors[0], 1/periodicBoxVectors[4], 1/periodicBoxVectors[8]) : make_float3(0, 0, 0));
    const float cutoff2 = cutoff*cutoff;

    // Each warp loops over atoms.

    for (int atom1 = warp; atom1 < numAtoms; atom1 += numWarps) {
        float3 pos1 = positions[atom1];

        // The threads in the warp loop over second atoms.

        for (int atom2 = atom1+indexInWarp; atom2 < numAtoms; atom2 += 32) {
            float3 pos2 = positions[atom2];
            float3 delta;
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(pos1, pos2, delta, r2, periodicBoxVectors, invBoxSize);
            bool isNeighbor = (r2 < cutoff2 && atom1 != atom2);
            int neighborFlags = __ballot_sync(0xFFFFFFFF, isNeighbor);
            int startIndex = 0;
            if (indexInWarp == 0)
                startIndex = atomicAdd(neighborCount, __popc(neighborFlags));
            startIndex = __shfl_sync(0xFFFFFFFF, startIndex, 0);
            if (isNeighbor) {
                int index = startIndex + __popc(neighborFlags&warpMask);
                neighbors[index] = make_int2(atom1, atom2);
                neighborDistances[index] = sqrtf(r2);
                neighborDeltas[index] = delta;
            }
        }
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

    CHECK_RESULT(cudaMemsetAsync(neighborCount, 0, sizeof(int)));
    int blockSize = 192;
    int numBlocks = min(numMultiprocessors*2, getNumAtoms());
    if (getPeriodic()) {
        if (triclinic)
            buildNeighborList<true, true><<<numBlocks, blockSize>>>(getNumAtoms(), getCutoff(), neighbors, neighborCount, neighborDistances, neighborDeltas, (float3*) devicePositions, deviceBoxVectors);
        else
            buildNeighborList<true, false><<<numBlocks, blockSize>>>(getNumAtoms(), getCutoff(), neighbors, neighborCount, neighborDistances, neighborDeltas, (float3*) devicePositions, deviceBoxVectors);
    }
    else
        buildNeighborList<false, false><<<numBlocks, blockSize>>>(getNumAtoms(), getCutoff(), neighbors, neighborCount, neighborDistances, neighborDeltas, (float3*) devicePositions, deviceBoxVectors);
}

CudaCFConv::CudaCFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth,
                       ActivationFunction activation, const float* w1, const float* b1, const float* w2, const float* b2) :
            CFConv(numAtoms, width, numGaussians, cutoff, periodic, gaussianWidth, activation),
            input(0), output(0), inputDeriv(0), positionDeriv(0), w1(0), b1(0), w2(0), b2(0) {
    // Allocate memory on the device for the layer parameters.

    CHECK_RESULT(cudaMallocManaged(&this->w1, numGaussians*width*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&this->w2, width*width*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&this->b1, width*sizeof(float)));
    CHECK_RESULT(cudaMallocManaged(&this->b2, width*sizeof(float)));

    // Copy the layer parameters to device memory.  The weight matrices are stored in transposed order,
    // since that allows more efficient access in the kernels.

    CHECK_RESULT(cudaMemcpyAsync(this->b1, b1, width*sizeof(float), cudaMemcpyDefault));
    CHECK_RESULT(cudaMemcpyAsync(this->b2, b2, width*sizeof(float), cudaMemcpyDefault));
    for (int i = 0; i < numGaussians; i++)
        for (int j = 0; j < width; j++)
            this->w1[i*width+j] = w1[i+j*numGaussians];
    for (int i = 0; i < width; i++)
        for (int j = 0; j < width; j++)
            this->w2[i*width+j] = w2[i+j*width];

    // We will limit how many thread blocks we try to launch based on the size of the GPU.

    int device;
    CHECK_RESULT(cudaGetDevice(&device));
    CHECK_RESULT(cudaDeviceGetAttribute(&numMultiprocessors, cudaDevAttrMultiProcessorCount, device));
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

static __device__ float cutoffFunction(float r, float rc) {
    return 0.5f * cosf(Pi*r/rc) + 0.5f;
}

static __device__ float cutoffDeriv(float r, float rc) {
    return -(0.5f*Pi/rc) * sinf(Pi*r/rc);
}

__global__ void computeCFConv(int numAtoms, int numGaussians, int width, float cutoff, float gaussianWidth, int activation,
            const int2* neighbors, const int* neighborCount, const float* neighborDistance, const float* input,
            float* output, const float* w1, const float* b1, const float* w2, const float* b2) {
    const int warp = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int indexInWarp = threadIdx.x%32;
    const int numWarps = (gridDim.x*blockDim.x)/32;
    const int warpInBlock = threadIdx.x/32;
    const int numWarpsInBlock = blockDim.x/32;
    const int tempSize = max(numGaussians, width);
    extern __shared__ float tempArrays[];
    float* temp1 = &tempArrays[tempSize*warpInBlock];
    float* temp2 = &tempArrays[tempSize*(numWarpsInBlock+warpInBlock)];

    // Each warp loops over pairs of atoms.

    for (int pair = warp; pair < *neighborCount; pair += numWarps) {
        int atom1 = neighbors[pair].x;
        int atom2 = neighbors[pair].y;
        float r = neighborDistance[pair];

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
                sum += temp1[j]*w1[i+j*width];
            if (activation == 0)
                temp2[i] = logf(0.5f*expf(sum) + 0.5f);
            else
                temp2[i] = tanhf(sum);
        }
        __syncwarp();

        // Apply the second dense layer, storing the result in temp1.

        float cutoffScale = cutoffFunction(r, cutoff);
        for (int i = indexInWarp; i < width; i += 32) {
            float sum = b2[i];
            for (int j = 0; j < width; j++)
                sum += temp2[j]*w2[i+j*width];
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

void CudaCFConv::compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                const float* input, float* output) {
    // Get device pointers to all the data we need, copying it if necessary.

    const float* deviceInput = ensureOnDevice(input, this->input, getNumAtoms()*getWidth()*sizeof(float));
    float* deviceOutput = ensureOnDevice(output, this->output, getNumAtoms()*getWidth()*sizeof(float));

    // Clear the output array.

    CHECK_RESULT(cudaMemsetAsync(deviceOutput, 0, getNumAtoms()*getWidth()*sizeof(float)));

    // Invoke the kernel.

    const int blockSize = 512;
    const CudaCFConvNeighbors& cudaNeighbors = dynamic_cast<const CudaCFConvNeighbors&>(neighbors);
    const int numBlocks = numMultiprocessors*2;
    const int tempSize = max(getNumGaussians(), getWidth())*(blockSize/32);
    computeCFConv<<<numBlocks, blockSize, 2*tempSize*sizeof(float)>>>(getNumAtoms(), getNumGaussians(),
        getWidth(), getCutoff(), getGaussianWidth(), getActivation(), cudaNeighbors.getNeighbors(),
        cudaNeighbors.getNeighborCount(), cudaNeighbors.getNeighborDistances(), deviceInput,
        deviceOutput, w1, b1, w2, b2);

    // If necessary, copy the output.

    if (deviceOutput == this->output)
        CHECK_RESULT(cudaMemcpy(output, deviceOutput, getNumAtoms()*getWidth()*sizeof(float), cudaMemcpyDefault));
}

template <bool PERIODIC, bool TRICLINIC>
__global__ void backpropCFConv(int numAtoms, int numGaussians, int width, float cutoff, float gaussianWidth,
            int activation, const int2* neighbors, const int* neighborCount, const float3* neighborDeltas,
            const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv, const float* w1,
            const float* b1, const float* w2, const float* b2) {
    const int warp = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int indexInWarp = threadIdx.x%32;
    const int numWarps = (gridDim.x*blockDim.x)/32;
    const int warpInBlock = threadIdx.x/32;
    const int numWarpsInBlock = blockDim.x/32;
    const int tempSize = max(numGaussians, width);
    extern __shared__ float tempArrays[];
    float* temp1 = &tempArrays[tempSize*warpInBlock];
    float* temp2 = &tempArrays[tempSize*(numWarpsInBlock+warpInBlock)];
    float* dtemp1 = &tempArrays[tempSize*(2*numWarpsInBlock+warpInBlock)];
    float* dtemp2 = &tempArrays[tempSize*(3*numWarpsInBlock+warpInBlock)];

    // Each warp loops over pairs of atoms.

    for (int pair = warp; pair < *neighborCount; pair += numWarps) {
        int atom1 = neighbors[pair].x;
        int atom2 = neighbors[pair].y;
        float3 delta = neighborDeltas[pair];
        float r = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
        float rInv = 1/r;

        // Compute the Gaussian basis functions and store them in temp1.

        for (int i = indexInWarp; i < numGaussians; i += 32) {
            float gaussianPos = i*cutoff/(numGaussians-1);
            float x = (r-gaussianPos)/gaussianWidth;
            float gaussian = expf(-0.5f*x*x);
            temp1[i] = gaussian;
            dtemp1[i] = -x*gaussian/gaussianWidth;
        }
        __syncwarp();

        // Apply the first dense layer, storing the result in temp2.

        for (int i = indexInWarp; i < width; i += 32) {
            float sum = b1[i], dSumdR = 0;
            for (int j = 0; j < numGaussians; j++) {
                float w = w1[i+j*width];
                sum += temp1[j]*w;
                dSumdR += dtemp1[j]*w;
            }
            if (activation == 0) {
                float expSum = expf(sum);
                temp2[i] = logf(0.5f*expSum + 0.5f);
                dtemp2[i] = dSumdR*expSum/(expSum + 1);
            }
            else {
                float th = tanhf(sum);
                temp2[i] = th;
                dtemp2[i] = dSumdR*(1-th*th);
            }
        }
        __syncwarp();

        // Apply the second dense layer, storing the result in temp1.

        float cutoffScale = cutoffFunction(r, cutoff);
        float dCutoffdR = cutoffDeriv(r, cutoff);
        for (int i = indexInWarp; i < width; i += 32) {
            float sum = b2[i], dSumdR = 0;
            for (int j = 0; j < width; j++) {
                float w = w2[i+j*width];
                sum += temp2[j]*w;
                dSumdR += dtemp2[j]*w;
            }
            temp1[i] = cutoffScale*sum;
            dtemp1[i] = dCutoffdR*sum + cutoffScale*dSumdR;
        }
        __syncwarp();

        // Add it to the output.

        for (int i = indexInWarp; i < width; i += 32) {
            int index1 = atom1*width+i;
            int index2 = atom2*width+i;
            atomicAdd(&inputDeriv[index1], temp1[i]*outputDeriv[index2]);
            atomicAdd(&inputDeriv[index2], temp1[i]*outputDeriv[index1]);
            float scale = rInv*dtemp1[i]*(input[index2]*outputDeriv[index1] + input[index1]*outputDeriv[index2]);
            for (int offset = 16; offset > 0; offset /= 2)
                scale += __shfl_down_sync(0xFFFFFFFF, scale, offset);
            if (indexInWarp == 0) {
                float dVdX[3] = {scale*delta.x, scale*delta.y, scale*delta.z};
                for (int j = 0; j < 3; j++) {
                    atomicAdd(&positionDeriv[atom1*3+j], -dVdX[j]);
                    atomicAdd(&positionDeriv[atom2*3+j], dVdX[j]);
                }
            }
        }
    }
}


void CudaCFConv::backprop(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                        const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv) {
    // Get device pointers to all the data we need, copying it if necessary.

    const float* deviceInput = ensureOnDevice(input, this->input, getNumAtoms()*getWidth()*sizeof(float));
    const float* deviceOutputDeriv = ensureOnDevice(outputDeriv, this->output, getNumAtoms()*getWidth()*sizeof(float));
    float* deviceInputDeriv = ensureOnDevice(inputDeriv, this->inputDeriv, getNumAtoms()*getWidth()*sizeof(float));
    float* devicePositionDeriv = ensureOnDevice(positionDeriv, this->positionDeriv, getNumAtoms()*sizeof(float3));

    // Clear the output arrays.

    CHECK_RESULT(cudaMemsetAsync(deviceInputDeriv, 0, getNumAtoms()*getWidth()*sizeof(float)));
    CHECK_RESULT(cudaMemsetAsync(devicePositionDeriv, 0, getNumAtoms()*sizeof(float3)));

    // Invoke the kernel.

    const int blockSize = 512;
    const CudaCFConvNeighbors& cudaNeighbors = dynamic_cast<const CudaCFConvNeighbors&>(neighbors);
    const int numBlocks = numMultiprocessors*2;
    const int tempSize = max(getNumGaussians(), getWidth())*(blockSize/32);
    if (getPeriodic()) {
        if (neighbors.getTriclinic())
            backpropCFConv<true, true><<<numBlocks, blockSize, 4*tempSize*sizeof(float)>>>(getNumAtoms(), getNumGaussians(),
                getWidth(), getCutoff(), getGaussianWidth(), getActivation(), cudaNeighbors.getNeighbors(),
                cudaNeighbors.getNeighborCount(), cudaNeighbors.getNeighborDeltas(), deviceInput, deviceOutputDeriv,
                deviceInputDeriv, devicePositionDeriv, w1, b1, w2, b2);
        else
            backpropCFConv<true, false><<<numBlocks, blockSize, 4*tempSize*sizeof(float)>>>(getNumAtoms(), getNumGaussians(),
                getWidth(), getCutoff(), getGaussianWidth(), getActivation(), cudaNeighbors.getNeighbors(),
                cudaNeighbors.getNeighborCount(), cudaNeighbors.getNeighborDeltas(), deviceInput, deviceOutputDeriv,
                deviceInputDeriv, devicePositionDeriv, w1, b1, w2, b2);
    }
    else
        backpropCFConv<false, false><<<numBlocks, blockSize, 4*tempSize*sizeof(float)>>>(getNumAtoms(), getNumGaussians(),
            getWidth(), getCutoff(), getGaussianWidth(), getActivation(), cudaNeighbors.getNeighbors(),
            cudaNeighbors.getNeighborCount(), cudaNeighbors.getNeighborDeltas(), deviceInput, deviceOutputDeriv,
            deviceInputDeriv, devicePositionDeriv, w1, b1, w2, b2);

    // If necessary, copy the output.

    if (deviceInputDeriv == this->inputDeriv)
        CHECK_RESULT(cudaMemcpy(inputDeriv, deviceInputDeriv, getNumAtoms()*getWidth()*sizeof(float), cudaMemcpyDefault));
    if (devicePositionDeriv == this->positionDeriv)
        CHECK_RESULT(cudaMemcpy(positionDeriv, devicePositionDeriv, getNumAtoms()*sizeof(float3), cudaMemcpyDefault));
}
