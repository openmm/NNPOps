#ifndef CUDA_CFCONV
#define CUDA_CFCONV

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

#include "CFConv.h"

/**
 * This class represents a neighbor list for use in computing CFConv layers on a GPU.
 * A single object can be used for all layers in the models.  Create it at the
 * same time you create the model, call build() every time the atom positions
 * change, and pass it to the methods of CFConv objects that do computations
 * based on positions.
 */
class CudaCFConvNeighbors : public CFConvNeighbors {
public:
    /**
     * Create an object for computing neighbor lists on a CPU.
     * 
     * @param numAtoms    the number of atoms in the system being modeled
     * @param cutoff      the cutoff distance
     * @param periodic    whether to use periodic boundary conditions.
     */
    CudaCFConvNeighbors(int numAtoms, float cutoff, bool periodic);
    ~CudaCFConvNeighbors();
    /**
     * Rebuild the neighbor list based on a new set of positions.  All of the
     * pointers passed to this method may refer to either host or device memory.
     * 
     * @param positions           an array of shape [numAtoms][3] containing the position of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     */
    void build(const float* positions, const float* periodicBoxVectors);
    /**
     * Get whether the periodic box vectors specified in the most recent call to build() described
     * a triclinic (not rectangular) box.
     */
    bool getTriclinic() const {
        return triclinic;
    }
    /**
     * Get a pointer to device memory containing the positions passed to the most recent call
     * to build().
     */
    const float* getDevicePositions() const {
        return devicePositions;
    }
    /**
     * Get a pointer to device memory containing the box vectors passed to the most recent call
     * to build().
     */
    const float* getDeviceBoxVectors() const {
        return deviceBoxVectors;
    }
    /**
     * Get the neighbors of each atom based on the most recent call to build().
     * The neighbors of atom i begin at element i*numAtoms.
     */
    const int* getNeighbors() const {
        return neighbors;
    }
    /**
     * Get the number of neighbors of each atom based on the most recent call to build().
     */
    const int* getNeighborCount() const {
        return neighborCount;
    }
    /**
     * Get the distance between each pair of atoms.  Each element refers to the
     * corresponding element of the array returned by getNeighbors().
     */
    const float* getNeighborDistances() const {
        return neighborDistances;
    }
    /**
     * Get the maximum number of thread blocks we should try to launch on this
     * GPU for any kernel.
     */
    int getMaxBlocks() const {
        return maxBlocks;
    }
private:
    float* positions;
    float* periodicBoxVectors;
    float* devicePositions;
    float* deviceBoxVectors;
    int* neighbors;
    int* neighborCount;
    float* neighborDistances;
    bool triclinic;
    int maxBlocks;
};

/**
 * A GPU implementation of the continuous filter convolution (cfconv) function used in SchNet.
 * For every pair of atoms, it performs the following calculations:
 * 
 * 1. Compute a set of Gaussian basis functions describing the distance between them.
 * 2. Pass them through a dense layer.
 * 3. Apply a shifted softplus activation function.
 * 4. Pass the result through a second dense layer.
 * 5. Apply a cosine cutoff function to make interactions smoothly go to zero at the cutoff.
 * 
 * For each atom, the output is the sum over all neighbors of the above calculation multiplied
 * by the neighbor's input vector.
 * 
 * This calculation is designed to match the behavior of SchNetPack.  It is similar but not
 * identical to that described in the original SchNet publication.
 * 
 * Create an instance of this class at the same time you create the model and then reuse it for every
 * calculation on that model.  You also can share a single object between multiple layers if the
 * hyperparameters specified in the constructor are the same for all of them.
 */
class CudaCFConv : public CFConv {
public:
    /**
     * Construct on object for computing continuous filter convolution (cfconv) functions.
     *
     * @param numAtoms       the number of atoms in the system
     * @param width          the number of elements in the input and output vectors
     * @param numGaussians   the number of Gaussian basis functions to use, uniformly spaced between 0 and cutoff
     * @param cutoff         the cutoff distance
     * @param periodic       whether to apply periodic boundary conditions
     * @param gaussianWidth  the width of the Gaussian basis functions
     */
    CudaCFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth);
    ~CudaCFConv();
    /**
     * Compute the output of the layer.  All of the pointers passed to this method may refer to either host or device memory.
     *
     * @param neighbors           a neighbor list for accelerating the calculation.  You must have already called
     *                            build() on the neighbor list with the same positions and box vectors.
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     * @param input               an array of shape [numAtoms][width] containing the input vectors
     * @param output              an array of shape [numAtoms][width] to store the output vectors into
     * @param w1                  an array of shape [numGaussians][width] containing the weights of the first dense layer
     * @param b1                  an array of length [width] containing the biases of the first dense layer
     * @param w2                  an array of shape [width][width] containing the weights of the second dense layer
     * @param b2                  an array of length [width] containing the biases of the second dense layer
     */
    void compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                 const float* input, float* output, const float* w1, const float* b1, const float* w2, const float* b2);
    /**
     * Given the derivatives of some function E (typically energy) with respect to the outputs, backpropagate them
     * to find the derivates of E with respect to the inputs and atom positions.  All of the pointers passed to this
     * method may refer to either host or device memory.
     *
     * @param neighbors           a neighbor list for accelerating the calculation.  You must have already called
     *                            build() on the neighbor list with the same positions and box vectors.
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     * @param input               an array of shape [numAtoms][width] containing the input vectors
     * @param outputDeriv         an array of shape [numAtoms][width] containing the derivative of E with respect to each output value
     * @param inputDeriv          an array of shape [numAtoms][width] to store the derivative of E with respect to each input value into
     * @param positionDeriv       an array of shape [numAtoms][3] to store the derivative of E with respect to the atom positions into
     * @param w1                  an array of shape [numGaussians][width] containing the weights of the first dense layer
     * @param b1                  an array of length [width] containing the biases of the first dense layer
     * @param w2                  an array of shape [width][width] containing the weights of the second dense layer
     * @param b2                  an array of length [width] containing the biases of the second dense layer
     */
    void backprop(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                  const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv, const float* w1, const float* b1,
                  const float* w2, const float* b2);
private:
    /**
     * Ensure that the data for an argument is stored on the device.
     * 
     * @param arg           the argument that was passed in
     * @param deviceMemory  an array on the device to which we can copy the data, if arg
     *                      is on the host.  If the array has not yet been allocated, this
     *                      will do so.
     * @param size          the size of the data in bytes
     * @returns a pointer to device memory containing the data
     */
    float* ensureOnDevice(float* arg, float*& deviceMemory, int size);
    const float* ensureOnDevice(const float* arg, float*& deviceMemory, int size);
    float *input, *output, *inputDeriv, *positionDeriv;
    float *w1, *b1, *w2, *b2;
};

#endif
