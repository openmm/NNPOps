#ifndef CPU_EQUIVARIANT_TRANSFORMER
#define CPU_EQUIVARIANT_TRANSFORMER

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

#include "EquivariantTransformer.h"
#include <vector>

/**
 * This class represents a neighbor list for use in computing equivariant transformers on a CPU.
 * A single object can be used for all layers in the models.  Create it at the
 * same time you create the model, call build() every time the atom positions
 * change, and pass it to the methods of EquivariantTransformer objects that do computations
 * based on positions.
 */
class CpuEquivariantTransformerNeighbors : public EquivariantTransformerNeighbors {
public:
    /**
     * Create an object for computing neighbor lists on a CPU.
     * 
     * @param numAtoms     the number of atoms in the system being modeled
     * @param cutoff       the cutoff distance
     * @param periodic     whether to use periodic boundary conditions.
     */
    CpuEquivariantTransformerNeighbors(int numAtoms, float cutoff, bool periodic);
    /**
     * Rebuild the neighbor list based on a new set of positions.
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
     * Get the neighbors of each atom based on the most recent call to build().
     * Element i lists the indices of all neighbors of atom i.
     */
    const std::vector<std::vector<int> >& getNeighbors() const {
        return neighbors;
    }
    /**
     * Get the distance between each pair of atoms.  Each element refers to the
     * corresponding element of the vectors returned by getNeighbors().
     */
    const std::vector<std::vector<float> >& getNeighborDistances() const {
        return neighborDistances;
    }
private:
    template <bool PERIODIC, bool TRICLINIC>
    void findNeighbors(const float* positions, const float* periodicBoxVectors);
    std::vector<std::vector<int> > neighbors;
    std::vector<std::vector<float> > neighborDistances;
    bool triclinic;
};

/**
 * A CPU implementation of an update layer for an equivariant transformer model.
 * Create an instance of this class at the same time you create the model and then reuse it
 * for every calculation on that model.
 */
class CpuEquivariantTransformerLayer : public EquivariantTransformerLayer {
public:
    /**
     * Construct on object for computing continuous filter convolution (cfconv) functions.
     *
     * @param numAtoms  the number of atoms in the system
     * @param width     the number of elements in the input and output vectors
     * @param numHeads  the number of attention heads
     * @param numRBF    the number of radial basis functions
     * @param rbfMus    the mu parameters of the radial basis functions
     * @param rbfBetas  the beta parameters of the radial basis functions
     * @param qw        an array of shape [width][width] containing the weights of the Q transform
     * @param qb        an array of length [width] containing the biases of the Q transform
     * @param kw        an array of shape [width][width] containing the weights of the K transform
     * @param kb        an array of length [width] containing the biases of the K transform
     * @param vw        an array of shape [3*width][width] containing the weights of the V transform
     * @param vb        an array of length [3*width] containing the biases of the V transform
     * @param ow        an array of shape [3*width][width] containing the weights of the O transform
     * @param ob        an array of length [3*width] containing the biases of the O transform
     * @param uw        an array of shape [3*width][width] containing the weights of the U transform
     * @param ub        an array of length [3*width] containing the biases of the U transform
     * @param dkw       an array of shape [width][numRBF] containing the weights of the DK transform
     * @param dkb       an array of length [width] containing the biases of the DK transform
     * @param dvw       an array of shape [3*width][numRBF] containing the weights of the DV transform
     * @param dvb       an array of length [3*width] containing the biases of the DV transform
     */
    CpuEquivariantTransformerLayer(int numAtoms, int width, int numHeads, int numRBF, const float* rbfMus, const float* rbfBetas,
              const float* qw, const float* qb, const float* kw, const float* kb, const float* vw, const float* vb,
              const float* ow, const float* ob, const float* uw, const float* ub, const float* dkw, const float* dkb,
              const float* dvw, const float* dvb);
    /**
     * Compute the output of the layer.
     *
     * @param neighbors           a neighbor list for accelerating the calculation.  You must have already called
     *                            build() on the neighbor list with the same positions and box vectors.
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     * @param x                   an array of shape [numAtoms][width] containing the input scalar features
     * @param vec                 an array of shape [numAtoms][3][width] containing the input vector features
     * @param dx                  an array of shape [numAtoms][width] containing the output scalar features
     * @param dvec                an array of shape [numAtoms][3][width] containing the output vector features
     */
    void compute(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                 const float* x, const float* vec, float* dx, float* dvec);
    /**
     * Given the derivatives of some function E (typically energy) with respect to the outputs, backpropagate them
     * to find the derivates of E with respect to the inputs and atom positions.
     *
     * This method reuses values that were calculated in compute().  You must have already called compute() before
     * calling this, and identical values must have been passed for all arguments that are shared between the two methods.
     *
     * @param neighbors           a neighbor list for accelerating the calculation.  You must have already called
     *                            build() on the neighbor list with the same positions and box vectors.
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     * @param x                   an array of shape [numAtoms][width] containing the input scalar features
     * @param vec                 an array of shape [numAtoms][3][width] containing the input vector features
     * @param dxDeriv             an array of shape [numAtoms][width] containing the derivative of E with respect to the output scalar features
     * @param dvecDeriv           an array of shape [numAtoms][3][width] containing the derivative of E with respect to the output vector features
     * @param xDeriv              an array of shape [numAtoms][width] to store the derivative of E with respect to each input scalar feature into
     * @param vecDeriv            an array of shape [numAtoms][3][width] to store the derivative of E with respect to each input vector feature into
     * @param positionDeriv       an array of shape [numAtoms][3] to store the derivative of E with respect to the atom positions into
     */
    void backprop(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                  const float* x, const float* vec, const float* dxDeriv, const float* dvecDeriv, float* xDeriv, float* vecDeriv, float* positionDeriv);
private:
    template <bool PERIODIC, bool TRICLINIC>
    void backpropImpl(const CpuEquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                      const float* x, const float* vec, const float* dxDeriv, const float* dvecDeriv, float* xDeriv, float* vecDeriv, float* positionDeriv);
    /**
     * Compute the value of the cutoff function.  The implementation assumes the caller has already
     * verified that r <= cutoff.
     *
     * @param r       the distance at which it is being evaluated
     * @param cutoff  the cutoff distance
     */
    float cutoffFunction(float r, float cutoff);
    /**
     * Compute the derivative of the cutoff function.  The implementation assumes the caller has already
     * verified that r <= cutoff.
     *
     * @param r       the distance at which it is being evaluated
     * @param cutoff  the cutoff distance
     */
    float cutoffDeriv(float r, float cutoff);
    std::vector<float> rbfMus, rbfBetas, qw, qb, kw, kb, vw, vb, ow, ob, uw, ub, dkw, dkb, dvw, dvb;
    std::vector<std::vector<float> > q, k, v, o, u, s3, s;
};

#endif
