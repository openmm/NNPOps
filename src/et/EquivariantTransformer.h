#ifndef EQUIVARIANT_TRANSFORMER
#define EQUIVARIANT_TRANSFORMER

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

/**
 * This class represents a neighbor list for use in computing equivariant transformers.
 * A single object can be used for all layers in the models.  Create it at the
 * same time you create the model, call build() every time the atom positions
 * change, and pass it to the methods of objects that do computations
 * based on positions.
 * 
 * This is an abstract class.  Subclasses provide implementations on particular
 * types of hardware.
 */
class EquivariantTransformerNeighbors {
public:
    /**
     * Create an object for computing neighbor lists.
     * 
     * @param numAtoms     the number of atoms in the system being modeled
     * @param cutoff       the cutoff distance
     * @param periodic     whether to use periodic boundary conditions.
     */
    EquivariantTransformerNeighbors(int numAtoms, float cutoff, bool periodic) : numAtoms(numAtoms),
            cutoff(cutoff), periodic(periodic) {
    }
    virtual ~EquivariantTransformerNeighbors() {
    }
    /**
     * Rebuild the neighbor list based on a new set of positions.
     * 
     * @param positions           an array of shape [numAtoms][3] containing the position of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     */
    virtual void build(const float* positions, const float* periodicBoxVectors) = 0;
    /**
     * Get whether the periodic box vectors specified in the most recent call to build() described
     * a triclinic (not rectangular) box.
     */
    virtual bool getTriclinic() const = 0;
    /**
     * Get the number of atoms in the system.
     */
    int getNumAtoms() const {
        return numAtoms;
    }
    /**
     * Get the cutoff distance.
     */
    float getCutoff() const {
        return cutoff;
    }
    /**
     * Get whether to apply periodic boundary conditions.
     */
    bool getPeriodic() const {
        return periodic;
    }
private:
    int numAtoms;
    float cutoff;
    bool periodic;
};

/**
 * This class computes an update layer for an equivariant transformer model. Create
 * an instance of this class at the same time you create the model and then reuse it
 * for every calculation on that model.
 * 
 * This is an abstract class.  Subclasses provide implementations on particular
 * types of hardware.
 */
class EquivariantTransformerLayer {
public:
    /**
     * Construct on object for computing an update layer.
     *
     * @param numAtoms  the number of atoms in the system
     * @param width     the number of elements in the input and output vectors
     * @param numHeads  the number of attention heads
     * @param numRBF    the number of radial basis functions
     */
    EquivariantTransformerLayer(int numAtoms, int width, int numHeads, int numRBF) :
            numAtoms(numAtoms), width(width), numHeads(numHeads), numRBF(numRBF) {
    }
    virtual ~EquivariantTransformerLayer() {
    }
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
    virtual void compute(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                         const float* x, const float* vec, float* dx, float* dvec) = 0;
    /**
     * Given the derivatives of some function E (typically energy) with respect to the outputs, backpropagate them
     * to find the derivates of E with respect to the inputs and atom positions.
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
    virtual void backprop(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                          const float* x, const float* vec, const float* dxDeriv, const float* dvecDeriv, float* xDeriv, float* vecDeriv, float* positionDeriv) = 0;
    /**
     * Get the number of atoms in the system.
     */
    int getNumAtoms() const {
        return numAtoms;
    }
    /**
     * Get the number of elements in the input and output vectors.
     */
    int getWidth() const {
        return width;
    }
    /**
     * Get the number of attention heads.
     */
    int getNumHeads() const {
        return numHeads;
    }
    /**
     * Get the number of radial basis functions.
     */
    int getNumRBF() const {
        return numRBF;
    }
protected:
    const int numAtoms, width, numHeads, numRBF;
};

#endif
