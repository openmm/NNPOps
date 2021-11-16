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
     * @param lowerCutoff  the lower cutoff distance
     * @param upperCutoff  the upper cutoff distance
     * @param periodic     whether to use periodic boundary conditions.
     */
    EquivariantTransformerNeighbors(int numAtoms, float lowerCutoff, float upperCutoff, bool periodic) : numAtoms(numAtoms),
            lowerCutoff(lowerCutoff), upperCutoff(upperCutoff), periodic(periodic) {
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
     * Get the lower cutoff distance.
     */
    float getLowerCutoff() const {
        return lowerCutoff;
    }
    /**
     * Get the upper cutoff distance.
     */
    float getUpperCutoff() const {
        return upperCutoff;
    }
    /**
     * Get whether to apply periodic boundary conditions.
     */
    bool getPeriodic() const {
        return periodic;
    }
private:
    int numAtoms;
    float lowerCutoff, upperCutoff;
    bool periodic;
};

/**
 * This class computes an update layer for an equivariant transformer model. Create
 * an instance of this class at the same time you create the model and then reuse it
 * for every calculation on that model.
 * 
 * For each pair of atoms, it performs the following calculations:
 * 
 * 1. Compute a set of Gaussian basis functions describing the distance between them.
 * 2. Pass them through a dense layer.
 * 3. Apply an activation function.
 * 4. Pass the result through a second dense layer.
 * 5. Apply a cosine cutoff function to make interactions smoothly go to zero at the cutoff.
 * 
 * For each atom, the output is the sum over all neighbors of the above calculation multiplied
 * by the neighbor's input vector.
 * 
 * This calculation is designed to match the behavior of SchNetPack.  It is similar but not
 * identical to that described in the original SchNet publication.
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
     * @param rbfMus    the mu parameters of the radial basis functions
     * @param rbfBetas  the beta parameters of the radial basis functions
     */
    EquivariantTransformerLayer(int numAtoms, int width, int numHeads, const std::vector<float>& rbfMus, const std::vector<float>& rbfBetas) :
            numAtoms(numAtoms), width(width), numHeads(numHeads), rbfMus(rbfMus), rbfBetas(rbfBetas) {
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
     * @param input               an array of shape [numAtoms][width] containing the input vectors
     * @param outputDeriv         an array of shape [numAtoms][width] containing the derivative of E with respect to each output value
     * @param inputDeriv          an array of shape [numAtoms][width] to store the derivative of E with respect to each input value into
     * @param positionDeriv       an array of shape [numAtoms][3] to store the derivative of E with respect to the atom positions into
     */
    virtual void backprop(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                          const float* input, const float* outputDeriv, float* inputDeriv, float* positionDeriv) = 0;
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
     * Get the mu parameters of the radial basis functions.
     */
    const std::vector<float>& getRBFMus() const {
        return rbfMus;
    }
    /**
     * Get the beta parameters of the radial basis functions.
     */
    const std::vector<float>& getRBFBetas() const {
        return rbfBetas;
    }
protected:
    const int numAtoms, width, numHeads;
    const std::vector<float> rbfMus, rbfBetas;
};

#endif
