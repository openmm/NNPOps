#ifndef CFCONV
#define CFCONV

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

/**
 * This class represents a neighbor list for use in computing CFConv layers.
 * A single object can be used for all layers in the models.  Create it at the
 * same time you create the model, call build() every time the atom positions
 * change, and pass it to the methods of CFConv objects that do computations
 * based on positions.
 * 
 * This is an abstract class.  Subclasses provide implementations on particular
 * types of hardware.
 */
class CFConvNeighbors {
public:
    /**
     * Create an object for computing neighbor lists.
     * 
     * @param numAtoms    the number of atoms in the system being modeled
     * @param cutoff      the cutoff distance
     * @param periodic    whether to use periodic boundary conditions.
     */
    CFConvNeighbors(int numAtoms, float cutoff, bool periodic) : numAtoms(numAtoms), cutoff(cutoff), periodic(periodic) {
    }
    virtual ~CFConvNeighbors() {
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
 * This class computes the continuous filter convolution (cfconv) function used in SchNet.
 * Create an instance of this class at the same time you create the model and then reuse it
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
class CFConv {
public:
    /**
     * This is an enumeration of supported activation functions.
     */
    enum ActivationFunction {
        /**
         * y(x) = log(0.5*exp(x) + 0.5)
         */
        ShiftedSoftplus = 0,
        /**
         * y(x) = tanh(x)
         */
        Tanh = 1
    };
    /**
     * Construct on object for computing continuous filter convolution (cfconv) functions.
     *
     * @param numAtoms       the number of atoms in the system
     * @param width          the number of elements in the input and output vectors
     * @param numGaussians   the number of Gaussian basis functions to use, uniformly spaced between 0 and cutoff
     * @param cutoff         the cutoff distance
     * @param periodic       whether to apply periodic boundary conditions
     * @param gaussianWidth  the width of the Gaussian basis functions
     * @param activation     the activation function to use between the two dense layers
     */
    CFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth,
           ActivationFunction activation) : numAtoms(numAtoms), width(width), numGaussians(numGaussians),
           cutoff(cutoff), periodic(periodic), gaussianWidth(gaussianWidth), activation(activation) {
    }
    virtual ~CFConv() {
    }
    /**
     * Compute the output of the layer.
     *
     * @param neighbors           a neighbor list for accelerating the calculation.  You must have already called
     *                            build() on the neighbor list with the same positions and box vectors.
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     * @param input               an array of shape [numAtoms][width] containing the input vectors
     * @param output              an array of shape [numAtoms][width] to store the output vectors into
     */
    virtual void compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                         const float* input, float* output) = 0;
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
    virtual void backprop(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
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
     * Get the number of Gaussian basis functions.
     */
    int getNumGaussians() const {
        return numGaussians;
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
    /**
     * Get the width of the Gaussian basis functions.
     */
    float getGaussianWidth() const {
        return gaussianWidth;
    }
    /**
     * Get the activation function used between the two dense layers.
     */
    ActivationFunction getActivation() const {
        return activation;
    }
protected:
    const int numAtoms, width, numGaussians;
    const float cutoff, gaussianWidth;
    const bool periodic;
    const ActivationFunction activation;
};

#endif
