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

class CFConvNeighbors {
public:
    CFConvNeighbors(int numAtoms, float cutoff, bool periodic) : numAtoms(numAtoms), cutoff(cutoff), periodic(periodic) {
    }
    virtual ~CFConvNeighbors() {
    }
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
}

class CFConv {
public:
    /**
     * Construct on object for computing continuous filter convolution (cfconv) functions.
     *
     * @param numAtoms            the number of atoms in the system
     * @param cutoff              the cutoff distance
     * @param periodic            whether to apply periodic boundary conditions
     */
    CFConv(int numAtoms, int width, int numGaussians, float cutoff, bool periodic, float gaussianWidth) :
           numAtoms(numAtoms), width(width), numGaussians(numGaussians), cutoff(cutoff), periodic(periodic), gaussianWidth(gaussianWidth) {
    }
    virtual ~CFConv() {
    }
    /**
     * Compute the output of the layer.
     *
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     */
    virtual void compute(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                         float* input, float* output, const float* w1, const float* b1, const float* w2, const float* b2) = 0;
    /**
     * Given the derivatives of some function E (typically energy) with respect to the outputs, backpropagate them
     * to find the derivates of E with respect to the inputs and atom positions.
     *
     * This must be called after compute().  It uses the atom positions and box vectors that were specified in the most
     * recent call to that function.
     *
     * @param outputDeriv      an array of shape [numAtoms][outputWidth] containing the derivative of E with respect to each output value
     * @param inputDeriv       an array of shape [numAtoms][inputWidth] to store the derivative of E with respect to each input value into
     * @param positionDeriv    an array of shape [numAtoms][3] to store the derivative of E with respect to the atom positions into
     */
    virtual void backprop(const CFConvNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                          const float* outputDeriv, float* inputDeriv, float* positionDeriv, const float* w1, const float* b1,
                          const float* w2, const float* b2) = 0;
    /**
     * Get the number of atoms in the system.
     */
    int getNumAtoms() const {
        return numAtoms;
    }
    int getWidth() const {
        return width;
    }
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
    float getGaussianWidth() const {
        return gaussianWidth;
    }
protected:
    const int numAtoms, width, numGaussians;
    const float cutoff, gaussianWidth;
    const bool periodic;
};

#endif
