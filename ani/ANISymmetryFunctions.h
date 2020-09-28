#ifndef ANI_SYMMETRY_FUCTIONS
#define ANI_SYMMETRY_FUCTIONS

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

#include <vector>

struct RadialFunction {
    float eta;
    float rs;
};

struct AngularFunction {
    float eta;
    float rs;
    float zeta;
    float thetas;
};

class ANISymmetryFunctions {
public:
    /**
     * Construct on object for computing ANI symmetry functions.
     *
     * @param numAtoms            the number of atoms in the system
     * @param numSpecies          the number of species of atoms
     * @param radialCutoff        the cutoff distance for radial symmetry functions
     * @param angularCutoff       the cutoff distance for angular symmetry functions
     * @param periodic            whether to apply periodic boundary conditions
     * @param atomSpecies         a vector of length numAtoms containing the species of each atom, represented as an integer
     *                            between 0 and numSpecies-1
     * @param radialFunctions     the radial symmetry functions to compute
     * @param angularFunctions    the angular symmetry functions to compute
     * @param torchani            if false, perform calculations as described in the original publication (https://doi.org/10.1039/C6SC05720A).
     *                            If true, perform them as implemented in TorchANI (https://github.com/aiqm/torchani).  They differ in two ways.
     *                            First, TorchANI divides the radial symmetry functions by 4.  Second, when computing angles it multiplies the
     *                            dot product by 0.95.  This leads to large errors in angles, especially ones that are close to 0 or pi.
     */
    ANISymmetryFunctions(int numAtoms, int numSpecies, float radialCutoff, float angularCutoff, bool periodic, const std::vector<int>& atomSpecies,
            const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions, bool torchani) :
               numAtoms(numAtoms), numSpecies(numSpecies), radialCutoff(radialCutoff), angularCutoff(angularCutoff),
               periodic(periodic), atomSpecies(atomSpecies), radialFunctions(radialFunctions), angularFunctions(angularFunctions), torchani(torchani) {
    }
    /**
     * Compute the symmetry functions.
     *
     * @param positions           an array of shape [numAtoms][3] containing the positions of each atom
     * @param periodicBoxVectors  an array of shape [3][3] containing the periodic box vectors.  If periodic boundary conditions are
     *                            not used, this is ignored and may be NULL.
     * @param radial              an array of shape [numAtoms][numSpecies][radialFunctions.size()] to store the
     *                            radial symmetry function values into
     * @param angular             an array of shape [numAtoms][numSpecies*(numSpecies+1)/2][angularFunctions.size()] to store the
     *                            angular symmetry function values into
     */
    virtual void computeSymmetryFunctions(const float* positions, const float* periodicBoxVectors, float* radial, float* angular) = 0;
    /**
     * Given the derivatives of some function E (typically energy) with respect to the symmetry functions, backpropagate them
     * to find the derivates of E with respect to the atom positions.
     *
     * This must be called after computeSymmetryFunctions().  It uses the atom positions and box vectors that were specified in the most
     * recent call to that function.
     *
     * @param radialDeriv      an array of shape [numAtoms][numSpecies][radialFunctions.size()] containing the derivative
     *                         of E with respect to each radial symmetry function
     * @param angularDeriv     an array of shape [numAtoms][numSpecies*(numSpecies+1)/2][angularFunctions.size()] containing the derivative
     *                         of E with respect to each angular symmetry function
     * @param positionDeriv    an array of shape [numAtoms][3] to store the derivative of E with respect to the atom positions into
     */
    virtual void backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv) = 0;
    /**
     * Get the number of atoms in the system.
     */
    int getNumAtoms() const {
        return numAtoms;
    }
    /**
     * Get the number of species of atoms.
     */
    int getNumSpecies() const {
        return numSpecies;
    }
    /**
     * Get the cutoff distance for radial symmetry functions.
     */
    float getRadialCutoff() const {
        return radialCutoff;
    }
    /**
     * Get the cutoff distance for angular symmetry functions.
     */
    float getAngularCutoff() const {
        return angularCutoff;
    }
    /**
     * Get whether to apply periodic boundary conditions.
     */
    bool getPeriodic() const {
        return periodic;
    }
    /**
     * Get whether calculations are done as described in the publication or as implemented in TorchANI.
     */
    bool getTorchANI() const {
        return torchani;
    }
    /**
     * Get the species of each atom, represented as an integer between 0 and numSpecies-1.
     */
    const std::vector<int>& getAtomSpecies() const {
        return atomSpecies;
    }
    /**
     * Get the radial symmetry functions to compute.
     */
    const std::vector<RadialFunction>& getRadialFunctions() const {
        return radialFunctions;
    }
    /**
     * Get the angular symmetry functions to compute.
     */
    const std::vector<AngularFunction>& getAngularFunctions() const {
        return angularFunctions;
    }
protected:
    const int numAtoms, numSpecies;
    const float radialCutoff, angularCutoff;
    const bool periodic, torchani;
    const std::vector<int> atomSpecies;
    const std::vector<RadialFunction> radialFunctions;
    const std::vector<AngularFunction> angularFunctions;
};

#endif
