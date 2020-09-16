#ifndef CPU_ANI_SYMMETRY_FUCTIONS
#define CPU_ANI_SYMMETRY_FUCTIONS

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

#include "ANISymmetryFunctions.h"

class CpuANISymmetryFunctions : public ANISymmetryFunctions {
public:
    /**
     * Construct on object for computing ANI symmetry functions on a CPU.
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
    CpuANISymmetryFunctions(int numAtoms, int numSpecies, float radialCutoff, float angularCutoff, bool periodic, const std::vector<int>& atomSpecies,
            const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions, bool torchani);
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
    void computeSymmetryFunctions(const float* positions, const float* periodicBoxVectors, float* radial, float* angular);
    /**
     * Given the derivatives of some function E (typically energy) with respect to the symmetry functions, backpropagate them
     * to find the derivates of E with respect to the atom positions.
     *
     * This must be called after computeSymmetryFunctions().  It uses the atom positions and box vectors that were specified in the most
     * recent call to that function.
     *
     * @param radialDeriv      an array of shape [numAtoms][numSpecies][radialFunctions.size()] containing the derivative
     *                         of E with respect to each radial symmetry function
     * @param angularDeriv     an array of shape [numSpecies*(numSpecies+1)/2][angularFunctions.size()] containing the derivative
     *                         of E with respect to each angular symmetry function
     * @param positionDeriv    an array of shape [numAtoms][3] to store the derivative of E with respect to the atom positions into
     */
    void backprop(const float* radialDeriv, const float* angularDeriv, float* positionDeriv);
private:
    /**
     * Compute the values of the radial symmetry functions.  At the same time we also build a neighbor list
     * that will be used when computing the angular symmetry functions.
     */
    template <bool PERIODIC, bool TRICLINIC>
    void computeRadialFunctions(float* radial);
    /**
     * Compute the values of the angular symmetry functions.  This must be called after computeRadialFunctions.
     */
    template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
    void computeAngularFunctions(float* angular);
    /**
     * Backpropagate through the radial symmetry functions to compute the derivatives with respect to positions.
     */
    template <bool PERIODIC, bool TRICLINIC>
    void backpropRadialFunctions(const float* radialDeriv, float* positionDeriv);
    /**
     * Backpropagate through the angular symmetry functions to compute the derivatives with respect to positions.
     */
    template <bool PERIODIC, bool TRICLINIC, bool TORCHANI>
    void backpropAngularFunctions(const float* angularDeriv, float* positionDeriv);
    /**
     * Compute the displacement between two positions.
     *
     * @param pos1     the first position
     * @param pos2     the second position
     * @param delta    the displacement vector is stored into this
     * @param r2       the squared distance is stored into this
     */
    template <bool PERIODIC, bool TRICLINIC>
    void computeDisplacement(const float* pos1, const float* pos2, float* delta, float& r2);
    /**
     * Compute the value of the cutoff function.  The implementation assumes the caller has already
     * verified that r <= rc.
     *
     * @param r     the distance at which it is being evaluated
     * @param rc    the cutoff distance
     */
    float cutoffFunction(float r, float rc);
    /**
     * Compute the derivative of the cutoff function.  The implementation assumes the caller has already
     * verified that r <= rc.
     *
     * @param r     the distance at which it is being evaluated
     * @param rc    the cutoff distance
     */
    float cutoffDeriv(float r, float rc);
    /**
     * Compute the angle between two vectors.
     *
     * @param vec1    the first vector
     * @param vec2    the second vector
     * @param r1      the length of the first vector
     * @param r2      the length of the second vector
     */
    template <bool TORCHANI>
    float computeAngle(const float* vec1, const float* vec2, float r1, float r2);
    /**
     * Compute the gradients of the angle between two vectors with respect to the vectors.
     *
     * @param vec1    the first vector
     * @param vec2    the second vector
     * @param r1      the length of the first vector
     * @param r2      the length of the second vector
     * @param grad1   the gradient with respect to the first vector is stored into this
     * @param grad2   the gradient with respect to the second vector is stored into this
    */
    template <bool TORCHANI>
    void computeAngleGradients(const float* vec1, const float* vec2, float r1, float r2, float* grad1, float* grad2);
    /**
     * Compute the cross product of two vectors.
     *
     * @param vec1    the first vector
     * @param vec2    the second vector
     * @param c       the cross product is stored into this
     */
    void computeCross(const float* vec1, const float* vec2, float* c);
    std::vector<std::vector<int> > neighbors;
    std::vector<std::vector<int> > angularIndex;
    std::vector<float> positions;
    float periodicBoxVectors[3][3];
    float invBoxSize[3];
    bool triclinic;
};

#endif