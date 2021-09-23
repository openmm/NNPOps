#include "CpuANISymmetryFunctions.h"

ANISymmetryFunctions* createSymmetryCalculator(int numAtoms, int numSpecies, float radialCutoff, float angularCutoff, bool periodic, const std::vector<int>& atomSpecies,
            const std::vector<RadialFunction>& radialFunctions, const std::vector<AngularFunction>& angularFunctions, bool torchani) {
    return new CpuANISymmetryFunctions(numAtoms, numSpecies, radialCutoff, angularCutoff, periodic, atomSpecies, radialFunctions, angularFunctions, torchani);
}

#include "TestANISymmetryFunctions.h"
