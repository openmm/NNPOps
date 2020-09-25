/**
 * Copyright (c) 2020 Acellera
 * Authors: Raimondas Galvelis
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

#include <torch/script.h>
#include "CpuANISymmetryFunctions.h"

static torch::autograd::tensor_list ANISymmetryFunction(int64_t numSpecies,
                                                        double Rcr,
                                                        double Rca,
                                                        const std::vector<double>& EtaR,
                                                        const std::vector<double>& ShfR,
                                                        const std::vector<double>& EtaA,
                                                        const std::vector<double>& Zeta,
                                                        const std::vector<double>& ShfA,
                                                        const std::vector<double>& ShfZ,
                                                        const std::vector<int64_t>& atomSpecies_,
                                                        const torch::Tensor& positions_) {

    const int numAtoms = atomSpecies_.size();
    const std::vector<int> atomSpecies(atomSpecies_.begin(), atomSpecies_.end());

    std::vector<RadialFunction> radialFunctions;
    for (const float eta: EtaR)
        for (const float rs: ShfR)
            radialFunctions.push_back({eta, rs});

    std::vector<AngularFunction> angularFunctions;
    for (const float eta: EtaA)
        for (const float zeta: Zeta)
            for (const float rs: ShfA)
                for (const float thetas: ShfZ)
                    angularFunctions.push_back({eta, rs, zeta, thetas});

    CpuANISymmetryFunctions sf(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies, radialFunctions, angularFunctions, true);

    const auto positions = positions_.toType(torch::kFloat);
    auto radial  = torch::empty({numAtoms, numSpecies * (int)radialFunctions.size()}, torch::kFloat);
    auto angular = torch::empty({numAtoms, numSpecies * (numSpecies + 1) / 2 * (int)angularFunctions.size()}, torch::kFloat);

    sf.computeSymmetryFunctions(positions.data_ptr<float>(), nullptr, radial.data_ptr<float>(), angular.data_ptr<float>());

    return {radial, angular};
}

TORCH_LIBRARY(NNPOps, m) {
    m.def("ANISymmetryFunction", ANISymmetryFunction);
}
