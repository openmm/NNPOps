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

class CustomCpuANISymmetryFunctions : public torch::CustomClassHolder {
public:
    CustomCpuANISymmetryFunctions(int64_t numSpecies_,
                                  double Rcr,
                                  double Rca,
                                  const std::vector<double>& EtaR,
                                  const std::vector<double>& ShfR,
                                  const std::vector<double>& EtaA,
                                  const std::vector<double>& Zeta,
                                  const std::vector<double>& ShfA,
                                  const std::vector<double>& ShfZ,
                                  const std::vector<int64_t>& atomSpecies_) : torch::CustomClassHolder() {

        numAtoms = atomSpecies_.size();
        numSpecies = numSpecies_;
        const std::vector<int> atomSpecies(atomSpecies_.begin(), atomSpecies_.end());

        for (const float eta: EtaR)
            for (const float rs: ShfR)
                radialFunctions.push_back({eta, rs});

        for (const float eta: EtaA)
            for (const float zeta: Zeta)
                for (const float rs: ShfA)
                    for (const float thetas: ShfZ)
                        angularFunctions.push_back({eta, rs, zeta, thetas});

        symFunc = std::make_shared<CpuANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies, radialFunctions, angularFunctions, true);
    };

    torch::autograd::tensor_list forward(const torch::Tensor& positions_) {

        const auto positions = positions_.toType(torch::kFloat);
        auto radial  = torch::empty({numAtoms, numSpecies * (int)radialFunctions.size()}, torch::kFloat);
        auto angular = torch::empty({numAtoms, numSpecies * (numSpecies + 1) / 2 * (int)angularFunctions.size()}, torch::kFloat);

        symFunc->computeSymmetryFunctions(positions.data_ptr<float>(), nullptr, radial.data_ptr<float>(), angular.data_ptr<float>());

        return {radial, angular};
    };

private:
    int numAtoms;
    int numSpecies;
    std::vector<RadialFunction> radialFunctions;
    std::vector<AngularFunction> angularFunctions;
    std::shared_ptr<CpuANISymmetryFunctions> symFunc;
};

class GradANISymmetryFunction : public torch::autograd::Function<GradANISymmetryFunction> {

public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext *ctx,
                                                int64_t numSpecies,
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

        const auto symFunc = torch::make_custom_class<CustomCpuANISymmetryFunctions>(numSpecies, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, atomSpecies_);
        ctx->saved_data["symFunc"] = symFunc;

        return symFunc.toCustomClass<CustomCpuANISymmetryFunctions>()->forward(positions_);
    };

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grads) {

        const auto& radialGrad = grads[0];
        const auto& angularGrad = grads[1];

        const auto symFunc = ctx->saved_data["symFunc"].toCustomClass<CustomCpuANISymmetryFunctions>();

        // compute the gradients

        torch::Tensor positionsGrad = torch::Tensor();

        return { torch::Tensor(), // numSpecies
                 torch::Tensor(), // Rcr
                 torch::Tensor(), // Rca
                 torch::Tensor(), // EtaR
                 torch::Tensor(), // ShfR
                 torch::Tensor(), // EtaA
                 torch::Tensor(), // Zeta
                 torch::Tensor(), // ShfA
                 torch::Tensor(), // ShfZ
                 torch::Tensor(), // atomSpecies
                 positionsGrad }; // positions
    };
};

static torch::autograd::tensor_list ANISymmetryFunction(int64_t numSpecies,
                                                        double Rcr,
                                                        double Rca,
                                                        const std::vector<double>& EtaR,
                                                        const std::vector<double>& ShfR,
                                                        const std::vector<double>& EtaA,
                                                        const std::vector<double>& Zeta,
                                                        const std::vector<double>& ShfA,
                                                        const std::vector<double>& ShfZ,
                                                        const std::vector<int64_t>& atomSpecies,
                                                        const torch::Tensor& positions) {
    return GradANISymmetryFunction::apply(numSpecies, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, atomSpecies, positions);
}

TORCH_LIBRARY(NNPOps, m) {
    m.class_<CustomCpuANISymmetryFunctions>("CustomCpuANISymmetryFunctions")
        .def(torch::init<int64_t,                        // numSpecies
                         double,                         // Rcr
                         double,                         // Rca
                         const std::vector<double>&,     // EtaR
                         const std::vector<double>&,     // ShfR
                         const std::vector<double>&,     // EtaA
                         const std::vector<double>&,     // Zeta
                         const std::vector<double>&,     // ShfA
                         const std::vector<double>&,     // ShfZ
                         const std::vector<int64_t>&>()) // atomSpecies
        .def("forward", &CustomCpuANISymmetryFunctions::forward);
    m.def("ANISymmetryFunction", ANISymmetryFunction);
}