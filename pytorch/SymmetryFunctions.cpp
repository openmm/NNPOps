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
#include "CudaANISymmetryFunctions.h"

class CustomANISymmetryFunctions : public torch::CustomClassHolder {
public:
    CustomANISymmetryFunctions(int64_t numSpecies_,
                               double Rcr,
                               double Rca,
                               const std::vector<double>& EtaR,
                               const std::vector<double>& ShfR,
                               const std::vector<double>& EtaA,
                               const std::vector<double>& Zeta,
                               const std::vector<double>& ShfA,
                               const std::vector<double>& ShfZ,
                               const std::vector<int64_t>& atomSpecies_,
                               const torch::Tensor& positions) : torch::CustomClassHolder() {

        tensorOptions = positions.device();; // Data type of float by default
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

        if (tensorOptions.device().is_cpu())
            symFunc = std::make_shared<CpuANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies, radialFunctions, angularFunctions, true);
        if (tensorOptions.device().is_cuda())
            symFunc = std::make_shared<CudaANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies, radialFunctions, angularFunctions, true);
    };

    torch::autograd::tensor_list forward(const torch::Tensor& positions_) {

        const auto positions = positions_.to(tensorOptions);
        auto radial  = torch::empty({numAtoms, numSpecies * (int)radialFunctions.size()}, tensorOptions);
        auto angular = torch::empty({numAtoms, numSpecies * (numSpecies + 1) / 2 * (int)angularFunctions.size()}, tensorOptions);

        symFunc->computeSymmetryFunctions(positions.data_ptr<float>(), nullptr, radial.data_ptr<float>(), angular.data_ptr<float>());

        return {radial, angular};
    };

    torch::Tensor backward(const torch::autograd::tensor_list& grads) {

        const auto radialGrad = grads[0].clone();
        const auto angularGrad = grads[1].clone();
        auto positionsGrad = torch::empty({numAtoms, 3}, tensorOptions);

        symFunc->backprop(radialGrad.data_ptr<float>(), angularGrad.data_ptr<float>(), positionsGrad.data_ptr<float>());

        return positionsGrad;
    }

private:
    torch::TensorOptions tensorOptions;
    int numAtoms;
    int numSpecies;
    std::vector<RadialFunction> radialFunctions;
    std::vector<AngularFunction> angularFunctions;
    std::shared_ptr<ANISymmetryFunctions> symFunc;
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
                                                const std::vector<int64_t>& atomSpecies,
                                                const torch::Tensor& positions) {

        const auto symFunc = torch::intrusive_ptr<CustomANISymmetryFunctions>::make(numSpecies, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, atomSpecies, positions);
        ctx->saved_data["symFunc"] = symFunc;

        return symFunc->forward(positions);
    };

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, const torch::autograd::tensor_list& grads) {

        const auto symFunc = ctx->saved_data["symFunc"].toCustomClass<CustomANISymmetryFunctions>();
        torch::Tensor positionsGrad = symFunc->backward(grads);

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
    m.class_<CustomANISymmetryFunctions>("CustomANISymmetryFunctions")
        .def(torch::init<int64_t,                        // numSpecies
                         double,                         // Rcr
                         double,                         // Rca
                         const std::vector<double>&,     // EtaR
                         const std::vector<double>&,     // ShfR
                         const std::vector<double>&,     // EtaA
                         const std::vector<double>&,     // Zeta
                         const std::vector<double>&,     // ShfA
                         const std::vector<double>&,     // ShfZ
                         const std::vector<int64_t>&,    // atomSpecies
                         const torch::Tensor&>())        // positions
        .def("forward", &CustomANISymmetryFunctions::forward)
        .def("backward", &CustomANISymmetryFunctions::backward);
    m.def("ANISymmetryFunction", ANISymmetryFunction);
}