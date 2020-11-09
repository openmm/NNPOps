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

#include <stdexcept>
#include <cuda_runtime.h>
#include <torch/script.h>
#include "CpuANISymmetryFunctions.h"
#include "CudaANISymmetryFunctions.h"

#define CHECK_CUDA_RESULT(result) \
    if (result != cudaSuccess) { \
        throw std::runtime_error(std::string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+std::to_string(__LINE__));\
    }

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

        tensorOptions = torch::TensorOptions().device(positions.device()); // Data type of float by default
        int numAtoms = atomSpecies_.size();
        int numSpecies = numSpecies_;
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

        const torch::Device& device = tensorOptions.device();
        if (device.is_cpu())
            symFunc = std::make_shared<CpuANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies, radialFunctions, angularFunctions, true);
        if (device.is_cuda()) {
            // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
            CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
            symFunc = std::make_shared<CudaANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies, radialFunctions, angularFunctions, true);
        }

        radial  = torch::empty({numAtoms, numSpecies * (int)radialFunctions.size()}, tensorOptions);
        angular = torch::empty({numAtoms, numSpecies * (numSpecies + 1) / 2 * (int)angularFunctions.size()}, tensorOptions);
        positionsGrad = torch::empty({numAtoms, 3}, tensorOptions);
    };

    torch::autograd::tensor_list forward(const torch::Tensor& positions_, const torch::optional<torch::Tensor>& periodicBoxVectors_) {

        const torch::Tensor positions = positions_.to(tensorOptions);

        torch::Tensor periodicBoxVectors;
        float* periodicBoxVectorsPtr = nullptr;
        if (periodicBoxVectors_) {
            periodicBoxVectors = periodicBoxVectors_->to(tensorOptions);
            float* periodicBoxVectorsPtr = periodicBoxVectors.data_ptr<float>();
        }

        symFunc->computeSymmetryFunctions(positions.data_ptr<float>(), periodicBoxVectorsPtr, radial.data_ptr<float>(), angular.data_ptr<float>());

        return {radial, angular};
    };

    torch::Tensor backward(const torch::autograd::tensor_list& grads) {

        const torch::Tensor radialGrad = grads[0].clone();
        const torch::Tensor angularGrad = grads[1].clone();

        symFunc->backprop(radialGrad.data_ptr<float>(), angularGrad.data_ptr<float>(), positionsGrad.data_ptr<float>());

        return positionsGrad;
    }

private:
    torch::TensorOptions tensorOptions;
    std::shared_ptr<ANISymmetryFunctions> symFunc;
    torch::Tensor radial;
    torch::Tensor angular;
    torch::Tensor positionsGrad;
};

class GradANISymmetryFunction : public torch::autograd::Function<GradANISymmetryFunction> {

public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        const torch::intrusive_ptr<CustomANISymmetryFunctions>& symFunc,
        const torch::Tensor& positions,
        const torch::optional<torch::Tensor>& periodicBoxVectors) {

        ctx->saved_data["symFunc"] = symFunc;

        return symFunc->forward(positions, periodicBoxVectors);
    };

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        const torch::autograd::tensor_list& grads) {

        const auto symFunc = ctx->saved_data["symFunc"].toCustomClass<CustomANISymmetryFunctions>();
        torch::Tensor positionsGrad = symFunc->backward(grads);
        ctx->saved_data.erase("symFunc");

        return { torch::Tensor(),   // symFunc
                 positionsGrad,     // positions
                 torch::Tensor() }; // periodicBoxVectors
    };
};

static torch::autograd::tensor_list ANISymmetryFunctionsOp(
    const torch::optional<torch::intrusive_ptr<CustomANISymmetryFunctions>>& symFunc,
    const torch::Tensor& positions,
    const torch::optional<torch::Tensor>& periodicBoxVectors) {

    return GradANISymmetryFunction::apply(*symFunc, positions, periodicBoxVectors);
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
    m.def("ANISymmetryFunctions", ANISymmetryFunctionsOp);
}