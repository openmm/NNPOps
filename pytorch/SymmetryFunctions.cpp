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

namespace NNPOps {
namespace ANISymmetryFunctions {

class Holder;
using std::vector;
using HolderPtr = torch::intrusive_ptr<Holder>;
using torch::Tensor;
using torch::optional;
using Context = torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class Holder : public torch::CustomClassHolder {
public:
    Holder(int64_t numSpecies_,
           double Rcr,
           double Rca,
           const vector<double>& EtaR,
           const vector<double>& ShfR,
           const vector<double>& EtaA,
           const vector<double>& Zeta,
           const vector<double>& ShfA,
           const vector<double>& ShfZ,
           const vector<int64_t>& atomSpecies_,
           const Tensor& positions) : torch::CustomClassHolder() {

        if (numSpecies_ == 0)
            return;

        tensorOptions = torch::TensorOptions().device(positions.device()); // Data type of float by default
        int numAtoms = atomSpecies_.size();
        int numSpecies = numSpecies_;
        const vector<int> atomSpecies(atomSpecies_.begin(), atomSpecies_.end());

        vector<RadialFunction> radialFunctions;
        for (const float eta: EtaR)
            for (const float rs: ShfR)
                radialFunctions.push_back({eta, rs});

        vector<AngularFunction> angularFunctions;
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

    tensor_list forward(const Tensor& positions_, const optional<Tensor>& periodicBoxVectors_) {

        const Tensor positions = positions_.to(tensorOptions);

        Tensor periodicBoxVectors;
        float* periodicBoxVectorsPtr = nullptr;
        if (periodicBoxVectors_) {
            periodicBoxVectors = periodicBoxVectors_->to(tensorOptions);
            float* periodicBoxVectorsPtr = periodicBoxVectors.data_ptr<float>();
        }

        symFunc->computeSymmetryFunctions(positions.data_ptr<float>(), periodicBoxVectorsPtr, radial.data_ptr<float>(), angular.data_ptr<float>());

        return {radial, angular};
    };

    Tensor backward(const tensor_list& grads) {

        const Tensor radialGrad = grads[0].clone();
        const Tensor angularGrad = grads[1].clone();

        symFunc->backprop(radialGrad.data_ptr<float>(), angularGrad.data_ptr<float>(), positionsGrad.data_ptr<float>());

        return positionsGrad;
    };

    bool is_initialized() {
        return bool(symFunc);
    };

private:
    torch::TensorOptions tensorOptions;
    std::shared_ptr<::ANISymmetryFunctions> symFunc;
    Tensor radial;
    Tensor angular;
    Tensor positionsGrad;
};

class AutogradFunctions : public torch::autograd::Function<AutogradFunctions> {

public:
    static tensor_list forward(Context *ctx,
                               const HolderPtr& holder,
                               const Tensor& positions,
                               const optional<Tensor>& periodicBoxVectors) {

        ctx->saved_data["holder"] = holder;

        return holder->forward(positions, periodicBoxVectors);
    };

    static tensor_list backward(Context *ctx, const tensor_list& grads) {

        const auto holder = ctx->saved_data["holder"].toCustomClass<Holder>();
        Tensor positionsGrad = holder->backward(grads);
        ctx->saved_data.erase("holder");

        return { Tensor(),      // symFunc
                 positionsGrad, // positions
                 Tensor() };    // periodicBoxVectors
    };
};

tensor_list operation(const optional<HolderPtr>& holder,
                      const Tensor& positions,
                      const optional<Tensor>& periodicBoxVectors) {

    return AutogradFunctions::apply(*holder, positions, periodicBoxVectors);
}

TORCH_LIBRARY(NNPOpsANISymmetryFunctions, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<int64_t,                // numSpecies
                         double,                 // Rcr
                         double,                 // Rca
                         const vector<double>&,  // EtaR
                         const vector<double>&,  // ShfR
                         const vector<double>&,  // EtaA
                         const vector<double>&,  // Zeta
                         const vector<double>&,  // ShfA
                         const vector<double>&,  // ShfZ
                         const vector<int64_t>&, // atomSpecies
                         const Tensor&>())       // positions
        .def("forward", &Holder::forward)
        .def("backward", &Holder::backward)
        .def("is_initialized", &Holder::is_initialized);
    m.def("operation", operation);
}

} // namespace ANISymmetryFunctions
} // namespace NNPOps