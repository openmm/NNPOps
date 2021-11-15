/**
 * Copyright (c) 2020-2021 Acellera
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
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include "CpuCFConv.h"
#include "CudaCFConv.h"

#define CHECK_CUDA_RESULT(result) \
    if (result != cudaSuccess) { \
        throw std::runtime_error(std::string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+std::to_string(__LINE__));\
    }

namespace NNPOps {
namespace SchNet {

class Holder;
using std::vector;
using HolderPtr = torch::intrusive_ptr<Holder>;
using torch::Tensor;
using torch::optional;
using Context = torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class Holder : public torch::CustomClassHolder {
public:

    // Constructor for an uninitialized object
    // Note: this is need for serialization
    Holder() : torch::CustomClassHolder() {};

    Holder(int64_t numAtoms,
           int64_t numFilters,
           int64_t numGausians,
           double cutoff,
           double gaussianWidth,
           int64_t activation_,
           const vector<double>& linear1_weights_,
           const vector<double>& linear1_biases_,
           const vector<double>& linear2_weights_,
           const vector<double>& linear2_biases_,
           const Tensor& positions) : torch::CustomClassHolder() {

        // Construct an uninitialized object
        // Note: this is needed for Python bindings
        if (numAtoms == 0)
            return;

        const CFConv::ActivationFunction activation = static_cast<CFConv::ActivationFunction>(activation_);

        const vector<float> linear1_weights = {linear1_weights_.begin(), linear1_weights_.end()};
        const vector<float> linear1_biases = {linear1_weights_.begin(), linear1_biases_.end()};
        const vector<float> linear2_weights = {linear2_weights_.begin(), linear2_weights_.end()};
        const vector<float> linear2_biases = {linear1_weights_.begin(), linear2_biases_.end()};

        tensorOptions = torch::TensorOptions().device(positions.device()); // Data type of float by default

        const torch::Device& device = tensorOptions.device();
        if (device.is_cpu())
            neighbors = std::make_shared<CpuCFConvNeighbors>(numAtoms, cutoff, false);
            conv = std::make_shared<CpuCFConv>(numAtoms, numFilters, numGausians, cutoff, false, gaussianWidth, activation,
                                               linear1_weights.data(), linear1_biases.data(), linear2_weights.data(), linear2_biases.data());
        if (device.is_cuda()) {
            // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
            CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
            neighbors = std::make_shared<CudaCFConvNeighbors>(numAtoms, cutoff, false);
            conv = std::make_shared<CudaCFConv>(numAtoms, numFilters, numGausians, cutoff, false, gaussianWidth, activation,
                                                linear1_weights.data(), linear1_biases.data(), linear2_weights.data(), linear2_biases.data());
        }

        output  = torch::empty({numAtoms, numFilters}, tensorOptions);
        inputGrad = torch::empty({numAtoms, numFilters}, tensorOptions);
        positionsGrad = torch::empty({numAtoms, 3}, tensorOptions);

        // cudaConv = dynamic_cast<CudaCFConv*>(conv.get());
    };

    tensor_list forward(const Tensor& positions_, const optional<Tensor>& periodicBoxVectors_, const Tensor& input_) {

        positions = positions_.to(tensorOptions).clone();
        input = input_.to(tensorOptions).clone();

        periodicBoxVectorsPtr = nullptr;
        if (periodicBoxVectors_) {
            periodicBoxVectors = periodicBoxVectors_->to(tensorOptions);
            periodicBoxVectorsPtr = periodicBoxVectors.data_ptr<float>();
        }

        // if (cudaConv) {
        //     const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(tensorOptions.device().index());
        //     cudaConv->setStream(stream.stream());
        // }

        neighbors->build(positions.data_ptr<float>(), periodicBoxVectorsPtr);
        conv->compute(*neighbors.get(), positions.data_ptr<float>(), periodicBoxVectorsPtr, positions.data_ptr<float>(), output.data_ptr<float>());

        return {output};
    };

    tensor_list backward(const tensor_list& grads) {

        const Tensor outputGrad = grads[0].clone();

        // if (cudaConv) {
        //     const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(tensorOptions.device().index());
        //     cudaConv->setStream(stream.stream());
        // }

        conv->backprop(*neighbors.get(), positions.data_ptr<float>(), periodicBoxVectorsPtr, input.data_ptr<float>(),
                       outputGrad.data_ptr<float>(), inputGrad.data_ptr<float>(), positionsGrad.data_ptr<float>());

        return {positionsGrad, inputGrad};
    };

    bool is_initialized() {
        return bool(conv);
    };

private:
    torch::TensorOptions tensorOptions;
    std::shared_ptr<::CFConvNeighbors> neighbors;
    std::shared_ptr<::CFConv> conv;
    Tensor positions;
    Tensor periodicBoxVectors;
    float* periodicBoxVectorsPtr;
    Tensor input;
    Tensor output;
    Tensor positionsGrad;
    Tensor inputGrad;
    // CudaCFConv* cudaConv;
};

class AutogradFunctions : public torch::autograd::Function<AutogradFunctions> {

public:
    static tensor_list forward(Context *ctx,
                               const HolderPtr& holder,
                               const Tensor& positions,
                               const optional<Tensor>& periodicBoxVectors,
                               const Tensor& input) {

        ctx->saved_data["holder"] = holder;

        return holder->forward(positions, periodicBoxVectors, input);
    };

    static tensor_list backward(Context *ctx, const tensor_list& grads) {

        const auto holder = ctx->saved_data["holder"].toCustomClass<Holder>();
        tensor_list output = holder->backward(grads);
        ctx->saved_data.erase("holder");

        return { Tensor(),   // holder
                 output[0],  // positions
                 Tensor(),   // periodicBoxVectors
                 output[1]}; // input
    };
};

tensor_list operation(const optional<HolderPtr>& holder,
                      const Tensor& positions,
                      const optional<Tensor>& periodicBoxVectors,
                      const Tensor& input) {

    return AutogradFunctions::apply(*holder, positions, periodicBoxVectors, input);
}

TORCH_LIBRARY(NNPOpsSchNetCFConv, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<int64_t,               // nunAtoms
                         int64_t,               // numFilters
                         int64_t,               // numGausians
                         double,                // cutoff
                         double,                // gaussianWidth
                         int64_t,               // activation
                         const vector<double>&, // linear1_weights
                         const vector<double>&, // linear1_biases
                         const vector<double>&, // linear2_weights
                         const vector<double>&, // linear2_biases
                         const Tensor&>())      // positions
        .def("forward", &Holder::forward)
        .def("backward", &Holder::backward)
        .def("is_initialized", &Holder::is_initialized)
        .def_pickle(
            // __getstate__
            // Note: nothing is during serialization
            [](const HolderPtr& self) -> int64_t { return 0; },
            // __setstate__
            // Note: a new uninitialized object is create during deserialization
            [](int64_t state) -> HolderPtr { return HolderPtr::make(); }
        );
    m.def("operation", operation);
}

} // namespace SchNet
} // namespace NNPOps