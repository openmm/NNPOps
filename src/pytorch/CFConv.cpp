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
#include <cuda_runtime.h>
#include <torch/script.h>
// #include <c10/cuda/CUDAStream.h>
#include "CpuCFConv.h"
#include "CudaCFConv.h"
#include "CFConvNeighbors.h"

#define CHECK_CUDA_RESULT(result) \
    if (result != cudaSuccess) { \
        throw std::runtime_error(std::string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+std::to_string(__LINE__));\
    }

namespace NNPOps {
namespace CFConv {

class Holder;
using HolderPtr = torch::intrusive_ptr<Holder>;
using Neighbors = NNPOps::CFConvNeighbors::Holder;
using torch::IValue;
using torch::Tensor;
using torch::autograd::tensor_list;
using torch::optional;
using Context = torch::autograd::AutogradContext;

class Holder : public torch::CustomClassHolder {
public:

    // Constructor for an uninitialized object
    // Note: this is need for serialization
    Holder() : torch::CustomClassHolder() {};

    Holder(const IValue& neighbors,
           int64_t numAtoms,
           int64_t numFilters,
           int64_t numGausians,
           double gaussianWidth,
           int64_t activation_,
           const Tensor& weights1_,
           const Tensor& biases1_,
           const Tensor& weights2_,
           const Tensor& biases2_,
           const Tensor& positions) : torch::CustomClassHolder(), neighbors(neighbors.toCustomClass<Neighbors>()) {

        // Construct an uninitialized object
        // Note: this is needed for Python bindings
        if (numAtoms == 0)
            return;

        tensorOptions = torch::TensorOptions().device(positions.device()); // Data type of float by default

        double cutoff = this->neighbors->getCutoff();
        const ::CFConv::ActivationFunction activation = static_cast<::CFConv::ActivationFunction>(activation_);

        // Note: weights and biases have to be in the CPU memory
        const Tensor weights1 = weights1_.to(tensorOptions).cpu();
        const Tensor biases1 = biases1_.to(tensorOptions).cpu();
        const Tensor weights2 = weights2_.to(tensorOptions).cpu();
        const Tensor biases2 = biases2_.to(tensorOptions).cpu();

        const torch::Device& device = tensorOptions.device();

        if (device.is_cpu()) {
            conv = std::make_shared<::CpuCFConv>(numAtoms, numFilters, numGausians, cutoff, false, gaussianWidth, activation,
                                                 weights1.data_ptr<float>(), biases1.data_ptr<float>(), weights2.data_ptr<float>(), biases2.data_ptr<float>());
        } else if (device.is_cuda()) {
            // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
            CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
            conv = std::make_shared<::CudaCFConv>(numAtoms, numFilters, numGausians, cutoff, false, gaussianWidth, activation,
                                                  weights1.data_ptr<float>(), biases1.data_ptr<float>(), weights2.data_ptr<float>(), biases2.data_ptr<float>());
        } else
            throw std::runtime_error("Unsupported device");

        output = torch::empty({numAtoms, numFilters}, tensorOptions);
        inputGrad = torch::empty({numAtoms, numFilters}, tensorOptions);
        positionsGrad = torch::empty({numAtoms, 3}, tensorOptions);

        // cudaConv = dynamic_cast<CudaCFConv*>(conv.get());
    };

    Tensor forward(const Tensor& positions_, const Tensor& input_) {

        positions = positions_.to(tensorOptions).clone();
        input = input_.to(tensorOptions).clone();

        // if (cudaConv) {
        //     const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(tensorOptions.device().index());
        //     cudaConv->setStream(stream.stream());
        // }

        conv->compute(neighbors->getImpl(), positions.data_ptr<float>(), nullptr, positions.data_ptr<float>(), output.data_ptr<float>());

        return output;
    };

    tensor_list backward(const Tensor& outputGrad_) {

        const Tensor outputGrad = outputGrad_.clone();

        // if (cudaConv) {
        //     const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(tensorOptions.device().index());
        //     cudaConv->setStream(stream.stream());
        // }

        conv->backprop(neighbors->getImpl(), positions.data_ptr<float>(), nullptr, input.data_ptr<float>(),
                       outputGrad.data_ptr<float>(), inputGrad.data_ptr<float>(), positionsGrad.data_ptr<float>());

        return {positionsGrad, inputGrad};
    };

    bool is_initialized() {
        return bool(conv);
    };

private:
    torch::intrusive_ptr<Neighbors> neighbors;
    torch::TensorOptions tensorOptions;
    std::shared_ptr<::CFConv> conv;
    Tensor positions;
    Tensor input;
    Tensor output;
    Tensor positionsGrad;
    Tensor inputGrad;
    // CudaCFConv* cudaConv;
};

class AutogradFunctions : public torch::autograd::Function<AutogradFunctions> {

public:
    static Tensor forward(Context *ctx,
                          const HolderPtr& holder,
                          const Tensor& positions,
                          const Tensor& input) {

        ctx->saved_data["holder"] = holder;

        return holder->forward(positions, input);
    };

    static tensor_list backward(Context *ctx, const tensor_list& grads) {

        const HolderPtr holder = ctx->saved_data["holder"].toCustomClass<Holder>();
        tensor_list output = holder->backward(grads[0]);
        ctx->saved_data.erase("holder");

        return { Tensor(),   // holder
                 output[0],  // positions
                 output[1]}; // input
    };
};

Tensor operation(const optional<HolderPtr>& holder,
                 const Tensor& positions,
                 const Tensor& input) {

    return AutogradFunctions::apply(*holder, positions, input);
}

TORCH_LIBRARY(NNPOpsCFConv, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<const IValue&,    // neighbors
                         int64_t,          // nunAtoms
                         int64_t,          // numFilters
                         int64_t,          // numGausians
                         double,           // gaussianWidth
                         int64_t,          // activation
                         const Tensor&,    // linear1_weights
                         const Tensor&,    // linear1_biases
                         const Tensor&,    // linear2_weights
                         const Tensor&,    // linear2_biases
                         const Tensor&>()) // positions
        .def("forward", &Holder::forward)
        .def("backward", &Holder::backward)
        .def("is_initialized", &Holder::is_initialized)
        .def_pickle(
            // __getstate__
            // Note: nothing is done during serialization
            [](const HolderPtr& self) -> int64_t { return 0; },
            // __setstate__
            // Note: a new uninitialized object is create during deserialization
            [](int64_t state) -> HolderPtr { return HolderPtr::make(); }
        );
    m.def("operation", operation);
}

} // namespace CFConv
} // namespace NNPOps