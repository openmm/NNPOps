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
using Activation = ::CFConv::ActivationFunction;
using Context = torch::autograd::AutogradContext;
using HolderPtr = torch::intrusive_ptr<Holder>;
using Neighbors = NNPOps::CFConvNeighbors::Holder;
using NeighborsPtr = torch::intrusive_ptr<Neighbors>;
using torch::autograd::tensor_list;
using torch::Device;
using torch::IValue;
using torch::optional;
using torch::Tensor;
using torch::TensorOptions;

class Holder : public torch::CustomClassHolder {
public:

    // Constructor for an uninitialized object
    // Note: this is need for serialization
    Holder() : torch::CustomClassHolder(), device(torch::kCPU) {};

    Holder(double gaussianWidth,
           int64_t activation,
           const Tensor& weights1,
           const Tensor& biases1,
           const Tensor& weights2,
           const Tensor& biases2) :

        torch::CustomClassHolder(),
        gaussianWidth(gaussianWidth),
        activation(static_cast<Activation>(activation)),
        // Note: weights and biases have to be in the CPU memory
        weights1(weights1.to(torch::kFloat32).cpu().clone()),
        biases1(biases1.to(torch::kFloat32).cpu().clone()),
        weights2(weights2.to(torch::kFloat32).cpu().clone()),
        biases2(biases2.to(torch::kFloat32).cpu().clone()),
        device(torch::kCPU)
    {};

    Tensor forward(const IValue& neighbors_, const Tensor& positions, const Tensor& input) {

        neighbors = neighbors_.toCustomClass<Neighbors>(); // save for the backward pass

        this->positions = positions; // save for the backward pass
        if (positions.scalar_type() != torch::kFloat32)
            throw std::runtime_error("The type of \"positions\" has to be float32");
        if (positions.dim() != 2)
            throw std::runtime_error("The shape of \"positions\" has to have 2 dimensions");
        if (positions.size(1) != 3)
            throw std::runtime_error("The size of the 2nd dimension of \"positions\" has to be 3");

        this->input = input; // save for the backward pass
        if (input.device() != positions.device())
            throw std::runtime_error("The device of \"input\" and \"positions\" has to be the same");
        if (input.scalar_type() != torch::kFloat32)
            throw std::runtime_error("The type of \"input\" has to be float32");
        if (input.dim() != 2)
            throw std::runtime_error("The shape of \"input\" has to have 2 dimensions");
        if (input.size(0) != positions.size(0))
            throw std::runtime_error("The size of the 1nd dimension of \"input\" has to be equal to the 1st dimension of \"positions\"");

        if(!conv) {
            device = positions.device();
            numAtoms = positions.size(0);
            numFilters = input.size(1);
            cutoff = neighbors->getCutoff();

            if (weights1.dim() != 2)
                throw std::runtime_error("The shape of \"weights1\" has to have 2 dimensions");
            int64_t numGaussians = weights1.size(0);
            if (weights1.size(1) != numFilters)
                throw std::runtime_error("The size of the 2nd dimension of \"weights1\" has to be equal to the 2st dimension of \"input\"");

            if (biases1.dim() != 1)
                throw std::runtime_error("The shape of \"biases1\" has to have 1 dimension");
            if (biases1.size(0) != numFilters)
                throw std::runtime_error("The size of \"biases1\" has to be equal to the 2st dimension of \"input\"");

            if (weights2.dim() != 2)
                throw std::runtime_error("The shape of \"weights2\" has to have 2 dimensions");
            if (weights2.size(0) != numFilters)
                throw std::runtime_error("The size of the 1nd dimension of \"weights2\" has to be equal to the 2st dimension of \"input\"");
            if (weights2.size(1) != numFilters)
                throw std::runtime_error("The size of the 2nd dimension of \"weights2\" has to be equal to the 2st dimension of \"input\"");

            if (biases2.dim() != 1)
                throw std::runtime_error("The shape of \"biases2\" has to have 1 dimension");
            if (biases2.size(0) != numFilters)
                throw std::runtime_error("The size of \"biases2\" has to be equal to the 2st dimension of \"input\"");

            if (device.is_cpu()) {
                conv = std::make_shared<::CpuCFConv>(numAtoms, numFilters, numGaussians, cutoff, false, gaussianWidth, activation,
                                                     weights1.data_ptr<float>(), biases1.data_ptr<float>(), weights2.data_ptr<float>(), biases2.data_ptr<float>());
            } else if (device.is_cuda()) {
                // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
                CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
                conv = std::make_shared<::CudaCFConv>(numAtoms, numFilters, numGaussians, cutoff, false, gaussianWidth, this->activation,
                                                      weights1.data_ptr<float>(), biases1.data_ptr<float>(), weights2.data_ptr<float>(), biases2.data_ptr<float>());
            } else
                throw std::runtime_error("Unsupported device");

            // Create the output tensors
            const TensorOptions options = torch::TensorOptions().device(device); // Data type of float by default
            output = torch::empty({numAtoms, numFilters}, options);
            inputGrad = torch::empty({numAtoms, numFilters}, options);
            positionsGrad = torch::empty({numAtoms, 3}, options);

            // cudaConv = dynamic_cast<CudaCFConv*>(conv.get());
        }

        if (neighbors->getCutoff() != cutoff)
            throw std::runtime_error("The cutoff of \"neighbors\" has changed");

        if (positions.size(0) != numAtoms)
            throw std::runtime_error("The size of the 1nd dimension of \"positions\" has changed");
        if (positions.device() != device)
            throw std::runtime_error("The device of \"positions\" has changed");

        if (input.size(0) != numAtoms)
            throw std::runtime_error("The size of the 1nd dimension of \"input\" has changed");
        if (input.size(1) != numFilters)
            throw std::runtime_error("The size of the 2nd dimension of \"input\" has changed");
        if (input.device() != device)
            throw std::runtime_error("The device of \"input\" has changed");

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
    Activation activation;
    Tensor biases1;
    Tensor biases2;
    std::shared_ptr<::CFConv> conv;
    // CudaCFConv* cudaConv;
    double cutoff;
    Device device;
    Tensor input;
    Tensor inputGrad;
    double gaussianWidth;
    NeighborsPtr neighbors;
    int64_t numAtoms;
    int64_t numFilters;
    Tensor output;
    Tensor positions;
    Tensor positionsGrad;
    Tensor weights1;
    Tensor weights2;
};

class AutogradFunctions : public torch::autograd::Function<AutogradFunctions> {

public:
    static Tensor forward(Context *ctx,
                          const HolderPtr& holder,
                          const IValue& neighbors,
                          const Tensor& positions,
                          const Tensor& input) {

        ctx->saved_data["holder"] = holder;

        return holder->forward(neighbors, positions, input);
    };

    static tensor_list backward(Context *ctx, const tensor_list& grads) {

        const HolderPtr holder = ctx->saved_data["holder"].toCustomClass<Holder>();
        tensor_list output = holder->backward(grads[0]);
        ctx->saved_data.erase("holder");

        return { Tensor(),   // holder
                 Tensor(),   // neighbors
                 output[0],  // positions
                 output[1]}; // input
    };
};

Tensor operation(const optional<HolderPtr>& holder,
                 const IValue& neighbors,
                 const Tensor& positions,
                 const Tensor& input) {

    return AutogradFunctions::apply(*holder, neighbors, positions, input);
}

TORCH_LIBRARY(NNPOpsCFConv, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<double,           // gaussianWidth
                         int64_t,          // activation
                         const Tensor&,    // weights1
                         const Tensor&,    // biases1
                         const Tensor&,    // weights2
                         const Tensor&>()) // biases2
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