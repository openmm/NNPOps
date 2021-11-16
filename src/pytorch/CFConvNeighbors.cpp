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

#define int2 int[2]
#define float3 float[3]

#define CHECK_CUDA_RESULT(result) \
    if (result != cudaSuccess) { \
        throw std::runtime_error(std::string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+std::to_string(__LINE__));\
    }

namespace NNPOps {
namespace CFConvNeighbors {

class Holder;
using std::vector;
using HolderPtr = torch::intrusive_ptr<Holder>;
using torch::Device;
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
           double cutoff,
           const Device& device) : torch::CustomClassHolder() {

        // Construct an uninitialized object
        // Note: this is needed for Python bindings
        if (numAtoms == 0)
            return;

        tensorOptions = torch::TensorOptions().device(device); // Data type of float by default

        if (device.is_cpu())
            neighbors = std::make_shared<CpuCFConvNeighbors>(numAtoms, cutoff, false);
        if (device.is_cuda()) {
            // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
            CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
            neighbors = std::make_shared<CudaCFConvNeighbors>(numAtoms, cutoff, false);
        }

        // cudaNeighbors = dynamic_cast<CudaCFConvNeighbors*>(neighbors.get());
    };

    void build(const Tensor& positions_) {

        const Tensor positions = positions_.to(tensorOptions).clone();

        // if (cudaNeighbors) {
        //     const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(tensorOptions.device().index());
        //     cudaNeighbors->setStream(stream.stream());
        // }

        neighbors->build(positions.data_ptr<float>(), nullptr);
    };

    bool is_initialized() {
        return bool(neighbors);
    };

private:
    torch::TensorOptions tensorOptions;
    std::shared_ptr<::CFConvNeighbors> neighbors;
    // CudaCFConvNeighbors* cudaNeighbors;
};

TORCH_LIBRARY(NNPOpsCFConvNeighbors, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<int64_t,          // nunAtoms
                         double,           // cutoff
                         const Device&>()) // device
        .def("build", &Holder::build)
        .def("is_initialized", &Holder::is_initialized)
        .def_pickle(
            // __getstate__
            // Note: nothing is done during serialization
            [](const HolderPtr& self) -> int64_t { return 0; },
            // __setstate__
            // Note: a new uninitialized object is create during deserialization
            [](int64_t state) -> HolderPtr { return HolderPtr::make(); }
        );
}

} // namespace CFConvNeighbors
} // namespace NNPOps