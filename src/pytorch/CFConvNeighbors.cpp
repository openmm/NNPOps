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
#include "CFConvNeighbors.h"
#include "CpuCFConv.h"

#ifdef ENABLE_CUDA
#include <stdexcept>
#include <cuda_runtime.h>
// #include <c10/cuda/CUDAStream.h>
#include "CudaCFConv.h"

#define int2 int[2]
#define float3 float[3]

#define CHECK_CUDA_RESULT(result) \
    if (result != cudaSuccess) { \
        throw std::runtime_error(std::string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+std::to_string(__LINE__));\
    }
#endif

namespace NNPOps {
namespace CFConvNeighbors {

using HolderPtr = torch::intrusive_ptr<Holder>;
using torch::Tensor;

Holder::Holder(double cutoff) : torch::CustomClassHolder(), cutoff(cutoff), numAtoms(0), device(torch::kCPU), impl(nullptr) {}

void Holder::build(const Tensor& positions) {

    if (positions.scalar_type() != torch::kFloat32)
        throw std::runtime_error("The type of \"positions\" has to be float32");
    if (positions.dim() != 2)
        throw std::runtime_error("The shape of \"positions\" has to have 2 dimensions");
    if (positions.size(1) != 3)
        throw std::runtime_error("The size of the 2nd dimension of \"positions\" has to be 3");

    if (!impl) {
        numAtoms = positions.size(0);
        device = positions.device();
        if (device.is_cpu()) {
            impl = std::make_shared<::CpuCFConvNeighbors>(numAtoms, cutoff, false);
#ifdef ENABLE_CUDA
        } else if (device.is_cuda()) {
            // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
            CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
            impl = std::make_shared<::CudaCFConvNeighbors>(numAtoms, cutoff, false);
#endif
        } else
            throw std::runtime_error("Unsupported device");
    }

    if (positions.size(0) != numAtoms)
        throw std::runtime_error("The size of the 2nd dimension of \"positions\" has changed");
    if (positions.device() != device)
        throw std::runtime_error("The device of \"positions\" has changed");

    // ::CudaCFConvNeighbors* cudaImpl = dynamic_cast<CudaCFConvNeighbors*>(impl.get());
    // if (cudaImpl) {
    //     const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(positions.device().index());
    //     cudaImpl->setStream(stream.stream());
    // }

    impl->build(positions.data_ptr<float>(), nullptr);
}

TORCH_LIBRARY(NNPOpsCFConvNeighbors, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<double>())
        .def("build", &Holder::build)
        .def_pickle(
            [](const HolderPtr& self) -> double { return self->getCutoff(); }, // __getstate__
            [](double cutoff) -> HolderPtr { return HolderPtr::make(cutoff); } // __setstate__
        );
}

} // namespace CFConvNeighbors
} // namespace NNPOps