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
#include <torch/script.h>
#include <torch/serialize/archive.h>
#include <c10/cuda/CUDAStream.h>
#include "CpuANISymmetryFunctions.h"
#include "CudaANISymmetryFunctions.h"

#define CHECK_CUDA_RESULT(result) \
    if (result != cudaSuccess) { \
        throw std::runtime_error(std::string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+std::to_string(__LINE__));\
    }

namespace NNPOps {
namespace ANISymmetryFunctions {

class Holder;
using Context = torch::autograd::AutogradContext;
using HolderPtr = torch::intrusive_ptr<Holder>;
using std::string;
using std::vector;
using torch::autograd::tensor_list;
using torch::Device;
using torch::IValue;
using torch::optional;
using torch::Tensor;
using torch::TensorOptions;

class Holder : public torch::CustomClassHolder {
public:
    Holder(int64_t numSpecies,
           double Rcr,
           double Rca,
           const vector<double>& EtaR,
           const vector<double>& ShfR,
           const vector<double>& EtaA,
           const vector<double>& Zeta,
           const vector<double>& ShfA,
           const vector<double>& ShfZ,
           const vector<int64_t>& atomSpecies) :

        torch::CustomClassHolder(),
        numSpecies(numSpecies),
        Rcr(Rcr), Rca(Rca),
        EtaR(EtaR), ShfR(ShfR), EtaA(EtaA), Zeta(Zeta), ShfA(ShfA), ShfZ(ShfZ),
        atomSpecies(atomSpecies),
        device(torch::kCPU) {};

    tensor_list forward(const Tensor& positions_, const optional<Tensor>& periodicBoxVectors_) {

        device = positions_.device();

        const TensorOptions tensorOptions = TensorOptions().device(device); // Data type of float by default

        const Tensor positions = positions_.to(tensorOptions);

        if (!impl) {

            int numAtoms = atomSpecies.size();
            const vector<int> atomSpecies_(atomSpecies.begin(), atomSpecies.end()); // vector<int64_t> --> vector<int>

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

            if (device.is_cpu())
                impl = std::make_shared<CpuANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies_, radialFunctions, angularFunctions, true);
            if (device.is_cuda()) {
                // PyTorch allow to chose GPU with "torch.device", but it doesn't set as the default one.
                CHECK_CUDA_RESULT(cudaSetDevice(device.index()));
                impl = std::make_shared<CudaANISymmetryFunctions>(numAtoms, numSpecies, Rcr, Rca, false, atomSpecies_, radialFunctions, angularFunctions, true);
            }

            radial  = torch::empty({numAtoms, numSpecies * (int)radialFunctions.size()}, tensorOptions);
            angular = torch::empty({numAtoms, numSpecies * (numSpecies + 1) / 2 * (int)angularFunctions.size()}, tensorOptions);
            positionsGrad = torch::empty({numAtoms, 3}, tensorOptions);

            cudaImpl = dynamic_cast<CudaANISymmetryFunctions*>(impl.get());
        }

        Tensor periodicBoxVectors;
        float* periodicBoxVectorsPtr = nullptr;
        if (periodicBoxVectors_) {
            periodicBoxVectors = periodicBoxVectors_->to(tensorOptions);
            float* periodicBoxVectorsPtr = periodicBoxVectors.data_ptr<float>();
        }

        if (cudaImpl) {
            const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(device.index());
            cudaImpl->setStream(stream.stream());
        }

        impl->computeSymmetryFunctions(positions.data_ptr<float>(), periodicBoxVectorsPtr, radial.data_ptr<float>(), angular.data_ptr<float>());

        return {radial, angular};
    };

    tensor_list backward(const tensor_list& grads) {

        const Tensor radialGrad = grads[0].clone();
        const Tensor angularGrad = grads[1].clone();

        if (cudaImpl) {
            const torch::cuda::CUDAStream stream = torch::cuda::getCurrentCUDAStream(device.index());
            cudaImpl->setStream(stream.stream());
        }

        impl->backprop(radialGrad.data_ptr<float>(), angularGrad.data_ptr<float>(), positionsGrad.data_ptr<float>());

        return { Tensor(), positionsGrad, Tensor() }; // empty grad for the holder and periodicBoxVectors
    };

    bool is_initialized() {
        return bool(impl);
    };

    static const string serialize(const HolderPtr& self) {

        torch::serialize::OutputArchive archive;
        archive.write("numSpecies", self->numSpecies);
        archive.write("Rcr", self->Rcr);
        archive.write("Rca", self->Rca);
        archive.write("EtaR", self->EtaR);
        archive.write("ShfR", self->ShfR);
        archive.write("EtaA", self->EtaA);
        archive.write("Zeta", self->Zeta);
        archive.write("ShfA", self->ShfA);
        archive.write("ShfZ", self->ShfZ);
        archive.write("atomSpecies", self->atomSpecies);

        std::stringstream stream;
        archive.save_to(stream);
        return stream.str();
    };

    static HolderPtr deserialize(const string& state) {

        std::stringstream stream(state);
        torch::serialize::InputArchive archive;
        archive.load_from(stream, torch::kCPU);

        IValue numSpecies, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, atomSpecies;
        archive.read("numSpecies", numSpecies);
        archive.read("Rcr", Rcr);
        archive.read("Rca", Rca);
        archive.read("EtaR", EtaR);
        archive.read("ShfR", ShfR);
        archive.read("EtaA", EtaA);
        archive.read("Zeta", Zeta);
        archive.read("ShfA", ShfA);
        archive.read("ShfZ", ShfZ);
        archive.read("atomSpecies", atomSpecies);

        return HolderPtr::make(numSpecies.toInt(), Rcr.toDouble(), Rca.toDouble(),
                               EtaR.toDoubleVector(), ShfR.toDoubleVector(), EtaA.toDoubleVector(),
                               Zeta.toDoubleVector(), ShfA.toDoubleVector(), ShfZ.toDoubleVector(),
                               atomSpecies.toIntVector());
    }

private:
    int numSpecies;
    double Rcr, Rca;
    vector<double> EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ;
    vector<int64_t> atomSpecies;
    torch::TensorOptions tensorOptions;
    Device device;
    std::shared_ptr<::ANISymmetryFunctions> impl;
    Tensor radial;
    Tensor angular;
    Tensor positionsGrad;
    CudaANISymmetryFunctions* cudaImpl;
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
        ctx->saved_data.erase("holder");

        return holder->backward(grads);
    };
};

tensor_list operation(const optional<HolderPtr>& holder,
                      const Tensor& positions,
                      const optional<Tensor>& periodicBoxVectors) {

    return AutogradFunctions::apply(*holder, positions, periodicBoxVectors);
}

TORCH_LIBRARY(NNPOpsANISymmetryFunctions, m) {
    m.class_<Holder>("Holder")
        .def(torch::init<int64_t,                   // numSpecies
                         double,                    // Rcr
                         double,                    // Rca
                         const vector<double>&,     // EtaR
                         const vector<double>&,     // ShfR
                         const vector<double>&,     // EtaA
                         const vector<double>&,     // Zeta
                         const vector<double>&,     // ShfA
                         const vector<double>&,     // ShfZ
                         const vector<int64_t>&>()) // atomSpecies
        .def("forward", &Holder::forward)
        .def("backward", &Holder::backward)
        .def("is_initialized", &Holder::is_initialized)
        .def_pickle(
            [](const HolderPtr& self) -> const string { return Holder::serialize(self); }, // __getstate__
            [](const string& state) -> HolderPtr { return Holder::deserialize(state); }    // __setstate__
        );
    m.def("operation", operation);
}

} // namespace ANISymmetryFunctions
} // namespace NNPOps