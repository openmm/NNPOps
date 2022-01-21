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
#ifndef NNPOPS_CFCONVNEIGHBORS
#define NNPOPS_CFCONVNEIGHBORS

#include <torch/script.h>
#include "CFConv.h"

namespace NNPOps {
namespace CFConvNeighbors {

using torch::Device;
using torch::Tensor;

class Holder : public torch::CustomClassHolder {
public:
    Holder(double cutoff);
    void build(const Tensor& positions);
    double getCutoff() const { return cutoff; }
    const ::CFConvNeighbors& getImpl() const { return *impl.get(); }
private:
    double cutoff;
    int numAtoms;
    Device device;
    std::shared_ptr<::CFConvNeighbors> impl;
};

} // namespace CFConvNeighbors
} // namespace NNPOps

#endif