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

using Context = torch::autograd::AutogradContext;
using Tensor = torch::Tensor;
using tensor_list = torch::autograd::tensor_list;

class BatchedLinerFunction : public torch::autograd::Function<BatchedLinerFunction> {
public:
    static Tensor forward(Context* ctx, const Tensor& vectors, const Tensor& weights, const Tensor& biases) {
        ctx->save_for_backward({weights});
        return torch::matmul(weights, vectors) + biases;
    };
    static tensor_list backward(Context *ctx, const tensor_list& grads) {
        const Tensor grad_in = grads[0].squeeze(-1).unsqueeze(-2);
        const Tensor weights = ctx->get_saved_variables()[0];
        const Tensor grad_out = torch::matmul(grad_in, weights).squeeze(-2).unsqueeze(-1);
        return {grad_out, torch::Tensor(), torch::Tensor()};
    };
};

static Tensor BatchedLinear(const Tensor& vector, const Tensor& weights, const Tensor& biases) {
    return BatchedLinerFunction::apply(vector, weights, biases);
}

TORCH_LIBRARY(NNPOpsBatched, m) {
    m.def("BatchedLinear", BatchedLinear);
}