#ifndef NNPOPS_ACCESSOR_H
#define NNPOPS_ACCESSOR_H

#include <torch/extension.h>

template <typename scalar_t, int num_dims>
    using Accessor = torch::PackedTensorAccessor32<scalar_t, num_dims, torch::RestrictPtrTraits>;

template <typename scalar_t, int num_dims>
inline Accessor<scalar_t, num_dims> get_accessor(const torch::Tensor& tensor) {
    return tensor.packed_accessor32<scalar_t, num_dims, torch::RestrictPtrTraits>();
};

#endif