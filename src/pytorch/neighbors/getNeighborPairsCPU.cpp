#include <torch/extension.h>
#include <tuple>

using std::tuple;
using torch::div;
using torch::full;
using torch::index_select;
using torch::indexing::Slice;
using torch::arange;
using torch::frobenius_norm;
using torch::kInt32;
using torch::Scalar;
using torch::hstack;
using torch::vstack;
using torch::Tensor;

static tuple<Tensor, Tensor, Tensor> forward(const Tensor& positions,
                                             const Scalar& cutoff,
                                             const Scalar& max_num_neighbors) {

    TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
    TORCH_CHECK(positions.size(0) > 0, "Expected the 1nd dimension size of \"positions\" to be more than 0");
    TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
    TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

    TORCH_CHECK(cutoff.to<double>() > 0, "Expected \"cutoff\" to be positive");

    const int max_num_neighbors_ = max_num_neighbors.to<int>();
    TORCH_CHECK(max_num_neighbors_ > 0 || max_num_neighbors_ == -1,
        "Expected \"max_num_neighbors\" to be positive or equal to -1");

    const int num_atoms = positions.size(0);
    const int num_pairs = num_atoms * (num_atoms - 1) / 2;

    const Tensor indices = arange(0, num_pairs, positions.options().dtype(kInt32));
    Tensor rows = (((8 * indices + 1).sqrt() + 1) / 2).floor().to(kInt32);
    rows -= (rows * (rows - 1) > 2 * indices).to(kInt32);
    const Tensor columns = indices - div(rows * (rows - 1), 2, "floor");

    Tensor neighbors = vstack({rows, columns});
    Tensor deltas = index_select(positions, 0, rows) - index_select(positions, 0, columns);
    Tensor distances = frobenius_norm(deltas, 1);

    if (max_num_neighbors_ == -1) {
        const Tensor mask = distances > cutoff;
        neighbors.index_put_({Slice(), mask}, -1);
        deltas = deltas.clone(); // Brake an autograd loop
        deltas.index_put_({mask, Slice()}, NAN);
        distances.index_put_({mask}, NAN);

    } else {
        const Tensor mask = distances <= cutoff;
        neighbors = neighbors.index({Slice(), mask});
        deltas = deltas.index({mask, Slice()});
        distances = distances.index({mask});

        const int num_pad = num_atoms * max_num_neighbors_ - distances.size(0);
        TORCH_CHECK(num_pad >= 0,
            "The maximum number of pairs has been exceed! Increase \"max_num_neighbors\"");

        if (num_pad > 0) {
            neighbors = hstack({neighbors, full({2, num_pad}, -1, neighbors.options())});
            deltas = vstack({deltas, full({num_pad, 3}, NAN, deltas.options())});
            distances = hstack({distances, full({num_pad}, NAN, distances.options())});
        }
    }

    return {neighbors, deltas, distances};
}

TORCH_LIBRARY_IMPL(neighbors, CPU, m) {
    m.impl("getNeighborPairs", &forward);
}