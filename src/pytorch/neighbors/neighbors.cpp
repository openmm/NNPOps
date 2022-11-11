#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("getNeighborPairs(Tensor positions, Scalar cutoff, Scalar max_num_neighbors, Tensor box_vectors) -> (Tensor neighbors, Tensor deltas, Tensor distances)");
}