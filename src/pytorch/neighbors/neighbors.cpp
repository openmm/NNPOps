#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("getNeighborPairs(Tensor positions, Scalar cutoff, Scalar max_num_neighbors, Tensor box_vectors, bool checkErrors) -> (Tensor neighbors, Tensor deltas, Tensor distances, Tensor num_pairs)");
}
