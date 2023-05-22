#include <torch/extension.h>

TORCH_LIBRARY(pme, m) {
    m.def("pme_reciprocal(Tensor positions, Tensor charges, Tensor box_vectors, Scalar gridx, Scalar gridy, Scalar gridz, Scalar order, Scalar alpha, Scalar coulomb, Tensor xmoduli, Tensor ymoduli, Tensor zmoduli) -> Tensor");
}
