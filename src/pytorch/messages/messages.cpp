#include <torch/extension.h>

TORCH_LIBRARY(messages, m) {
    m.def("passMessages(Tensor neighbors, Tensor messages, Tensor states) -> (Tensor states)");
}