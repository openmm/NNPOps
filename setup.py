# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
from os.path import join as opj


# If ENABLE_CUDA is defined
enable_cuda_env = os.environ.get("ENABLE_CUDA")
if enable_cuda_env is not None:
    if enable_cuda_env not in ("0", "1"):
        raise ValueError(f"ENABLE_CUDA must be 0 or 1, got {enable_cuda_env}")
    use_cuda = enable_cuda_env == "1"
else:
    use_cuda = torch.cuda._is_compiled()


def set_torch_cuda_arch_list():
    """Set the CUDA arch list according to the architectures the current torch installation was compiled for.
    This function is a no-op if the environment variable TORCH_CUDA_ARCH_LIST is already set or if torch was not compiled with CUDA support.
    """
    if use_cuda and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        arch_flags = torch._C._cuda_getArchFlags()
        sm_versions = [x[3:] for x in arch_flags.split() if x.startswith("sm_")]
        formatted_versions = ";".join([f"{y[0]}.{y[1]}" for y in sm_versions])
        formatted_versions += "+PTX"
        os.environ["TORCH_CUDA_ARCH_LIST"] = formatted_versions


set_torch_cuda_arch_list()

sources = [
    opj("src", "ani", "CpuANISymmetryFunctions.cpp"),
    opj("src", "NNPOps", "BatchedNN.cpp"),
    opj("src", "NNPOps", "CFConv.cpp"),
    opj("src", "NNPOps", "CFConvNeighbors.cpp"),
    opj("src", "NNPOps", "SymmetryFunctions.cpp"),
    opj("src", "NNPOps", "neighbors", "getNeighborPairsCPU.cpp"),
    opj("src", "NNPOps", "neighbors", "neighbors.cpp"),
    opj("src", "NNPOps", "pme", "pmeCPU.cpp"),
    opj("src", "NNPOps", "pme", "pme.cpp"),
    opj("src", "schnet", "CpuCFConv.cpp"),
]
if use_cuda:
    sources += [
        opj("src", "ani", "CudaANISymmetryFunctions.cu"),
        opj("src", "NNPOps", "neighbors", "getNeighborPairsCUDA.cu"),
        opj("src", "NNPOps", "pme", "pmeCUDA.cu"),
        opj("src", "schnet", "CudaCFConv.cu"),
    ]


ExtensionType = CppExtension if not use_cuda else CUDAExtension
extensions = ExtensionType(
    name="NNPOps.libNNPOpsPyTorch",
    sources=sources,
    include_dirs=[
        opj("src", "ani"),
        opj("src", "NNPOps"),
        opj("src", "NNPOps", "common"),
        opj("src", "NNPOps", "neighbors"),
        opj("src", "NNPOps", "pme"),
        opj("src", "schnet"),
    ],
    define_macros=[("ENABLE_CUDA", 1)] if use_cuda else [],
)

if __name__ == "__main__":
    setup(
        ext_modules=[extensions],
        cmdclass={
            "build_ext": BuildExtension.with_options(
                no_python_abi_suffix=True, use_ninja=False
            )
        },
    )
