# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
from os.path import join as opj


# If ENABLE_CUDA is defined
if os.environ.get("ENABLE_CUDA", "0") == "1":
    use_cuda = True
elif os.environ.get("ENABLE_CUDA", "0") == "0":
    use_cuda = False
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
    opj("NNPOps", "ani", "CpuANISymmetryFunctions.cpp"),
    opj("NNPOps", "BatchedNN.cpp"),
    opj("NNPOps", "CFConv.cpp"),
    opj("NNPOps", "CFConvNeighbors.cpp"),
    opj("NNPOps", "SymmetryFunctions.cpp"),
    opj("NNPOps", "neighbors", "getNeighborPairsCPU.cpp"),
    opj("NNPOps", "neighbors", "neighbors.cpp"),
    opj("NNPOps", "pme", "pmeCPU.cpp"),
    opj("NNPOps", "pme", "pme.cpp"),
    opj("NNPOps", "schnet", "CpuCFConv.cpp"),
]
if use_cuda:
    sources += [
        opj("NNPOps", "ani", "CudaANISymmetryFunctions.cu"),
        opj("NNPOps", "neighbors", "getNeighborPairsCUDA.cu"),
        opj("NNPOps", "pme", "pmeCUDA.cu"),
        opj("NNPOps", "schnet", "CudaCFConv.cu"),
    ]


ExtensionType = CppExtension if not use_cuda else CUDAExtension
extensions = ExtensionType(
    name="NNPOps.libNNPOpsPyTorch",
    sources=sources,
    include_dirs=[
        "NNPOps",
        opj("NNPOps", "ani"),
        opj("NNPOps", "common"),
        opj("NNPOps", "neighbors"),
        opj("NNPOps", "pme"),
        opj("NNPOps", "schnet"),
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
