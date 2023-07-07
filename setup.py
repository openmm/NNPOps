import os
import re
import subprocess
import sys
from pathlib import Path
#import git
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import torch

from typing import Optional, Dict
# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "", extra_args: Optional[Dict[str, str]] = None) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        #Store a list of extra arguments to pass to CMake, prepend -D to each
        if extra_args is not None:
            print("Extra args: ", extra_args)
        self.extra_args = [f"-D{key}={value}" for key, value in extra_args.items()] if extra_args else []



class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "make")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}{ext.name}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]
        cmake_args += ext.extra_args
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

extra_args = {}

if "CC" in os.environ:
    extra_args["CMAKE_C_COMPILER"] = os.environ.get("CC", "")
if "CXX" in os.environ:
    extra_args["CMAKE_CXX_COMPILER"] = os.environ.get("CXX", "")
# if torch.backends.cuda.is_built():
#     extra_args["ENABLE_CUDA"] = "ON"
#     ARCHES = [52, 60, 61, 70]
#     DEPRECATED_IN_11 = [35, 50]
#     cuda_version_major= int(torch.version.cuda.split(".")[0])
#     cuda_version_minor= int(torch.version.cuda.split(".")[1])
#     if cuda_version_major >= 11 or (cuda_version_major == 11 and cuda_version_minor >= 1):
#         LATEST_ARCH = 90
#         ARCHES += [75, 80, 86]
#     elif cuda_version_major == 11 and cuda_version_minor >= 1:
#         LATEST_ARCH = 86
#         ARCHES += [75, 80]
#     elif cuda_version_major == 11 and cuda_version_minor >= 0:
#         LATEST_ARCH = 80
#         ARCHES += [75]
#     elif cuda_version_major >= 10:
#         LATEST_ARCH = 75
#         ARCHES += DEPRECATED_IN_11
#     else:
#         raise RuntimeError("Unsupported CUDA version")
#     CMAKE_CUDA_ARCHS = ";".join([str(arch) for arch in ARCHES] + [f"{LATEST_ARCH}-real", f"{LATEST_ARCH}-virtual"])
#     extra_args["CMAKE_CUDA_ARCHITECTURES"] = "OFF" #CMAKE_CUDA_ARCHS
# else:
extra_args["CMAKE_CUDA_ARCHITECTURES"] = "OFF"
extra_args["ENABLE_CUDA"] = "OFF"

extra_args["CMAKE_PREFIX_PATH"] = torch.utils.cmake_prefix_path
torch_version = os.environ.get("TORCH_VERSION", torch.__version__)
cuda_version = os.environ.get("CUDA_VERSION", torch.version.cuda)
#tag = git.Repo(search_parent_directories=True).git.describe("--tags", always=True)
#version = tag.lstrip('v').split('-')[0]
setup(
    ext_modules=[CMakeExtension(name="NNPOps", sourcedir=".", extra_args=extra_args)],
    cmdclass={"build_ext": CMakeBuild},
    packages=["NNPOps", "NNPOps.neighbors", "NNPOps.pme"],
    package_dir={
        'NNPOps': 'src/pytorch',
        'NNPOps.neighbors': 'src/pytorch/neighbors',
        'NNPOps.pme': 'src/pytorch/pme'},
    package_data={'NNPOps': ['lib/*.so', 'lib/*.dll', 'lib/*.dylib']},
    install_requires=[f"torch=={torch_version}", f"nvidia-cuda-nvcc-cu{cuda_version}"],
)
