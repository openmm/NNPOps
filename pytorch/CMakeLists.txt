cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(NAME NNPOpsPyTorch)
project(${NAME})

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)

add_library(${NAME} SHARED SymmetryFunctions.cpp ../ani/CpuANISymmetryFunctions.cpp)
target_compile_features(${NAME} PRIVATE cxx_std_14)
target_include_directories(${NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(${NAME} PRIVATE ../ani)
target_link_libraries(${NAME} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})