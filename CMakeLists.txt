cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Configure
set(NAME NNPOps)
project(${NAME} LANGUAGES CXX)
option(ENABLE_CUDA "Enable biuilding CUDA components" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
endif(ENABLE_CUDA)

# Find dependencies
find_package(Python REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
enable_testing()

# Source files of the library
set(SRC_FILES src/ani/CpuANISymmetryFunctions.cpp
              src/ani/CudaANISymmetryFunctions.cu
              src/pytorch/BatchedNN.cpp
              src/pytorch/SymmetryFunctions.cpp
              src/schnet/CpuCFConv.cpp
              src/schnet/CudaCFConv.cu)

# Build the library
set(LIBRARY ${NAME}PyTorch)
add_library(${LIBRARY} SHARED ${SRC_FILES})
target_include_directories(${LIBRARY} PRIVATE src/ani src/schnet)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
if(ENABLE_CUDA)
    target_compile_definitions(${LIBRARY} PRIVATE ENABLE_CUDA)
endif(ENABLE_CUDA)

# Test of the library
set(TEST_PATHS src/ani/TestCpuANISymmetryFunctions.cpp
               src/schnet/TestCpuCFConv.cpp)
if(ENABLE_CUDA)
    list(APPEND TEST_PATHS src/ani/TestCudaANISymmetryFunctions.cpp
                           src/schnet/TestCudaCFConv.cpp)
endif(ENABLE_CUDA)
foreach(TEST_PATH ${TEST_PATHS})
    cmake_path(GET TEST_PATH STEM TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_PATH})
    target_link_libraries(${TEST_NAME} ${LIBRARY})
    add_test(${TEST_NAME} ${TEST_NAME})
endforeach()

# Tests of PyTorch wrappers
add_test(TestBatchedNN         pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestBatchedNN.py)
add_test(TestSpeciesConverter  pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestSpeciesConverter.py)
add_test(TestEnergyShifter     pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestEnergyShifter.py)
add_test(TestSymmetryFunctions pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestSymmetryFunctions.py)

# Installation
install(TARGETS ${LIBRARY} DESTINATION ${Python_SITEARCH}/${NAME})
install(FILES src/pytorch/__init__.py
              src/pytorch/BatchedNN.py
              src/pytorch/SpeciesConverter.py
              src/pytorch/EnergyShifter.py
              src/pytorch/SymmetryFunctions.py
        DESTINATION ${Python_SITEARCH}/${NAME})