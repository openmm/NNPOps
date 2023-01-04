cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Configure
set(NAME NNPOps)
project(${NAME} LANGUAGES CXX)
option(ENABLE_CUDA "Enable biuilding CUDA components" ON)
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
              src/pytorch/CFConv.cpp
              src/pytorch/CFConvNeighbors.cpp
              src/pytorch/SymmetryFunctions.cpp
              src/pytorch/messages/messages.cpp
              src/pytorch/messages/passMessagesCPU.cpp
              src/pytorch/messages/passMessagesCUDA.cu
              src/pytorch/neighbors/getNeighborPairsCPU.cpp
              src/pytorch/neighbors/getNeighborPairsCUDA.cu
              src/pytorch/neighbors/neighbors.cpp
              src/schnet/CpuCFConv.cpp
              src/schnet/CudaCFConv.cu)

# Build the library
set(LIBRARY ${NAME}PyTorch)
add_library(${LIBRARY} SHARED ${SRC_FILES})
set_property(TARGET ${LIBRARY} PROPERTY CXX_STANDARD 14)
target_include_directories(${LIBRARY} PRIVATE ${PYTHON_INCLUDE_DIRS}
                                              src/ani src/pytorch src/schnet)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
if(ENABLE_CUDA)
    set_property(TARGET ${LIBRARY} PROPERTY CUDA_STANDARD 14)
    target_compile_definitions(${LIBRARY} PRIVATE ENABLE_CUDA)
endif(ENABLE_CUDA)

# Tests of the library
set(TEST_PATHS src/ani/TestCpuANISymmetryFunctions.cpp
               src/schnet/TestCpuCFConv.cpp)
if(ENABLE_CUDA)
    list(APPEND TEST_PATHS src/ani/TestCudaANISymmetryFunctions.cu
                           src/schnet/TestCudaCFConv.cu)
endif(ENABLE_CUDA)
foreach(TEST_PATH ${TEST_PATHS})
    cmake_path(GET TEST_PATH STEM TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_PATH})
    set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 14)
    if(ENABLE_CUDA)
        set_property(TARGET ${TEST_NAME} PROPERTY CUDA_STANDARD 14)
    endif(ENABLE_CUDA)
    target_link_libraries(${TEST_NAME} ${LIBRARY})
    add_test(${TEST_NAME} ${TEST_NAME})
endforeach()

# Tests of PyTorch wrappers
add_test(TestBatchedNN         pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestBatchedNN.py)
add_test(TestCFConv            pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestCFConv.py)
add_test(TestCFConvNeighbors   pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestCFConvNeighbors.py)
add_test(TestEnergyShifter     pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestEnergyShifter.py)
add_test(TestOptimizedTorchANI pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestOptimizedTorchANI.py)
add_test(TestSpeciesConverter  pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestSpeciesConverter.py)
add_test(TestSymmetryFunctions pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestSymmetryFunctions.py)
add_test(TestMessages          pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/messages/TestMessages.py)
add_test(TestNeighbors         pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/neighbors/TestNeighbors.py)
add_test(TestPassMessages      pytest -v --doctest-modules ${CMAKE_SOURCE_DIR}/src/pytorch/messages/passMessages.py)
add_test(TestGetNeighborPairs  pytest -v --doctest-modules ${CMAKE_SOURCE_DIR}/src/pytorch/neighbors/getNeighborPairs.py)

# Installation
install(TARGETS ${LIBRARY} DESTINATION ${Python_SITEARCH}/${NAME})
install(FILES src/pytorch/__init__.py
              src/pytorch/BatchedNN.py
              src/pytorch/CFConv.py
              src/pytorch/CFConvNeighbors.py
              src/pytorch/EnergyShifter.py
              src/pytorch/OptimizedTorchANI.py
              src/pytorch/SpeciesConverter.py
              src/pytorch/SymmetryFunctions.py
        DESTINATION ${Python_SITEARCH}/${NAME})
install(FILES src/pytorch/messages/__init__.py
              src/pytorch/messages/passMessages.py
        DESTINATION ${Python_SITEARCH}/${NAME}/messages)
install(FILES src/pytorch/neighbors/__init__.py
              src/pytorch/neighbors/getNeighborPairs.py
        DESTINATION ${Python_SITEARCH}/${NAME}/neighbors)