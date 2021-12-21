cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(NAME NNPOps)
set(LIBRARY ${NAME}PyTorch)
project(${NAME} LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
enable_testing()

add_library(${LIBRARY} SHARED src/ani/CpuANISymmetryFunctions.cpp
                              src/ani/CudaANISymmetryFunctions.cu
                              src/pytorch/BatchedNN.cpp
                              src/pytorch/CFConv.cpp
                              src/pytorch/CFConvNeighbors.cpp
                              src/pytorch/SymmetryFunctions.cpp
                              src/schnet/CpuCFConv.cpp
                              src/schnet/CudaCFConv.cu)
target_include_directories(${LIBRARY} PRIVATE src/ani src/schnet)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

set(TEST_PATHS src/ani/TestCpuANISymmetryFunctions.cpp
               src/ani/TestCudaANISymmetryFunctions.cpp
               src/schnet/TestCpuCFConv.cpp
               src/schnet/TestCudaCFConv.cu)
foreach(TEST_PATH ${TEST_PATHS})
    cmake_path(GET TEST_PATH STEM TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_PATH})
    target_link_libraries(${TEST_NAME} ${LIBRARY})
    add_test(${TEST_NAME} ${TEST_NAME})
endforeach()

add_test(TestBatchedNN         pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestBatchedNN.py)
add_test(TestCFConv            pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestCFConv.py)
add_test(TestCFConvNeighbors   pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestCFConvNeighbors.py)
add_test(TestEnergyShifter     pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestEnergyShifter.py)
add_test(TestOptimizedTorchANI pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestOptimizedTorchANI.py)
add_test(TestSpeciesConverter  pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestSpeciesConverter.py)
add_test(TestSymmetryFunctions pytest -v ${CMAKE_SOURCE_DIR}/src/pytorch/TestSymmetryFunctions.py)

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