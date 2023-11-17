cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Configure
set(NAME NNPOps)
project(${NAME} LANGUAGES CXX)
option(ENABLE_CUDA "Enable biuilding CUDA components" ON)
if(ENABLE_CUDA)
    enable_language(CUDA)
endif(ENABLE_CUDA)

# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
enable_testing()

# Source files of the library
set(SRC_FILES src/ani/CpuANISymmetryFunctions.cpp
              src/ani/CudaANISymmetryFunctions.cu
              src/pytorch/BatchedNN.cpp
              src/pytorch/CFConv.cpp
              src/pytorch/CFConvNeighbors.cpp
              src/pytorch/SymmetryFunctions.cpp
              src/pytorch/neighbors/getNeighborPairsCPU.cpp
              src/pytorch/neighbors/getNeighborPairsCUDA.cu
              src/pytorch/neighbors/neighbors.cpp
              src/pytorch/pme/pmeCPU.cpp
              src/pytorch/pme/pmeCUDA.cu
              src/pytorch/pme/pme.cpp
              src/schnet/CpuCFConv.cpp
              src/schnet/CudaCFConv.cu)

# Build the library
set(LIBRARY ${NAME}PyTorch)
add_library(${LIBRARY} SHARED ${SRC_FILES})
# Specify the C++ version we are building for. Latest pytorch versions require C++17
message(STATUS "Found Torch: ${Torch_VERSION}")
if(${Torch_VERSION} VERSION_GREATER_EQUAL "2.1.0")
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Setting C++ standard to C++17")
else()
    set(CMAKE_CXX_STANDARD 14)
    message(STATUS "Setting C++ standard to C++14")
endif()

target_include_directories(${LIBRARY} PRIVATE ${Python3_INCLUDE_DIRS}
                                              src/ani src/pytorch src/schnet)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})
if(ENABLE_CUDA)
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
    target_link_libraries(${TEST_NAME} ${LIBRARY})
    add_test(${TEST_NAME} ${TEST_NAME})
endforeach()

# List of directories containing  test scripts
set(PYTHON_TEST_DIRS
  ${CMAKE_SOURCE_DIR}/src/pytorch/
  ${CMAKE_SOURCE_DIR}/src/pytorch/neighbors
  ${CMAKE_SOURCE_DIR}/src/pytorch/pme
)

# Move test scripts to a test folder in the build directory, create test folder if necessary
add_custom_target(copy_test ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test
)

# Add the data directory as well
add_custom_command(
  TARGET copy_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/src/pytorch/molecules
  ${CMAKE_BINARY_DIR}/test/molecules
)

# Copy each test to the binary dir and add it
foreach(TEST_DIR ${PYTHON_TEST_DIRS})
  file(GLOB TEST_SCRIPTS ${TEST_DIR}/Test*.py)
  foreach(TEST_SCRIPT ${TEST_SCRIPTS})
    get_filename_component(TEST_NAME ${TEST_SCRIPT} NAME_WE)
    file(RELATIVE_PATH RELATIVE_TEST_PATH ${CMAKE_SOURCE_DIR}/src/pytorch ${TEST_SCRIPT})
    add_custom_command(
      TARGET copy_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SCRIPT} ${CMAKE_BINARY_DIR}/test/${RELATIVE_TEST_PATH}
    )
    add_test(NAME ${TEST_NAME} COMMAND pytest -v ${CMAKE_BINARY_DIR}/test/${RELATIVE_TEST_PATH})
  endforeach()
endforeach()

add_custom_command(
  TARGET copy_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/src/pytorch/neighbors/getNeighborPairs.py ${CMAKE_BINARY_DIR}/test/getNeighborPairs.py
)

add_test(TestGetNeighborPairs  pytest -v --doctest-modules ${CMAKE_BINARY_DIR}/test/getNeighborPairs.py)

# Installation
install(TARGETS ${LIBRARY} DESTINATION ${Python3_SITEARCH}/${NAME})
install(FILES src/pytorch/__init__.py
              src/pytorch/BatchedNN.py
              src/pytorch/CFConv.py
              src/pytorch/CFConvNeighbors.py
              src/pytorch/EnergyShifter.py
              src/pytorch/OptimizedTorchANI.py
              src/pytorch/SpeciesConverter.py
              src/pytorch/SymmetryFunctions.py
        DESTINATION ${Python3_SITEARCH}/${NAME})
install(FILES src/pytorch/neighbors/__init__.py
              src/pytorch/neighbors/getNeighborPairs.py
        DESTINATION ${Python3_SITEARCH}/${NAME}/neighbors)
install(FILES src/pytorch/pme/__init__.py
              src/pytorch/pme/pme.py
        DESTINATION ${Python3_SITEARCH}/${NAME}/pme)
