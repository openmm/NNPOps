cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(NAME NNPOps)
set(LIBRARY ${NAME}PyTorch)
project(${NAME} LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

add_library(${LIBRARY} SHARED src/pytoch/SymmetryFunctions.cpp
                              src/ani/CpuANISymmetryFunctions.cpp
                              src/ani/CudaANISymmetryFunctions.cu)
target_compile_features(${LIBRARY} PRIVATE cxx_std_14)
target_include_directories(${LIBRARY} PRIVATE ${PYTHON_INCLUDE_DIRS} src/ani)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

install(TARGETS ${LIBRARY} DESTINATION ${Python_SITEARCH}/${NAME})
install(FILES SymmetryFunctions.py DESTINATION ${Python_SITEARCH}/${NAME})