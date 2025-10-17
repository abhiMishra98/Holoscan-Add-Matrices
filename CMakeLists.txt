cmake_minimum_required(VERSION 3.18)
project(Holoscan-Add-Matrices LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(Holoscan REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(Holoscan-Add-Matrices
    src/cudaTest.cpp
    src/cudaKernel.cu
)

target_link_libraries(Holoscan-Add-Matrices PRIVATE
    holoscan::core
    holoscan::ucp
    holoscan::ucxx
    holoscan::infer
    holoscan::logger
    CUDA::cudart
)

target_compile_features(Holoscan-Add-Matrices PRIVATE cxx_std_17)

set_target_properties(Holoscan-Add-Matrices PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

set_source_files_properties(src/cudaKernel.cu PROPERTIES LANGUAGE CUDA)
