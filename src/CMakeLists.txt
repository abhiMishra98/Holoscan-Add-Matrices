cmake_minimum_required(VERSION 3.18)
project(cudaTest LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(Holoscan REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cudaTest
    cudaTest.cpp
    cudaKernel.cu
)

target_link_libraries(cudaTest PRIVATE
    holoscan::core
    holoscan::ucp
    holoscan::ucxx
    holoscan::infer
    holoscan::logger
    CUDA::cudart
)

target_compile_features(cudaTest PRIVATE cxx_std_17)

set_target_properties(cudaTest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

set_source_files_properties(src/cudaKernel.cu PROPERTIES LANGUAGE CUDA)
