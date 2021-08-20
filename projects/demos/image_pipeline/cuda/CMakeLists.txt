cmake_minimum_required(VERSION 3.17)
project(image_pipeline_cuda LANGUAGES CXX CUDA)
find_package(CUDAToolkit)
include_directories("/usr/local/include" "${CUDA_DIR}/include")
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    )

add_executable(image_pipeline main.cpp opencv_interface.cpp image_pipeline.cu)
target_link_libraries(image_pipeline PRIVATE cudart ${OpenCV_LIBS})
