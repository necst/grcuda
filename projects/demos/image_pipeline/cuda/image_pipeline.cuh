#pragma once
#include <chrono>
#include <iostream>
#include <string>

#include "options.hpp"
#include "utils.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

class ImagePipeline {
   public:
    ImagePipeline(Options &options) : debug(options.debug),
                                      do_prefetch(options.prefetch),
                                      stream_attach(options.stream_attach),
                                      policy(options.policy_choice) {
        cudaDeviceGetAttribute(&pascalGpu, cudaDeviceAttr::cudaDevAttrConcurrentManagedAccess, 0);
        if (debug) {
            std::cout << "------------------------------" << std::endl;
            std::cout << "- policy=" << options.policy_map[policy] << std::endl;
            std::cout << "- block size 1d=" << options.block_size_1d << std::endl;
            std::cout << "- block size 2d=" << options.block_size_2d << std::endl;
            std::cout << "- num blocks=" << options.num_blocks << std::endl;
            std::cout << "------------------------------" << std::endl;
        }
        dim3 grid_size_2d(options.num_blocks, options.num_blocks);
        dim3 grid_size_1d(options.num_blocks * 2);
        dim3 block_size_2d(options.block_size_2d, options.block_size_2d);
        dim3 block_size_1d(options.block_size_1d);
    }
    void alloc();
    void init();
    void execute_sync();
    void execute_async();
    void run();
    std::string print_result(bool short_form = false);

   private:
    // General configuration settings;
    int debug = DEBUG;
    bool do_prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    int pascalGpu = 0;
    Policy policy;
    int err = 0;
    dim3 grid_size_2d;
    dim3 grid_size_1d;
    dim3 block_size_2d;
    dim3 block_size_1d;

    // Computation-specific settings;
    int N = 1024;
    int kernel_small_diameter = 3;
    int kernel_large_diameter = 7;
    int kernel_unsharpen_diameter = 3;
    float kernel_small_variance = 0.1;
    float kernel_large_variance = 20;
    float kernel_unsharpen_variance = 5;
    float unsharpen_amount = 30;

    int *image, *image3;
    float *image2, *image_unsharpen, *mask_small, *mask_large, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum_1, *minimum_1, *maximum_2, *minimum_2;
    cudaStream_t s1, s2, s3, s4, s5;

    inline void gaussian_kernel(float *kernel, int diameter, float sigma) {
        int mean = diameter / 2;
        float sum_tmp = 0;
        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
                kernel[i * diameter + j] = exp(-0.5 * ((i - mean) * (i - mean) + (j - mean) * (j - mean)) / (sigma * sigma));
                sum_tmp += kernel[i * diameter + j];
            }
        }
        for (int i = 0; i < diameter; i++) {
            for (int j = 0; j < diameter; j++) {
                kernel[i * diameter + j] /= sum_tmp;
            }
        }
    }
};
