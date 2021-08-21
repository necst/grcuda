#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <cuda_runtime.h> 
#include "options.hpp"
#include "utils.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

class ImagePipeline {
   public:
    ImagePipeline(Options &options) : debug(options.debug),
                                      black_and_white(options.black_and_white),
                                      image_width(options.resized_image_width),
                                      do_prefetch(options.prefetch),
                                      stream_attach(options.stream_attach),
                                      policy(options.policy_choice) {
        if (debug) {
            std::cout << "------------------------------" << std::endl;
            std::cout << "- policy=" << options.policy_map[policy] << std::endl;
            std::cout << "- block size 1d=" << options.block_size_1d << std::endl;
            std::cout << "- block size 2d=" << options.block_size_2d << std::endl;
            std::cout << "- num blocks=" << options.num_blocks << std::endl;
            std::cout << "------------------------------" << std::endl;
        }
        grid_size_2d = dim3(options.num_blocks, options.num_blocks);
        grid_size_1d = dim3(options.num_blocks * 2);
        block_size_2d = dim3(options.block_size_2d, options.block_size_2d);
        block_size_1d = dim3(options.block_size_1d);
    }
    std::string print_result(bool short_form = false);

    // Main execution functions;
    void run(unsigned char* input_image);

   private:

    // Instance-specific settings;
    bool black_and_white = DEFAULT_BLACK_AND_WHITE;  // Convert image to black and white;
    int image_width = DEFAULT_RESIZED_IMAGE_WIDTH;
    
    // General configuration settings;
    int debug = DEBUG;
    bool do_prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    int pascalGpu = 1;
    Policy policy;
    int err = 0;
    dim3 grid_size_2d;
    dim3 grid_size_1d;
    dim3 block_size_2d;
    dim3 block_size_1d;

    // Computation-specific settings;
    int kernel_small_diameter = 3;
    int kernel_large_diameter = 7;
    int kernel_unsharpen_diameter = 3;
    float kernel_small_variance = 0.1;
    float kernel_large_variance = 20;
    float kernel_unsharpen_variance = 5;
    float unsharpen_amount = 30;

    // GPU data;
    int *image, *image3;
    float *image2, *image_unsharpen, *mask_small, *mask_large, *blurred_small, *blurred_large, *blurred_unsharpen;
    float *kernel_small, *kernel_large, *kernel_unsharpen, *maximum_1, *minimum_1, *maximum_2, *minimum_2;
    cudaStream_t s1, s2, s3, s4, s5;

    // Utility functions;
    void alloc();
    void init(unsigned char* input_image, int channel);
    void execute_sync();
    void execute_async();
    void run_inner(unsigned char* input_image, int channel);

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
