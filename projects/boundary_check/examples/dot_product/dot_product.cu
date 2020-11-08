#include <chrono>
#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "../common/utils.hpp"

////////////////////////////////
////////////////////////////////

using std::cout;
using std::endl;

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define NUM_THREADS 128

///////////////////////////////
///////////////////////////////

void dot_product_cpu(float *x, float *y, int size, float *res) {
    for (int i = 0; i < size; i++) {
        *res += x[i] * y[i];
    }
}

extern "C" __global__ void dot_product(float *x, float *y, int size, float *res) {
    __shared__ float cache[NUM_THREADS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // This OoB access doesn't affect the result;
    cache[threadIdx.x] = x[i] * y[i];
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS / 2;
    while (i > 0) {
        // CUDA fails if an OoB access is done on shared memory.
        // Automatically fixing this pattern is not trivial:
        //   the starting value of "i" and the size of "cache" are known at compile time, 
        //   so we could add a check like "threadIdx.x + i < NUM_THREADS".
        //     This fixes the issue, but it's not logically equivalent to the "correct" solution below,
        //   and results in wasted computation;
        cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(res, cache[0]);
    }
}

extern "C" __global__ void dot_product_checked(float *x, float *y, int size, float *res) {
    __shared__ float cache[NUM_THREADS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        cache[threadIdx.x] = x[i] * y[i];
    }
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS / 2;
    while (i > 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(res, cache[0]);
    }
}

///////////////////////////////
///////////////////////////////

int main(int argc, char *argv[]) {

    bool human_readable = false;
    int size = 10;

    int opt;
    static struct option long_options[] =
        {
            {"human_readable", no_argument, 0, 'h'},
            {"size", required_argument, 0, 's'},
            {0, 0, 0, 0}};
    // getopt_long stores the option index here;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "hs:c", long_options, &option_index)) != EOF) {
        switch (opt) {
            case 'h':
                human_readable = true;
                break;
            case 's':
                size = atoi(optarg);
                break;
            default:
                return 0;
        }
    }

    int num_blocks = (size + NUM_THREADS - 1) / NUM_THREADS;

    // Allocate data on CPU;
    float *x = (float *)malloc(size * sizeof(float));
    float *y = (float *)malloc(size * sizeof(float));

    float res_gold = 0;
    float res = 0;

    // Fill with random values;
    create_sample_vector(x, size, true);
    create_sample_vector(y, size, true);

    if (human_readable) {
        cout << "\nX: " << endl;
        print_array_indexed(x, std::min(20, size));
        cout << "Y: " << endl;
        print_array_indexed(y, std::min(20, size));
        cout << "----------------------------------------------\n"
             << endl;
    }

    float *x_d;
    float *y_d;
    float *res_d;
    cudaMalloc(&x_d, sizeof(float) * size);
    cudaMalloc(&y_d, sizeof(float) * size);
    cudaMalloc(&res_d, sizeof(float));
    cudaMemcpy(x_d, x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size * sizeof(float), cudaMemcpyHostToDevice);

    // Compute result on CPU;
    auto start = clock_type::now();
    dot_product_cpu(x, y, size, &res_gold);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (human_readable) {
        cout << "CPU Result: " << res_gold << endl;
        cout << "Duration: " << duration << " ms" << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU - Checked;
    res = 0;
    cudaMemcpy(res_d, &res, sizeof(float), cudaMemcpyHostToDevice);

    start = clock_type::now();
    dot_product_checked<<<num_blocks, NUM_THREADS>>>(x_d, y_d, size, res_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(&res, res_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Checked: " << res << endl;
        cout << "Duration: " << duration << " ms" << endl;
    }
    bool equal = check_equality(res_gold, res, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Correct: " << equal << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    res = 0;
    cudaMemcpy(res_d, &res, sizeof(float), cudaMemcpyHostToDevice);

    start = clock_type::now();
    dot_product<<<num_blocks, NUM_THREADS>>>(x_d, y_d, size, res_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(&res, res_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Unchecked: " << res << endl;
        cout << "Duration: " << duration << " ms" << endl;
    }
    equal = check_equality(res_gold, res, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Correct: " << equal << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }
}
