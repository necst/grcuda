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
#define NUM_THREADS_3 16

///////////////////////////////
///////////////////////////////

void autocov_cpu(float *x, int k, int size, float *res) {
    for (int t = 0; t < k; t++) {
        for (int i = 0; i < size - t; i++) {
            res[t] += x[i] * x[i + t];
        }
        res[t] /= size;
    }
}

extern "C" __global__ void autocov_basic(float *x, int k, int size, float *res) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float res_curr = 0;
    if (t < k) {
        for (int i = 0; i < size - t; i++) {
            res_curr += x[i] * x[i + t];
        }
        res[t] = res_curr / size;
    }
}

extern "C" __global__ void autocov(float *x, int k, int size, float *res) {
    __shared__ float cache[NUM_THREADS_3][NUM_THREADS_3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    cache[threadIdx.y][threadIdx.x] = x[i] * x[i + t];
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS_3 / 2;
    while (i > 0) {
        cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(&res[t], cache[threadIdx.y][0] / size);
    }
}

extern "C" __global__ void autocov_checked(float *x, int k, int size, float *res) {
    __shared__ float cache[NUM_THREADS_3][NUM_THREADS_3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size - t) {
        cache[threadIdx.y][threadIdx.x] = x[i] * x[i + t];
    }
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS_3 / 2;
    while (i > 0) {
        if (t < k && threadIdx.x < i) {
            cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0 && t < k) {
        atomicAdd(&res[t], cache[threadIdx.y][0] / size);
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

    // Allocate data on CPU;
    float *x = (float *)malloc(size * sizeof(float));
    // Number of time steps to consider;
    int k = (int)log2((float)size);

    float *res_gold = (float *)calloc(k, sizeof(float));
    float *res = (float *)calloc(k, sizeof(float));

    // Fill with random values;
    create_sample_vector(x, size, true);
    normalize_vector(x, size);

    if (human_readable) {
        cout << "\nX: " << endl;
        print_array_indexed(x, std::min(20, size));
        cout << "----------------------------------------------\n"
             << endl;
    }

    float *x_d;
    float *res_d;
    cudaMalloc(&x_d, sizeof(float) * size);
    cudaMalloc(&res_d, sizeof(float) * k);
    cudaMemcpy(x_d, x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, res, k * sizeof(float), cudaMemcpyHostToDevice);

    // Compute result on CPU;
    auto start = clock_type::now();
    autocov_cpu(x, k, size, res_gold);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (human_readable) {
        cout << "CPU Result:" << endl;
        print_array_indexed(res_gold, std::min(20, k));
        cout << "Duration: " << duration << " ms" << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    //////////////////////////////
    //////////////////////////////

    // Compute on GPU;
    dim3 num_threads(NUM_THREADS_3, NUM_THREADS_3);
    dim3 num_blocks3((size + NUM_THREADS_3 - 1) / NUM_THREADS_3, (k + NUM_THREADS_3 - 1) / NUM_THREADS_3);
    start = clock_type::now();
    autocov_checked<<<num_blocks3, num_threads>>>(x_d, k, size, res_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(res, res_d, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Checked:" << endl;
        print_array_indexed(res, std::min(20, k));
        cout << "Duration: " << duration << " ms" << endl;
    }
    int num_errors = check_array_equality(res_gold, res, k, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Reset values of res;
    for (int i = 0; i < k; i++) {
        res[i] = 0;
    }
    cudaMemcpy(res_d, res, k * sizeof(float), cudaMemcpyHostToDevice);

    //////////////////////////////
    //////////////////////////////

    // Compute on GPU;
    start = clock_type::now();
    autocov<<<num_blocks3, num_threads>>>(x_d, k, size, res_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(res, res_d, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Unchecked:" << endl;
        print_array_indexed(res, std::min(20, k));
        cout << "Duration: " << duration << " ms" << endl;
    }
    num_errors = check_array_equality(res_gold, res, k, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }
}
