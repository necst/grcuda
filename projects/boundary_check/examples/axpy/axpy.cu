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

void axpy_cpu(float *x, float *y, float a, int size, float *res) {
    for (int i = 0; i < size; i++) {
        res[i] = a * x[i] + y[i];
    }
}

extern "C" __global__ void axpy(float *x, float *y, float a, int size, float *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a * x[i] + y[i];
}

extern "C" __global__ void axpy_checked(float *x, float *y, float a, int size, float *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        res[i] = a * x[i] + y[i];
    }
}

extern "C" __global__ void axpy_with_args(float *x, float *y, float a, float *res, int x_size, int y_size, int res_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a * x[i] + y[i];
}

extern "C" __global__ void axpy_with_args_checked(float *x, float *y, float a, float *res, int x_size, int y_size, int res_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < x_size && i < y_size && i < res_size) {
        res[i] = a * x[i] + y[i];
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
    float a = 0.f;
    float *x = (float *)malloc(size * sizeof(float));
    float *y = (float *)malloc(size * sizeof(float));

    float *res_gold = (float *)calloc(size, sizeof(float));
    float *res = (float *)calloc(size, sizeof(float));

    // Fill with random values;
    create_sample_vector(&a, 1, true, false);
    create_sample_vector(x, size, true);
    create_sample_vector(y, size, true);

    if (human_readable) {
        cout << "a: " << a << endl;
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
    cudaMalloc(&res_d, sizeof(float) * size);
    cudaMemcpy(x_d, x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, res, size * sizeof(float), cudaMemcpyHostToDevice);

    // Compute result on CPU;
    auto start = clock_type::now();
    axpy_cpu(x, y, a, size, res_gold);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (human_readable) {
        cout << "CPU Result:" << endl;
        print_array_indexed(res_gold, std::min(20, size));
        cout << "Duration: " << duration << " ms" << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    start = clock_type::now();
    axpy_checked<<<num_blocks, NUM_THREADS>>>(x_d, y_d, a, size, res_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(res, res_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Unchecked:" << endl;
        print_array_indexed(res, std::min(20, size));
        cout << "Duration: " << duration << " ms" << endl;
    }
    int num_errors = check_array_equality(res_gold, res, size, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU - Checked;
    start = clock_type::now();
    axpy<<<num_blocks, NUM_THREADS>>>(x_d, y_d, a, size, res_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(res, res_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Checked:" << endl;
        print_array_indexed(res, std::min(20, size));
        cout << "Duration: " << duration << " ms" << endl;
    }
    num_errors = check_array_equality(res_gold, res, size, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }
}
