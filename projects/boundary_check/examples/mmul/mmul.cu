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

#define NUM_THREADS 16

///////////////////////////////
///////////////////////////////

void mmul_cpu(float *X, float *Y, int X_dim_col, int X_dim_row, int Y_dim_row, float *Z) {
    for (int r = 0; r < X_dim_col; r++) {
        for (int c = 0; c < Y_dim_row; c++) {
            for (int i = 0; i < X_dim_row; i++) {
                Z[Y_dim_row * r + c] += X[X_dim_row * r + i] * Y[Y_dim_row * i + c];
            }
        }
    }
}

// The unchecked version is not logically valid!
// In a 3x3 matrix, one can compute position at index 4 in Z using both [1, 1] and [0, 4] (not valid!),
// but the second case doesn't cause any OoB access despite producing wrong results!
extern "C" __global__ void mmul_bad(float *X, float *Y, int X_dim_col, int X_dim_row, int Y_dim_row, float *Z) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float res = 0;
    for (uint i = 0; i < X_dim_row; i++) {
        res += X[X_dim_row * r + i] * Y[Y_dim_row * i + c];
    }
    Z[Y_dim_row * r + c] = res;
}

extern "C" __global__ void mmul(float *X, float *Y, int X_dim_col, int X_dim_row, int Y_dim_row, float *Z) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float res = 0;
    if (r < X_dim_col && c < Y_dim_row) {
        for (uint i = 0; i < X_dim_row; i++) {
            res += X[X_dim_row * r + i] * Y[Y_dim_row * i + c];
        }
    }
    atomicAdd(&Z[Y_dim_row * r + c], res);
}

extern "C" __global__ void mmul_checked(float *X, float *Y, int X_dim_col, int X_dim_row, int Y_dim_row, float *Z) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < X_dim_col && c < Y_dim_row) {
        float res = 0;
        for (uint i = 0; i < X_dim_row; i++) {
            res += X[X_dim_row * r + i] * Y[Y_dim_row * i + c];
        }
        Z[Y_dim_row * r + c] = res;
    }
}

///////////////////////////////
///////////////////////////////

int main(int argc, char *argv[]) {

    bool human_readable = false;
    int size = 16;

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
    int x_dim_row = int(sqrt(size));
    int x_dim_col = size / x_dim_row; // X is x_dim_col x x_dim_row
    int y_dim_row = x_dim_col;        // Y is x_dim_row x y_dim_row
    float *x = (float *)malloc(x_dim_row * x_dim_col * sizeof(float));
    float *y = (float *)malloc(x_dim_row * y_dim_row * sizeof(float));

    float *z_gold = (float *)calloc(x_dim_col * y_dim_row, sizeof(float));
    float *z = (float *)calloc(x_dim_col * y_dim_row, sizeof(float));

    // Fill with random values;
    create_sample_vector(x, x_dim_row * x_dim_col, true);
    create_sample_vector(y, x_dim_row * y_dim_row, true);

    if (human_readable) {
        cout << "\nX: " << endl;
        print_matrix_indexed(x, x_dim_col, x_dim_row);
        cout << "Y: " << endl;
        print_matrix_indexed(y, x_dim_row, y_dim_row);
        cout << "----------------------------------------------\n"
             << endl;
    }

    dim3 threads_per_block(NUM_THREADS, NUM_THREADS);
    uint num_blocks_rows = (x_dim_col + NUM_THREADS - 1) / NUM_THREADS;
    uint num_blocks_cols = (y_dim_row + NUM_THREADS - 1) / NUM_THREADS;
    dim3 num_blocks(num_blocks_rows, num_blocks_cols);

    float *x_d;
    float *y_d;
    float *z_d;
    cudaMalloc(&x_d, sizeof(float) * x_dim_row * x_dim_col);
    cudaMalloc(&y_d, sizeof(float) * x_dim_row * y_dim_row);
    cudaMalloc(&z_d, sizeof(float) * x_dim_col * y_dim_row);
    cudaMemcpy(x_d, x, x_dim_row * x_dim_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, x_dim_row * y_dim_row * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(z_d, z, x_dim_col * y_dim_row * sizeof(float), cudaMemcpyHostToDevice);

    // Compute result on CPU;
    auto start = clock_type::now();
    mmul_cpu(x, y, x_dim_col, x_dim_row, y_dim_row, z_gold);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (human_readable) {
        cout << "CPU Result:" << endl;
        print_matrix_indexed(z_gold, x_dim_col, y_dim_row);
        cout << "Duration: " << duration << " ms" << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    start = clock_type::now();
    mmul<<<num_blocks, threads_per_block>>>(x_d, y_d, x_dim_col, x_dim_row, y_dim_row, z_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(z, z_d, x_dim_col * y_dim_row * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Unchecked:" << endl;
        print_matrix_indexed(z, x_dim_col, y_dim_row);
        cout << "Duration: " << duration << " ms" << endl;
    }
    int num_errors = check_array_equality(z_gold, z, x_dim_col * y_dim_row, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Reset values of z;
    for (int i = 0; i < x_dim_col * y_dim_row; i++) {
        z[i] = 0;
    }
    cudaMemcpy(z_d, z, x_dim_col * y_dim_row * sizeof(float), cudaMemcpyHostToDevice);

    // Compute on GPU - Checked;
    start = clock_type::now();
    mmul_checked<<<num_blocks, threads_per_block>>>(x_d, y_d, x_dim_col, x_dim_row, y_dim_row, z_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(z, z_d, x_dim_col * y_dim_row * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Checked:" << endl;
        print_matrix_indexed(z, x_dim_col, y_dim_row);
        cout << "Duration: " << duration << " ms" << endl;
    }
    num_errors = check_array_equality(z_gold, z, x_dim_col * y_dim_row, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }
}
