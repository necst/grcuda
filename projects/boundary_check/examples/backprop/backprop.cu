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

#define THREADS 256
#define WIDTH 16  // shared memory width
#define HEIGHT 16 // shared memory height

#define ETA 0.3      //eta value
#define MOMENTUM 0.3 //momentum value
#define NUM_THREAD 4 //OpenMP threads

///////////////////////////////
///////////////////////////////

extern "C" __global__ void
backprop(float *input_cuda,
         float *output_hidden_cuda,
         float *input_hidden_cuda,
         float *hidden_partial_sum,
         int in,
         int hid) {

    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);

    int index_in = HEIGHT * by + ty + 1;

    __shared__ float input_node[HEIGHT];
    __shared__ float weight_matrix[HEIGHT][WIDTH];

    // Unsafe access;
    if (tx == 0) {
        input_node[ty] = input_cuda[index_in];
    }

    __syncthreads();

    // Unsafe access;
    weight_matrix[ty][tx] = input_hidden_cuda[index];

    __syncthreads();

    weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

    __syncthreads();

    for (int i = 1; i <= __log2f(HEIGHT); i++) {

        int power_two = __powf(2, i);

        if (ty % power_two == 0) {
            weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];
        }

        __syncthreads();
    }

    input_hidden_cuda[index] = weight_matrix[ty][tx];

    __syncthreads();

    // This can cause OoB accesses, hid and hidden_partial_sum.size are not related at all!
    if (tx == 0) {
        //if (WIDTH * in / 16 <= by * hid + ty) {
        //    printf("index: %d\n", by * hid + ty);
        //}
        hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
    }
}

///////////////////////////////
///////////////////////////////

extern "C" __global__ void backprop_2(float *delta,
                                         int hid,
                                         float *ly,
                                         int in,
                                         float *w,
                                         float *oldw) {

    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
    int index_y = HEIGHT * by + ty + 1;
    int index_x = tx + 1;

    w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
    oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

    __syncthreads();

    if (ty == 0 && by == 0) {
        w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
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
    int in = size;
    int hid = size;

    int num_blocks = in / 16; // Size must be a multiple of 16, otherwise the number of blocks doesn't add up;
    dim3 grid(1, num_blocks);
    dim3 threads(16, 16);

    float *input_units = (float *)malloc((in + 1) * sizeof(float));
    float *input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
    float *partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

    // Fill with random values;
    create_sample_vector(input_units, (in + 1), true);
    create_sample_vector(input_weights_one_dim, (in + 1) * (hid + 1), true);

    if (human_readable) {
        cout << "\nX: " << endl;
        print_array_indexed(input_units, std::min(20, (in + 1)));
        cout << "Y: " << endl;
        print_array_indexed(input_weights_one_dim, std::min(20, (in + 1) * (hid + 1)));
        cout << "----------------------------------------------\n"
             << endl;
    }

    float *input_hidden_cuda;
    float *input_cuda;
    float *output_hidden_cuda;
    float *hidden_partial_sum;

    cudaMalloc((void **)&input_cuda, (in + 1) * sizeof(float));
    cudaMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float));
    cudaMalloc((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
    cudaMalloc((void **)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
    cudaMemcpy(input_cuda, input_units, (in + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Compute on GPU;
    auto start = clock_type::now();
    backprop<<<grid, threads>>>(input_cuda,
                                output_hidden_cuda,
                                input_hidden_cuda,
                                hidden_partial_sum,
                                in,
                                hid);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "Forward - Unchecked:" << endl;
        print_array_indexed(partial_sum, std::min(20, num_blocks * WIDTH));
        cout << "Duration: " << duration << " ms" << endl;
    }

    // Second part, backpropagation;
    float *hidden_delta;
    float *hidden_delta_cuda;
    float *input_prev_weights_cuda;
    float *input_weights_prev_one_dim;

    input_weights_prev_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
    hidden_delta = (float *)malloc((hid + 1) * sizeof(float));

    // Fill with random values;
    create_sample_vector(hidden_delta, (hid + 1), true);
    create_sample_vector(input_weights_prev_one_dim, (in + 1) * (hid + 1), true);

    cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
    cudaMalloc((void **)&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

    cudaMemcpy(hidden_delta_cuda, hidden_delta, (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cout << "delta size: " << (hid + 1) << "; ly size: " << (in + 1) << "; w size:" << (in + 1) * (hid + 1) << endl;

    start = clock_type::now();
    backprop_2<<<grid, threads>>>(hidden_delta_cuda,
                                                hid,
                                                input_cuda,
                                                in,
                                                input_hidden_cuda,
                                                input_prev_weights_cuda);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cudaMemcpy(input_units, input_cuda, (in + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckError();
    if (human_readable) {
        cout << "Backward - Unchecked:" << endl;
        print_array_indexed(input_weights_one_dim, std::min(20, (in + 1) * (hid + 1)));
        cout << "Duration: " << duration << " ms" << endl;
    }
}
