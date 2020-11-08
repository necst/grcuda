#include <chrono>
#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "../common/utils.hpp"

////////////////////////////////
////////////////////////////////

using std::cout;
using std::endl;

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define NUM_THREADS 128
#define DANGLING 0.85

///////////////////////////////
///////////////////////////////

void pr_cpu(int *ptr, int *idx, float *pr, float *pr_old, int *outdegrees, int N, int E) {
    for (int v = 0; v < N; v++) {
        float sum = 0;
        for (int i = ptr[v]; i < ptr[v + 1]; i++) {
            sum += pr_old[idx[i]] / outdegrees[v];
        }
        pr[v] = (1 - DANGLING) / N + DANGLING * sum;
    }
}

extern "C" __global__ void pr_checked(int *ptr, int *idx, float *pr, float *pr_old, int *outdegrees, int N, int E) {

    int v = blockIdx.x * NUM_THREADS + threadIdx.x;
    if (v < N) {
        float sum = 0;
        for (int i = ptr[v]; i < ptr[v + 1]; i++) {
            sum += pr_old[idx[i]] / outdegrees[v];
        }
        pr[v] = (1 - DANGLING) / N + DANGLING * sum;
    }
}

extern "C" __global__ void pr(int *ptr, int *idx, float *pr, float *pr_old, int *outdegrees, int N, int E) {

    int v = blockIdx.x * NUM_THREADS + threadIdx.x;
    float sum = 0;
    for (int i = ptr[v]; i < ptr[v + 1]; i++) {
        sum += pr_old[idx[i]] / outdegrees[v];
    }
    pr[v] = (1 - DANGLING) / N + DANGLING * sum;
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

    int num_blocks = (size + 1 + NUM_THREADS - 1) / NUM_THREADS;

    // Allocate data on CPU;
    std::vector<int> ptr(size + 1, 0);
    std::vector<int> idx;

    // Fill with random values, interpreted as CSC;
    create_random_graph(ptr, idx);

    int N = size;
    int E = idx.size();

    std::vector<float> pr_old(N, 1.0 / N);
    std::vector<float> pr_gold(N, 1.0 / N);
    std::vector<float> pr_h(N, 1.0 / N);
    std::vector<int> outdegrees(N, 0);

    // Compute the outdegrees;
    for (int v : idx) {
        outdegrees[v]++;
    }

    if (human_readable) {
        print_graph(ptr, idx);
        cout << "Outdegrees:" << endl;
        print_array_indexed(outdegrees.data(), std::min(N, 20));
        cout << "----------------------------------------------\n"
             << endl;
    }

    int *ptr_d;
    int *idx_d;
    float *pr_d;
    float *pr_old_d;
    int *outdegrees_d;
    cudaMalloc(&ptr_d, sizeof(int) * (N + 1));
    cudaMalloc(&idx_d, sizeof(int) * E);
    cudaMalloc(&pr_d, sizeof(float) * N);
    cudaMalloc(&pr_old_d, sizeof(float) * N);
    cudaMalloc(&outdegrees_d, sizeof(int) * N);
    cudaMemcpy(ptr_d, ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(idx_d, idx.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pr_d, pr_old.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pr_old_d, pr_old.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outdegrees_d, outdegrees.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Compute result on CPU;
    auto start = clock_type::now();
    pr_cpu(ptr.data(), idx.data(), pr_gold.data(), pr_old.data(), outdegrees.data(), N, E);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (human_readable) {
        cout << "CPU Result:" << endl;
        print_array_indexed(pr_gold.data(), std::min(20, N));
        cout << "Duration: " << duration << " ms" << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    start = clock_type::now();
    pr_checked<<<num_blocks, NUM_THREADS>>>(ptr_d, idx_d, pr_d, pr_old_d, outdegrees_d, N, E);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(pr_h.data(), pr_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Checked:" << endl;
        print_array_indexed(pr_h.data(), std::min(20, N));
        cout << "Duration: " << duration << " ms" << endl;
    }
    int num_errors = check_array_equality(pr_gold.data(), pr_h.data(), N, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    std::fill(pr_h.begin(), pr_h.end(), 1.0 / N);
    cudaMemcpy(pr_d, pr_h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    start = clock_type::now();
    pr<<<num_blocks, NUM_THREADS>>>(ptr_d, idx_d, pr_d, pr_old_d, outdegrees_d, N, E);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(pr_h.data(), pr_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Unchecked:" << endl;
        print_array_indexed(pr_h.data(), std::min(20, N));
        cout << "Duration: " << duration << " ms" << endl;
    }
    num_errors = check_array_equality(pr_gold.data(), pr_h.data(), N, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }
}
