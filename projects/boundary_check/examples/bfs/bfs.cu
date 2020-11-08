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

///////////////////////////////
///////////////////////////////

void bfs_cpu(int *ptr, int *idx, int *res_gold, int iteration, int N, int E, bool *graph_mask, bool *graph_visited, bool *updating_graph_mask) {

    for (int v = 0; v < N; v++) {
        if (graph_mask[v]) {
            graph_mask[v] = false;
            for (int i = ptr[v]; i < ptr[v + 1]; i++) {
                int id = idx[i];
                if (!graph_visited[id]) {
                    res_gold[id] = iteration;
                    updating_graph_mask[id] = true;
                }
            }
        }
    }
}

extern "C" __global__ void bfs_checked(int *ptr, int *idx, int *res_gold, int iteration, int N, int E, bool *graph_mask, bool *graph_visited, bool *updating_graph_mask) {

    int v = blockIdx.x * NUM_THREADS + threadIdx.x;
    if (v < N && graph_mask[v]) {
        graph_mask[v] = false;
        for (int i = ptr[v]; i < ptr[v + 1]; i++) {
            int id = idx[i];
            if (!graph_visited[id]) {
                res_gold[id] = iteration;
                updating_graph_mask[id] = true;
            }
        }
    }
}

extern "C" __global__ void bfs(int *ptr, int *idx, int *res_gold, int iteration, int N, int E, bool *graph_mask, bool *graph_visited, bool *updating_graph_mask) {

    int v = blockIdx.x * NUM_THREADS + threadIdx.x;
    if (graph_mask[v]) {
        graph_mask[v] = false;
        for (int i = ptr[v]; i < ptr[v + 1]; i++) {
            int id = idx[i];
            if (!graph_visited[id]) {
                res_gold[id] = iteration;
                updating_graph_mask[id] = true;
            }
        }
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

    int num_blocks = (size + 1 + NUM_THREADS - 1) / NUM_THREADS;

    // Allocate data on CPU;
    std::vector<int> ptr(size + 1, 0);
    std::vector<int> idx;

    std::vector<int> res_gold(size, 0);
    std::vector<int> res(size, 0);
    bool *graph_mask = (bool *)malloc(size * sizeof(bool));
    bool *graph_visited = (bool *)malloc(size * sizeof(bool));
    bool *graph_mask_backup = (bool *)malloc(size * sizeof(bool));
    bool *updating_graph_mask = (bool *)malloc(size * sizeof(bool));

    // Fill with random values;
    create_random_graph(ptr, idx);

    // Start exploration from a random set of vertices;
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        graph_mask[i] = rand() % 2;
        graph_visited[i] = graph_mask[i];
        graph_mask_backup[i] = graph_mask[i];
        updating_graph_mask[i] = 0;
    }

    int N = size;
    int E = idx.size();

    if (human_readable) {
        print_graph(ptr, idx);
        cout << "\nInitial mask\n";
        print_array_indexed(graph_mask, std::min(20, N));
        cout << "----------------------------------------------\n"
             << endl;
    }

    int *ptr_d;
    int *idx_d;
    int *res_d;
    bool *graph_mask_d;
    bool *graph_visited_d;
    bool *updating_graph_mask_d;
    cudaMalloc(&ptr_d, sizeof(int) * (N + 1));
    cudaMalloc(&idx_d, sizeof(int) * E);
    cudaMalloc(&res_d, sizeof(int) * N);
    cudaMalloc(&graph_mask_d, sizeof(int) * N);
    cudaMalloc(&graph_visited_d, sizeof(int) * NL_TEXTMAX);
    cudaMalloc(&updating_graph_mask_d, sizeof(int) * N);
    cudaMemcpy(ptr_d, ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(idx_d, idx.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, res.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_mask_d, graph_mask, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_visited_d, graph_visited, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(updating_graph_mask_d, updating_graph_mask, N * sizeof(int), cudaMemcpyHostToDevice);

    // Compute result on CPU;
    auto start = clock_type::now();
    bfs_cpu(ptr.data(), idx.data(), res_gold.data(), 1, N, E, graph_mask, graph_visited, updating_graph_mask);
    auto end = clock_type::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (human_readable) {
        cout << "CPU Result:" << endl;
        print_array_indexed(res_gold.data(), std::min(20, N));
        cout << "Duration: " << duration << " ms" << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    start = clock_type::now();
    bfs_checked<<<num_blocks, NUM_THREADS>>>(ptr_d, idx_d, res_d, 1, N, E, graph_mask_d, graph_visited_d, updating_graph_mask_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(res.data(), res_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Checked:" << endl;
        print_array_indexed(res.data(), std::min(20, N));
        cout << "Duration: " << duration << " ms" << endl;
    }
    int num_errors = check_array_equality(res_gold.data(), res.data(), N, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }

    // Compute on GPU;
    std::fill(res.begin(), res.end(), 0);
    cudaMemcpy(res_d, res.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_mask_d, graph_mask_backup, N * sizeof(int), cudaMemcpyHostToDevice);

    start = clock_type::now();
    bfs<<<num_blocks, NUM_THREADS>>>(ptr_d, idx_d, res_d, 1, N, E, graph_mask_d, graph_visited_d, updating_graph_mask_d);
    end = clock_type::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cudaMemcpy(res.data(), res_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    if (human_readable) {
        cout << "GPU Result - Unchecked:" << endl;
        print_array_indexed(res.data(), std::min(20, N));
        cout << "Duration: " << duration << " ms" << endl;
    }
    num_errors = check_array_equality(res_gold.data(), res.data(), N, 0.00000001, human_readable);
    if (human_readable) {
        cout << "Number of errors: " << num_errors << endl;
        cout << "----------------------------------------------\n"
             << endl;
    }
}
