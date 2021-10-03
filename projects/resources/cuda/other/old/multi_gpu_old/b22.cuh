#pragma once
#include "benchmark.cuh"
// #define DIM 100

class Benchmark22 : public Benchmark {
   public:
    Benchmark22(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    void execute_cudagraph_single(int iter);
    void prefetch(cudaStream_t &s1, cudaStream_t &s2);
    std::string print_result(bool short_form = false);

   private:
    //float A[DIM][DIM], U[DIM][DIM], L[DIM][DIM], LT[DIM][DIM];
    float *A, *U, *L;
    cudaStream_t *s;
    
    //// for LU_v1.cu
    // cudaGraph_t graph;
    // cudaGraphExec_t graphExec;
    // std::vector<cudaGraphNode_t> nodeDependencies;
    // cudaGraphNode_t kernel_1, kernel_2, kernel_3;
    // cudaKernelNodeParams kernel_1_params;
    // cudaKernelNodeParams kernel_2_params;
    // cudaKernelNodeParams kernel_3_params;
};
