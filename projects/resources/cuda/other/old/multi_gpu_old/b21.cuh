#pragma once
#include "benchmark.cuh"

class Benchmark21 : public Benchmark {
   public:
    Benchmark21(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    void execute_cudagraph_single(int iter);
    std::string print_result(bool short_form = false);

   private:
    float *a_d, *b_d, *x_d, *x_result_d;
    float **result_d;
    cudaStream_t *s;

};
