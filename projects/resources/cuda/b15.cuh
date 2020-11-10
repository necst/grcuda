#pragma once
#include "benchmark.cuh"

class Benchmark15 : public Benchmark
{
public:
    Benchmark15(Options &options) : Benchmark(options) {}
    void alloc();
    void init();
    void reset();
    void execute_sync(int iter);
    void execute_async(int iter);
    void execute_cudagraph(int iter);
    void execute_cudagraph_manual(int iter);
    std::string print_result(bool short_form = false);

private:
    int M = 10;
    double **x, **y, *tmp_x;
    cudaStream_t *s;

};