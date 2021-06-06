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
    void execute_cudagraph_single(int iter);
    std::string print_result(bool short_form = false);

private:
    int M = 20;
    int NGPU = 4;
    double **x, **y, *tmp_x;
    double **yd;
    cudaStream_t *s;

};