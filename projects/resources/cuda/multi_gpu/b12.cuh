// Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <stdexcept>
#include <vector>
#include "../benchmark.cuh"
#include "../mmio.hpp"

using f32 = float;
using u32 = unsigned;

struct coo_matrix_t {
    int *x;
    int *y;
    float *val;
    int begin;
    int end;
    int N;
    int nnz;

};

class Benchmark12 : public Benchmark {
public:
    Benchmark12(Options &options) : Benchmark(options) {
        int *x, *y;
        f32 *val;
        int N, M, nnz;

        mm_read_unsymmetric_sparse(this->matrix_path.c_str(), &M, &N, &nnz, &val, &x, &y);

        this->matrix.begin = 0;
        this->matrix.end = nnz;
        this->matrix.N = N;
        this->matrix.x = x;
        this->matrix.y = y;
        this->matrix.val = val;
        this->matrix.nnz = nnz;
        this->num_gpus = options.max_devices;


    }
    void alloc();
    void init();
    void reset();
    void execute_sync(int);
    void execute_async(int);
    void execute_cudagraph(int);
    void execute_cudagraph_manual(int);
    void execute_cudagraph_single(int);
    void prefetch(std::vector<cudaStream_t>);
    std::string print_result(bool);

private:

    unsigned num_eigencomponents = 8;
    int num_gpus = -1;
    std::string matrix_path = "";

    coo_matrix_t matrix;
    std::vector<coo_matrix_t> coo_partitions;
    std::vector<float*> vec_in, spmv_vec_out, intermediate_dot_product_values, alpha_intermediate, beta_intermediate, vec_next, lanczos_vectors, normalized_out;


    void alloc_coo_partitions();
    void alloc_vectors();
    coo_matrix_t assign_partition(unsigned, unsigned, unsigned);

};

