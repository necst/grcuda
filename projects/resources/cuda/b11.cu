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

#include "b11.cuh"

//////////////////////////////
//////////////////////////////

#define P 16
#define ITER 1

extern "C" __global__ void matrix_vector_mult_1(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = sum;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark11::alloc() {
    M = N;
    S = (N + P - 1) / P;
    x_cpu = (float *) malloc(sizeof(float) * N * M);
    x = (float **) malloc(sizeof(float*) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(float) * S * M);
    }
    err = cudaMallocManaged(&y, sizeof(float) * M);
    err = cudaMallocManaged(&z, sizeof(float) * N);

    // Create P streams;
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    for (int i = 0; i < P; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s[i]);
    }
}

void Benchmark11::init() {
    for (int i = 0; i < N * M; i++) {
        x_cpu[i] = (float)(rand()) / (float)(RAND_MAX);
    }
}

void Benchmark11::reset() {
    for (int i = 0; i < M; i++) {
        y[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < S * M; j++) {
            x[i][j] = x_cpu[i * S * M + j];
        }
    }
}

void Benchmark11::execute_sync(int iter) {
    if (do_prefetch && pascalGpu) {
        for (int p = 0; p < P; p++) {
            cudaMemPrefetchAsync(x[p], sizeof(float) * S * M, 0, 0);
            cudaDeviceSynchronize();
        }
        cudaMemPrefetchAsync(y, sizeof(float) * M, 0, 0);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < ITER; i++) {
        for (int p = 0; p < P; p++) {
            matrix_vector_mult_1<<<num_blocks, block_size_1d>>>(x[p], i % 2 ? z : y, i % 2 ? y : z, std::min(S, N - p * S), M, p * S);
            cudaDeviceSynchronize();
        } 
    }
}

void Benchmark11::execute_async(int iter) {
    for (int p = 0; p < P; p++) {
        cudaSetDevice(select_gpu(p, max_devices));
        if (!pascalGpu || stream_attach) {
            cudaStreamAttachMemAsync(s[p], x[p], sizeof(float) * S * M);
        }
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[p], sizeof(float) * S * M, select_gpu(p, max_devices), s[p]);
            cudaMemPrefetchAsync(y, sizeof(float) * M, select_gpu(p, max_devices), s[p]);
        }
    }
    for (int i = 0; i < ITER; i++) {
        for (int p = 0; p < P; p++) {
            matrix_vector_mult_1<<<num_blocks, block_size_1d, 0, s[p]>>>(x[p], i % 2 ? z : y, i % 2 ? y : z, std::min(S, N - p * S), M, p * S);
        }
        for (int p = 0; p < P; p++) {
            err = cudaStreamSynchronize(s[p]);
        }
    }
}

std::string Benchmark11::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(z[0]);
    } else {
        std::string res = "[";
        for (int i = 0; i < std::min(100, N); i++) {
            res += std::to_string(z[i]) + ", ";
        }
        return res + "...]";
    }
}
