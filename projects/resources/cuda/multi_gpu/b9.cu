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

#include "b9.cuh"

//////////////////////////////
//////////////////////////////

#define P 16
#define ITER 10

// z = x @ y;
extern "C" __global__ void matrix_vector_mult(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = sum;
    }
}

// z := w + alpha * A @ y;
extern "C" __global__ void matrix_vector_mult_axpy(const float* x, const float* y, const float *w, const float alpha, float* z, int n, int m, int z_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y[j];
        }
        z[z_offset + i] = alpha * sum + w[z_offset + i];
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// z = <x, x>;
extern "C" __global__ void l2_norm(const float *x, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float x_tmp = x[i];
        sum += x_tmp * x_tmp;
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

// z = <x, y>;
extern "C" __global__ void dot(const float *x, const float *y, float* z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum); // The first thread in the warp updates the output;
}

// y = val + alpha * x;
extern "C" __global__ void saxpy(float* y, float *val, float *x, float alpha, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = val[i] + alpha * x[i];
    }
}

// Simply copy array x into y;
extern "C" __global__ void cpy(float *y, const float *x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i];
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark9M::alloc() {
    S = (N + P - 1) / P;
    err = cudaMallocManaged(&A, sizeof(float) * N * N);
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&b, sizeof(float) * N);
    err = cudaMallocManaged(&p, sizeof(float) * N);
    err = cudaMallocManaged(&r, sizeof(float) * N);
    err = cudaMallocManaged(&y, sizeof(float) * N);
    err = cudaMallocManaged(&t1, sizeof(float));
    err = cudaMallocManaged(&t2, sizeof(float));

    // Create streams;
    cudaStream_t s1, s2;
    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s2);

    // x_cpu = (float *) malloc(sizeof(float) * N * M);
    // x = (float **) malloc(sizeof(float*) * P);
    // for (int i = 0; i < P; i++) {
    //     err = cudaMallocManaged(&x[i], sizeof(float) * S * M);
    // }
    // err = cudaMallocManaged(&y, sizeof(float) * M);
    // err = cudaMallocManaged(&z, sizeof(float) * N);

    // // Create P streams;
    // s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    // for (int i = 0; i < P; i++) {
    //     cudaSetDevice(select_gpu(i, max_devices));
    //     err = cudaStreamCreate(&s[i]);
    // }
}

void Benchmark9M::init() {
    // for (int i = 0; i < N * N; i++) {
    //     x_cpu[i] = (float)(rand()) / (float)(RAND_MAX);
    // }

    // Random input matrix;
    float max = float(RAND_MAX);
    for (int i = 0; i < N * N; i++) {
        A[i] = float(rand()) / max;
    }
    // Random input b;
    for (int i = 0; i < N; i++) {
        b[i] = float(rand()) / max;
    }
}

void Benchmark9M::reset() {
    // Default init of solution x;
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
    }
    // Reset norms;
    *t1 = 0.0;
    *t2 = 0.0;

    // for (int i = 0; i < P; i++) {
    //     for (int j = 0; j < S * M; j++) {
    //         x[i][j] = x_cpu[i * S * M + j];
    //     }
    // }
}

void Benchmark9M::execute_sync(int iter) { 

    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(A, sizeof(float) * N * N, 0);
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0);
        cudaMemPrefetchAsync(b, sizeof(float) * N, 0);
        cudaMemPrefetchAsync(r, sizeof(float) * N, 0);
        cudaMemPrefetchAsync(p, sizeof(float) * N, 0);
    }

    matrix_vector_mult_axpy<<<num_blocks, block_size_1d>>>(A, x, b, -1, r, N, N, 0);
    cudaDeviceSynchronize();
    cpy<<<num_blocks, block_size_1d>>>(p, r, N);
    cudaDeviceSynchronize();
    l2_norm<<<num_blocks, block_size_1d>>>(r, t1, N);
    cudaDeviceSynchronize();
    for (int i = 0; i < ITER; i++) {
        matrix_vector_mult<<<num_blocks, block_size_1d>>>(A, p, y, N, N, 0);
        cudaDeviceSynchronize();
        dot<<<num_blocks, block_size_1d>>>(p, y, t2, N);
        cudaDeviceSynchronize();
        float alpha = *t1 / *t2;
        float old_t1 = *t1;
        *t1 = 0.0;
        saxpy<<<num_blocks, block_size_1d>>>(x, x, p, alpha, N);
        cudaDeviceSynchronize();
        saxpy<<<num_blocks, block_size_1d>>>(r, r, y, -1.0 * alpha, N);
        cudaDeviceSynchronize();
        l2_norm<<<num_blocks, block_size_1d>>>(r, t1, N);
        cudaDeviceSynchronize();
        float beta = *t1 / old_t1;
        saxpy<<<num_blocks, block_size_1d>>>(p, r, p, beta, N);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}

void Benchmark9M::execute_async(int iter) {
    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(A, sizeof(float) * N * N, 0, s1);
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(b, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(r, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(p, sizeof(float) * N, 0, s1);
    }

    matrix_vector_mult_axpy<<<num_blocks, block_size_1d, 0, s1>>>(A, x, b, -1, r, N, N, 0);
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1, s1);
    cpy<<<num_blocks, block_size_1d, 0, s1>>>(p, r, N);
    cudaStreamWaitEvent(s2, e1, 0);
    l2_norm<<<num_blocks, block_size_1d, 0, s2>>>(r, t1, N);
    for (int i = 0; i < ITER; i++) {
        matrix_vector_mult<<<num_blocks, block_size_1d, 0, s1>>>(A, p, y, N, N, 0);
        dot<<<num_blocks, block_size_1d, 0, s1>>>(p, y, t2, N);
        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);
        float alpha = *t1 / *t2;
        float old_t1 = *t1;
        *t1 = 0.0;
        saxpy<<<num_blocks, block_size_1d, 0, s1>>>(x, x, p, alpha, N);
        saxpy<<<num_blocks, block_size_1d, 0, s2>>>(r, r, y, -1.0 * alpha, N);
        l2_norm<<<num_blocks, block_size_1d, 0, s2>>>(r, t1, N);
        cudaStreamSynchronize(s2);
        float beta = *t1 / old_t1;
        saxpy<<<num_blocks, block_size_1d, 0, s1>>>(p, r, p, beta, N);
    }
    cudaStreamSynchronize(s1);
}

std::string Benchmark9M::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(x[0]);
    } else {
        std::string res = "[";
        for (int i = 0; i < std::min(100, N); i++) {
            res += std::to_string(x[i]) + ", ";
        }
        return res + "...]";
    }
}