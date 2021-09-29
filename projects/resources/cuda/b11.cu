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

#define P 1

extern "C" __global__ void matrix_vector_mult_1(const float* x, const float* y, float* z, int n, int m, int z_offset) {
    extern __shared__ float y_tmp[];
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        y_tmp[i] = y[i];
    }

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < m; j++) {                
            sum += x[i * m + j] * y_tmp[j];
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
    x = (float **) malloc(sizeof(float) * P);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(float) * S * M);
    }
    err = cudaMallocManaged(&y, sizeof(float) * M);
    err = cudaMallocManaged(&z, sizeof(float) * N);

    // Create P streams;
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * P);
    for (int i = 0; i < P; i++) {
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
        // std::cout << y[i] << ", ";
    }
    // std::cout << std::endl;
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < S * M; j++) {
            x[i][j] = x_cpu[i * S * M + j];
            // std::cout << x[i][j] << ", ";
        }
        // std::cout << std::endl;
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
    for (int p = 0; p < P; p++) {
        matrix_vector_mult_1<<<num_blocks, block_size_1d, M * sizeof(float)>>>(x[p], y, z, std::min(S, N - p * S), M, p * S);
        cudaDeviceSynchronize();
    }
}

void Benchmark11::execute_async(int iter) {
    // if (!pascalGpu || stream_attach) {
    //     cudaStreamAttachMemAsync(s1, x, sizeof(float) * x_len);
    //     cudaStreamAttachMemAsync(s1, x1, 0);
    //     cudaStreamAttachMemAsync(s1, x2, 0);
    //     // cudaStreamAttachMemAsync(s1, x3, 0);
    //     cudaStreamAttachMemAsync(s1, kernel_1, 0);
    //     cudaStreamAttachMemAsync(s1, kernel_2, 0);

    //     cudaStreamAttachMemAsync(s2, y, sizeof(float) * x_len);
    //     cudaStreamAttachMemAsync(s2, y1, 0);
    //     // cudaStreamAttachMemAsync(s2, y2, 0);
    //     // cudaStreamAttachMemAsync(s2, y3, 0);
    //     cudaStreamAttachMemAsync(s2, kernel_3, 0);
    //     cudaStreamAttachMemAsync(s2, kernel_4, 0);
    // }
    // if (do_prefetch && pascalGpu) {
    //     cudaMemPrefetchAsync(x, sizeof(float) * x_len, 0, 0);
    //     cudaMemPrefetchAsync(y, sizeof(float) * x_len, 0, 0);
    // }
    // dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    // dim3 grid_size(num_blocks, num_blocks);
    // dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

    // dim3 block_size_3d_dim(block_size_2d / 2, block_size_2d / 2, block_size_2d / 2);
    // dim3 grid_size_3(num_blocks / 2, num_blocks / 2, num_blocks / 2);

    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s1>>>(x1, x, kernel_1, N, N, channels, K, kn1, stride);
    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * channels * sizeof(float), s2>>>(y1, y, kernel_3, N, N, channels, K, kn1, stride);

    // mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s1>>>(x11, x1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);
    // mean_pooling<<<grid_size_3, block_size_3d_dim, 0, s2>>>(y11, y1, N / stride, N / stride, kn1, pooling_diameter, pooling_diameter);

    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x11, kernel_2, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);
    // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y11, kernel_4, N / stride / pooling_diameter, N / stride / pooling_diameter, kn1, K, kn2, stride);

    // // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s1>>>(x2, x1, kernel_2, N / stride, N / stride, kn1, K, kn2, stride);
    // // conv2d<<<grid_size_2, block_size_2d_dim, K * K * kn1 * kn2 * sizeof(float), s2>>>(y2, y1, kernel_4, N / stride, N / stride, kn1, K, kn2, stride);

    // // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float), s1>>>(x3, x2, N / (stride * stride), N / (stride * stride), kn2);
    // // gap<<<grid_size_2, block_size_2d_dim, kn2 * sizeof(float), s2>>>(y3, y2, N / (stride * stride), N / (stride * stride), kn2);

    // cudaEvent_t e1;
    // cudaEventCreate(&e1);
    // cudaEventRecord(e1, s2);
    // cudaStreamWaitEvent(s1, e1, 0);

    // concat<<<num_blocks, block_size_1d, 0, s1>>>(z, x2, y2, x2_len);

    // dot_product<<<num_blocks, block_size_1d, 0, s1>>>(z, dense_weights, res, x2_len);
    // cudaStreamSynchronize(s1);
}

void Benchmark11::execute_cudagraph(int iter) {
    std::cout << "cudagraph (standard) not implemented for b11" << std::endl;
}

void Benchmark11::execute_cudagraph_manual(int iter) {
    std::cout << "cudagraph (manual) not implemented for b11" << std::endl;
}

void Benchmark11::execute_cudagraph_single(int iter) {
    std::cout << "cudagraph (single) not implemented for b11" << std::endl;
}

std::string Benchmark11::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(z[0]);
    } else {
        std::string res = "[";
        for (int i = 0; i < std::min(10, N); i++) {
            res += std::to_string(z[i]) + ", ";
        }
        return res + "...]";
    }
}
