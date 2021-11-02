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

#include "b13.cuh"

//////////////////////////////
//////////////////////////////

// Assume that z is partitioned in blocks (old implementation, not used);
extern "C" __global__ void matrix_matrix_mult_1(const float* x, const float* y, float* z, int x_num_rows, int x_num_cols, int y_num_cols, int z_num_cols) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < x_num_rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < y_num_cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            for (int k = 0; k < x_num_cols; k++) {                
                sum += x[i * x_num_cols + k] * y[j * x_num_cols + k];
            }
            z[i * z_num_cols + j] = sum;
        }
    }
}

// Use a single array for z, but still access it as if it were divided in
extern "C" __global__ void matrix_matrix_mult_2(const float* x, const float* y, float* z, int x_num_rows, int x_num_cols, int y_num_cols, int x_offset, int y_offset) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < x_num_rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < y_num_cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            for (int k = 0; k < x_num_cols; k++) {                
                sum += x[i * x_num_cols + k] * y[j * x_num_cols + k];
            }
            z[(x_offset + i) * x_num_cols + (y_offset + j)] = sum;
        }
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark13M::alloc() {
    S = (N + P - 1) / P;
    PZ = P * P;
    // X is partitioned by rows, Y is partitioned by columns.
    // Z is partitioned in square blocks.
    // Assume that data in X are row-major, Y are column-major;
    x = (float **) malloc(sizeof(float*) * P);
    y = (float **) malloc(sizeof(float*) * P);
    // z = (float **) malloc(sizeof(float*) * PZ);
    for (int i = 0; i < P; i++) {
        err = cudaMallocManaged(&x[i], sizeof(float) * S * N);
        err = cudaMallocManaged(&y[i], sizeof(float) * S * N);
    }
    // for (int i = 0; i < PZ; i++) {
        // err = cudaMallocManaged(&z[i], sizeof(float) * S * S);
    // }
    err = cudaMallocManaged(&z, sizeof(float) * N * N);

    // Create P * P streams;
    s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * PZ);
    for (int i = 0; i < PZ; i++) {
        cudaSetDevice(select_gpu(i, max_devices));
        err = cudaStreamCreate(&s[i]);
    }
}

void Benchmark13M::init() {
}

void Benchmark13M::reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int p = i / S; 
            int s = (i * N + j) % (N * S);
            x[p][s] = float(i * N + j) / (N * N);
            y[p][s] = float(j * N + i) / (N * N);
        }
    }
    // for (int i = 0; i < P; i++) {
    //     for (int j = 0; j < S * N; j++) {
    //         std::cout << "x[" << i << "][" << j << "]=" << x[i][j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < P; i++) {
    //     for (int j = 0; j < S * N; j++) {
    //         std::cout << "y[" << i << "][" << j << "]=" << y[i][j] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
}

void Benchmark13M::execute_sync(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks / 2, num_blocks / 2);
    if (do_prefetch && pascalGpu) {
        for (int p1 = 0; p1 < P; p1++) {
            for (int p2 = 0; p2 < P; p2++) {
                int p = p1 * P + p2;
                // cudaMemPrefetchAsync(z[p], sizeof(float) * S * S, 0, 0);
                // Redundant prefetching in the sync implementation, but possibly necessary in multi-GPU;
                cudaMemPrefetchAsync(x[p1], sizeof(float) * S * N, 0, 0);
                cudaMemPrefetchAsync(y[p2], sizeof(float) * S * N, 0, 0);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaDeviceSynchronize();
    for (int p1 = 0; p1 < P; p1++) {
        for (int p2 = 0; p2 < P; p2++) {
            // matrix_matrix_mult_1<<<grid_size, block_size_2d_dim>>>(x[p1], y[p2], z[p1 * P + p2], std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), S);
            matrix_matrix_mult_2<<<grid_size, block_size_2d_dim>>>(x[p1], y[p2], z, std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), p1 * S, p2 * S);
            cudaDeviceSynchronize();
        }
    } 
}

void Benchmark13M::execute_async(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks / 2, num_blocks / 2);
    for (int p1 = 0; p1 < P; p1++) {
        for (int p2 = 0; p2 < P; p2++) {
            int p = p1 * P + p2;
            cudaSetDevice(select_gpu(p, max_devices));
            if (!pascalGpu || stream_attach) {
                // cudaStreamAttachMemAsync(s[p], z[p], sizeof(float) * S * S);
            }
            if (pascalGpu && do_prefetch) {
                cudaMemPrefetchAsync(x[p1], sizeof(float) * S * N, select_gpu(p, max_devices), s[p]);
                cudaMemPrefetchAsync(y[p2], sizeof(float) * S * N, select_gpu(p, max_devices), s[p]);
                // cudaMemPrefetchAsync(z[p], sizeof(float) * S * S, select_gpu(p, max_devices), s[p]);
            }
        }
    }
    for (int p1 = 0; p1 < P; p1++) {
        for (int p2 = 0; p2 < P; p2++) {
            int p = p1 * P + p2;
            cudaSetDevice(select_gpu(p, max_devices));
            // matrix_matrix_mult_1<<<grid_size, block_size_2d_dim, 0, s[p]>>>(x[p1], y[p2], z[p], std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), S);
            matrix_matrix_mult_2<<<grid_size, block_size_2d_dim, 0, s[p]>>>(x[p1], y[p2], z, std::min(S, N - p1 * S), N, std::min(S, N - p2 * S), p1 * S, p2 * S);
        }
    }
    for (int p1 = 0; p1 < P; p1++) {
        for (int p2 = 0; p2 < P; p2++) {
            err = cudaStreamSynchronize(s[p1 * P + p2]);
        }
    }
}

std::string Benchmark13M::print_result(bool short_form) {
    if (short_form) {
        // return std::to_string(z[0][0]);
    } else {
        int old_precision = std::cout.precision();
		std::cout.precision(2);
        std::string res = "[\n";
        for (int i = 0; i < std::min(30, N); i++) {
            res += "[";
            for (int j = 0; j < std::min(30, N); j++) {
                // int p1 = i / S; 
                // int p2 = j / S; 
                // res += std::to_string(z[p1 * P + p2][(i % S) * S + j % S]) + ", ";
                res += std::to_string(z[i * N + j]) + ", ";
            }
            res += "...]\n";
        }
        std::cout.precision(old_precision);
        return res + "...]";
    }
}