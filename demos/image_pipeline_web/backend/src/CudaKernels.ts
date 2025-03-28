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

export const GAUSSIAN_BLUR = `
extern "C" __global__ void gaussian_blur(const int *image, float *result, int rows, int cols, const float* kernel, int diameter) {
    extern __shared__ float kernel_local[];
    for(int i = threadIdx.x; i < diameter; i += blockDim.x) {
        for(int j = threadIdx.y; j < diameter; j += blockDim.y) {
            kernel_local[i * diameter + j] = kernel[i * diameter + j];
        }
    }
    __syncthreads();

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            int radius = diameter / 2;
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
                        sum += kernel_local[(x + radius) * diameter + (y + radius)] * (float(image[nx * cols + ny]) / 255);
                    }
                }
            }
            result[i * cols + j] = sum;
        }
    }
}
`

export const SOBEL = `
extern "C" __global__ void sobel(float *image, float *result, int rows, int cols) {
    // int SOBEL_X[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    // int SOBEL_Y[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    __shared__ int SOBEL_X[9];
    __shared__ int SOBEL_Y[9];
    if (threadIdx.x == 0 && threadIdx.y == 0) {   
        SOBEL_X[0] = -1;
        SOBEL_X[1] = -2;
        SOBEL_X[2] = -1;
        SOBEL_X[3] = 0;
        SOBEL_X[4] = 0;
        SOBEL_X[5] = 0;
        SOBEL_X[6] = 1;
        SOBEL_X[7] = 2;
        SOBEL_X[8] = 1;

        SOBEL_Y[0] = -1;
        SOBEL_Y[1] = 0;
        SOBEL_Y[2] = 1;
        SOBEL_Y[3] = -2;
        SOBEL_Y[4] = 0;
        SOBEL_Y[5] = 2;
        SOBEL_Y[6] = -1;
        SOBEL_Y[7] = 0;
        SOBEL_Y[8] = 1;
    }
    __syncthreads();
    
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
            float sum_gradient_x = 0.0, sum_gradient_y = 0.0;
            int radius = 1;
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
                        float neighbour = image[nx * cols + ny];
                        int s = (x + radius) * 3 + y + radius;
                        sum_gradient_x += SOBEL_X[s] * neighbour;
                        sum_gradient_y += SOBEL_Y[s] * neighbour;
                    }
                }
            }
            result[i * cols + j] = sqrt(sum_gradient_x * sum_gradient_x + sum_gradient_y * sum_gradient_y);
        }
    }
}
`

export const EXTEND_MASK = `
__device__ float atomicMinf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    // If val is smaller than current, don't do anything, else update the current value atomically;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    }
    return __int_as_float(old);
}

__inline__ __device__ float warp_reduce_max(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_min(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) 
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

extern "C" __global__ void maximum(float *in, float* out, int N) {
    int warp_size = 32;
    float maximum = -1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        maximum = max(maximum, in[i]);
    }
    maximum = warp_reduce_max(maximum); // Obtain the max of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMaxf(out, maximum); // The first thread in the warp updates the output;
}

extern "C" __global__ void minimum(float *in, float* out, int N) {
    int warp_size = 32;
    float minimum = 1000;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { 
        minimum = min(minimum, in[i]);
    }
    minimum = warp_reduce_min(minimum); // Obtain the min of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMinf(out, minimum); // The first thread in the warp updates the output;
}

extern "C" __global__ void extend(float *x, const float *minimum, const float *maximum, int n, int extend_factor) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = extend_factor * (x[i] - *minimum) / (*maximum - *minimum);
        x[i] = res_tmp > 1 ? 1 : res_tmp;
    }
}
`

export const UNSHARPEN = `
extern "C" __global__ void unsharpen(const int *x, const float *y, float *res, float amount, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        float res_tmp = (float(x[i]) / 255) * (1 + amount) - y[i] * amount;
        res_tmp = res_tmp > 1 ? 1 : res_tmp;
        res[i] = res_tmp < 0 ? 0 : res_tmp;
    }
}
`

export const COMBINE = `
extern "C" __global__ void combine(const float *x, const float *y, const float *mask, float *res, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
    }
}
`

export const COMBINE_2 = `
extern "C" __global__ void combine_lut(const float *x, const float *y, const float *mask, int *res, int n, int* lut) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        res[i] = lut[min(256 - 1, int(256 * (x[i] * mask[i] + y[i] * (1 - mask[i]))))];
    }
}
`

export const RESET = `
extern "C" __global__ void reset(float *x, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        x[i] = 0.0;
    }
}
`

export const INT_TO_FLOAT = `
extern "C" __global__ void int_to_float(const int *x, float *y, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        y[i] = float(x) / 255;
    }
}
`

export const FLOAT_TO_INT = `
extern "C" __global__ void float_to_int(const float *x, int *y, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { 
        y[i] = int(x[i] * 255);
    }
}
`