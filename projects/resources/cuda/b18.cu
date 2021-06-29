#include "b18.cuh"

//////////////////////////////
//////////////////////////////

extern "C" __global__ void gaussian_blur_multi(const float *image, float *result, int rows, int cols, const float *kernel, int diameter) {
    extern __shared__ float kernel_local[];
    for (int i = threadIdx.x; i < diameter; i += blockDim.x) {
        for (int j = threadIdx.y; j < diameter; j += blockDim.y) {
            kernel_local[i * diameter + j] = kernel[i * diameter + j];
        }
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
            float sum = 0;
            int radius = diameter / 2;
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
                        sum += kernel_local[(x + radius) * diameter + (y + radius)] * image[nx * cols + ny];
                    }
                }
            }
            result[i * cols + j] = sum;
        }
    }
}

extern "C" __global__ void sobel_multi(const float *image, float *result, int rows, int cols) {
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

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
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

__device__ float atomicMinf_multi(float *address, float val) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ float atomicMaxf_multi(float *address, float val) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    // If val is smaller than current, don't do anything, else update the current value atomically;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    }
    return __int_as_float(old);
}

__inline__ __device__ float warp_reduce_max_multi(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_min_multi(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

extern "C" __global__ void maximum_kernel_multi(const float *in, float *out, int N) {
    int warp_size = 32;
    float maximum = -1000;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        maximum = max(maximum, in[i]);
    }
    maximum = warp_reduce_max_multi(maximum);        // Obtain the max of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMaxf_multi(out, maximum);              // The first thread in the warp updates the output;
}

extern "C" __global__ void minimum_kernel_multi(const float *in, float *out, int N) {
    int warp_size = 32;
    float minimum = 1000;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        minimum = min(minimum, in[i]);
    }
    minimum = warp_reduce_min_multi(minimum);        // Obtain the min of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicMinf_multi(out, minimum);              // The first thread in the warp updates the output;
}

extern "C" __global__ void extend_multi(float *x, const float *minimum, const float *maximum, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float res_tmp = 5 * (x[i] - *minimum) / (*maximum - *minimum);
        x[i] = res_tmp > 1 ? 1 : res_tmp;
    }
}

extern "C" __global__ void unsharpen_multi(const float *x, const float *y, float *res, float amount, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float res_tmp = x[i] * (1 + amount) - y[i] * amount;
        res_tmp = res_tmp > 1 ? 1 : res_tmp;
        res[i] = res_tmp < 0 ? 0 : res_tmp;
    }
}

extern "C" __global__ void combine_multi(const float *x, const float *y, const float *mask, float *res, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
    }
}

extern "C" __global__ void reset_image_multi(float *x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] = 0.0;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark18::alloc() {
    cudaSetDevice(0);            // Set device 0 as current
    //s1
    err = cudaMallocManaged(&blurred_small, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_small, sizeof(float) * N * N);
    err = cudaMallocManaged(&image3, sizeof(float) * N * N);
    //s3
    err = cudaMallocManaged(&blurred_unsharpen, sizeof(float) * N * N);
    err = cudaMallocManaged(&image_unsharpen, sizeof(float) * N * N);
    err = cudaMallocManaged(&kernel_small, sizeof(float) * kernel_small_diameter * kernel_small_diameter);

    err = cudaMallocManaged(&kernel_unsharpen, sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter);
    err = cudaMallocManaged(&image_gpu0, sizeof(float) * N * N);

    err = cudaStreamCreate(&s1);
    err = cudaStreamCreate(&s3);

    cudaSetDevice(1);            // Set device 1 as current
    //s2
    err = cudaMallocManaged(&blurred_large, sizeof(float) * N * N);
    err = cudaMallocManaged(&mask_large, sizeof(float) * N * N);
    err = cudaMallocManaged(&image2, sizeof(float) * N * N);
    err = cudaMallocManaged(&kernel_large, sizeof(float) * kernel_large_diameter * kernel_large_diameter);
    err = cudaMallocManaged(&maximum, sizeof(float));
    err = cudaMallocManaged(&minimum, sizeof(float));
    err = cudaMallocManaged(&image_gpu1, sizeof(float) * N * N);

    //image duplicated for only read
    //err = cudaMallocManaged(&image, sizeof(float) * N * N);

    //err = cudaMallocManaged(&mask_unsharpen, sizeof(float) * N * N);
    err = cudaStreamCreate(&s2);
    err = cudaStreamCreate(&s4);
    err = cudaStreamCreate(&s5);

}

void Benchmark18::init() {
    cudaSetDevice(0);            // Set device 0 as current

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image_gpu0[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(kernel_small, kernel_small_diameter, 1);
    gaussian_kernel(kernel_unsharpen, kernel_unsharpen_diameter, 5);

    cudaSetDevice(1);            // Set device 1 as current

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            image_gpu1[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(kernel_large, kernel_large_diameter, 10);
}

void Benchmark18::reset() {
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         image3[i * N + j] = 0;
    //     }
    // }
    cudaSetDevice(0);            // Set device 0 as current
    memset(image3, 0, N * N * sizeof(float));
    reset_image_multi<<<num_blocks, block_size_1d>>>(image3, N * N);
    cudaDeviceSynchronize();

    cudaSetDevice(1);            // Set device 1 as current
    *maximum = 0;
    *minimum = 0;
}

void Benchmark18::execute_sync(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    dim3 grid_size_2(num_blocks / 2, num_blocks / 2);

    gaussian_blur_multi<<<grid_size_2, block_size_2d_dim, kernel_small_diameter * kernel_small_diameter * sizeof(float)>>>(image, blurred_small, N, N, kernel_small, kernel_small_diameter);
    cudaDeviceSynchronize();

    gaussian_blur_multi<<<grid_size_2, block_size_2d_dim, kernel_large_diameter * kernel_large_diameter * sizeof(float)>>>(image, blurred_large, N, N, kernel_large, kernel_large_diameter);
    cudaDeviceSynchronize();

    gaussian_blur_multi<<<grid_size_2, block_size_2d_dim, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float)>>>(image, blurred_unsharpen, N, N, kernel_unsharpen, kernel_unsharpen_diameter);
    cudaDeviceSynchronize();

    sobel_multi<<<grid_size_2, block_size_2d_dim>>>(blurred_small, mask_small, N, N);
    cudaDeviceSynchronize();

    sobel_multi<<<grid_size_2, block_size_2d_dim>>>(blurred_large, mask_large, N, N);
    cudaDeviceSynchronize();

    maximum_kernel_multi<<<num_blocks, block_size_1d>>>(mask_large, maximum, N * N);
    cudaDeviceSynchronize();

    minimum_kernel_multi<<<num_blocks, block_size_1d>>>(mask_large, minimum, N * N);
    cudaDeviceSynchronize();

    extend_multi<<<num_blocks, block_size_1d>>>(mask_large, minimum, maximum, N * N);
    cudaDeviceSynchronize();

    unsharpen_multi<<<num_blocks, block_size_1d>>>(image, blurred_unsharpen, image_unsharpen, 0.5, N * N);
    cudaDeviceSynchronize();

    combine_multi<<<num_blocks, block_size_1d>>>(image_unsharpen, blurred_large, mask_large, image2, N * N);
    cudaDeviceSynchronize();

    combine_multi<<<num_blocks, block_size_1d>>>(image2, blurred_small, mask_small, image3, N * N);

    // Extra
    // combine<<<num_blocks, block_size_1d>>>(blurred_small, blurred_large, blurred_unsharpen, image3, N * N);

    cudaDeviceSynchronize();
}

void Benchmark18::execute_async(int iter) {
    dim3 block_size_2d_dim(block_size_2d, block_size_2d);
    dim3 grid_size(num_blocks, num_blocks);
    int nb = num_blocks / 2;
    dim3 grid_size_2(nb, nb);

    cudaSetDevice(0);            // Set device 0 as current
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, blurred_small, 0);
        cudaStreamAttachMemAsync(s1, mask_small, 0);
        cudaStreamAttachMemAsync(s3, blurred_unsharpen, 0);
        cudaStreamAttachMemAsync(s3, image_unsharpen, 0);
        cudaStreamAttachMemAsync(s1, image3, 0);
    }

    cudaSetDevice(1);            // Set device 1 as current
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s2, blurred_large, 0);
        cudaStreamAttachMemAsync(s2, mask_large, 0);
        cudaStreamAttachMemAsync(s2, image2, 0);
    }
    cudaSetDevice(0);            // Set device 0 as current

    gaussian_blur_multi<<<grid_size_2, block_size_2d_dim, kernel_small_diameter * kernel_small_diameter * sizeof(float), s1>>>(image_gpu0, blurred_small, N, N, kernel_small, kernel_small_diameter);
    gaussian_blur_multi<<<grid_size_2, block_size_2d_dim, kernel_unsharpen_diameter * kernel_unsharpen_diameter * sizeof(float), s3>>>(image_gpu0, blurred_unsharpen, N, N, kernel_unsharpen, kernel_unsharpen_diameter);
    sobel_multi<<<grid_size_2, block_size_2d_dim, 0, s1>>>(blurred_small, mask_small, N, N);
    cudaSetDevice(1);            // Set device 1 as current
    gaussian_blur_multi<<<grid_size_2, block_size_2d_dim, kernel_large_diameter * kernel_large_diameter * sizeof(float), s2>>>(image_gpu1, blurred_large, N, N, kernel_large, kernel_large_diameter);
    sobel_multi<<<grid_size_2, block_size_2d_dim, 0, s2>>>(blurred_large, mask_large, N, N);

    cudaEvent_t e1, e2, e3, e4, e5;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);
    cudaEventCreate(&e5);

    cudaEventRecord(e1, s2);
    cudaStreamWaitEvent(s5, e1, 0);
    maximum_kernel_multi<<<num_blocks, block_size_1d, 0, s5>>>(mask_large, maximum, N * N);

    cudaStreamWaitEvent(s4, e1, 0);
    minimum_kernel_multi<<<num_blocks, block_size_1d, 0, s4>>>(mask_large, minimum, N * N);

    cudaEventRecord(e2, s4);
    cudaEventRecord(e5, s5);

    cudaStreamWaitEvent(s2, e2, 0);
    cudaStreamWaitEvent(s2, e5, 0);

    extend_multi<<<num_blocks, block_size_1d, 0, s2>>>(mask_large, minimum, maximum, N * N);

    cudaSetDevice(0);            // Set device 0 as current
    unsharpen_multi<<<num_blocks, block_size_1d, 0, s3>>>(image_gpu0, blurred_unsharpen, image_unsharpen, 0.5, N * N);
    cudaEventRecord(e3, s3);
    cudaStreamWaitEvent(s2, e3, 0);

    cudaSetDevice(1);            // Set device 1 as current

    combine_multi<<<num_blocks, block_size_1d, 0, s2>>>(image_unsharpen, blurred_large, mask_large, image2, N * N);
    cudaEventRecord(e4, s2);
    cudaStreamWaitEvent(s1, e4, 0);

    cudaSetDevice(0);            // Set device 0 as current
    if (!pascalGpu || stream_attach) {
        cudaStreamAttachMemAsync(s1, image2, 0);
    }
    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(image3, N * N * sizeof(float), 0, s1);
    }
    combine_multi<<<num_blocks, block_size_1d, 0, s1>>>(image2, blurred_small, mask_small, image3, N * N);

    // Extra
    // cudaEventRecord(e1, s2);
    // cudaEventRecord(e2, s3);
    // cudaStreamWaitEvent(s1, e1, 0);
    // cudaStreamWaitEvent(s1, e2, 0);
    // combine<<<num_blocks, block_size_1d, 0, s1>>>(blurred_small, blurred_large, blurred_unsharpen, image3, N * N);

    cudaStreamSynchronize(s1);
}

void Benchmark18::execute_cudagraph(int iter) {}

void Benchmark18::execute_cudagraph_manual(int iter) {}
void Benchmark18::execute_cudagraph_single(int iter) {}
std::string Benchmark18::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(image3[0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < 10; j++) {
            res += std::to_string(image3[j]) + ", ";
        }
        return res + ", ...]";
    }
}