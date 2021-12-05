// Use Java System to measure time;
const System = Java.type("java.lang.System");
// Load GrCUDA;
const cu = Polyglot.eval("grcuda", "CU");

/////////////////////////////
/////////////////////////////

GAUSSIAN_BLUR = `
#include <cuda_fp16.h>
extern "C" __global__ void gaussian_blur(const int *image, float *result, int rows, int cols, const float* kernel, int diameter, __half cazzo) {
    extern __shared__ float kernel_local[];
    for(int i = threadIdx.x; i < diameter; i += blockDim.x) {
        for(int j = threadIdx.y; j < diameter; j += blockDim.y) {
            kernel_local[i * diameter + j] = kernel[i * diameter + j] + float(cazzo);
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

// Build the CUDA kernels;
const GAUSSIAN_BLUR_KERNEL = cu.buildkernel(GAUSSIAN_BLUR, "gaussian_blur", "const pointer, pointer, sint32, sint32, const pointer, sint32, half");

const N = 10;
const x = cu.DeviceArray("half", N * N);
// console.log(x);