#define NUM_THREADS_3 16

extern "C" __global__ void autocov(float *x, int k, int size, float *res) {
    __shared__ float cache[NUM_THREADS_3][NUM_THREADS_3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    cache[threadIdx.y][threadIdx.x] = x[i] * x[i + t];
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS_3 / 2;
    while (i > 0) {
        cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(&res[t], cache[threadIdx.y][0] / size);
    }
}