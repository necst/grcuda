#define NUM_THREADS 128

extern "C" __global__ void dot_product_checked(float *x, float *y, int size, float *res) {
    __shared__ float cache[NUM_THREADS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        cache[threadIdx.x] = x[i] * y[i];
    }
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS / 2;
    while (i > 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(res, cache[0]);
    }
}

