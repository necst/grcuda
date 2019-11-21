#define NUM_THREADS 128

extern "C" __global__ void dot_product(float *x, float *y, int size, float *res) {
    __shared__ float cache[NUM_THREADS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // This OoB access doesn't affect the result;
    cache[threadIdx.x] = x[i] * y[i];
    __syncthreads();

    // Perform tree reduction;
    i = NUM_THREADS / 2;
    while (i > 0) {
        // CUDA fails if an OoB access is done on shared memory.
        // Automatically fixing this pattern is not trivial:
        //   the starting value of "i" and the size of "cache" are known at compile time, 
        //   so we could add a check like "threadIdx.x + i < NUM_THREADS".
        //     This fixes the issue, but it's not logically equivalent to the "correct" solution below,
        //   and results in wasted computation;
        cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(res, cache[0]);
    }
}

