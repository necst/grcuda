#include "b11.cuh"
#include <thread>
#include <vector>
//////////////////////////////
//////////////////////////////

__global__ void squareMulti(const float *x, float *y, int n)
{
    for (int i  = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        // float tmp = x[i];
        // float sum = 0;
        // for (int j = 0; j < 4; j++) {
        //     sum += tmp + j;
        // }

        y[i] = x[i] * x[i]; // tmp + tmp * tmp / 2 + tmp * tmp * tmp / 6;
    }
}

__inline__ __device__ float warp_reduceMulti(float val)
{
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// __device__ float atomicAddDouble(float* address, float val) {
//     unsigned long long int* address_as_ull = (unsigned long long int*) address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed, __float_as_longlong(val + __longlong_as_float(assumed)));
//     } while (assumed != old);
//     return __longlong_as_float(old);
// }

__global__ void reduceMulti(const float *x, const float *y, float *z, int N)
{
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum += x[i] - y[i];
    }
    sum = warp_reduceMulti(sum);                   // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                    // The first thread in the warp updates the output;
}

//////////////////////////////
//////////////////////////////

void Benchmark11::alloc()
{
    cudaSetDevice(0);            // Set device 0 as current
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&x1, sizeof(float) * N);
    err = cudaStreamCreate(&s1);
    cudaSetDevice(1);            // Set device 1 as current
    err = cudaMallocManaged(&y, sizeof(float) * N);
    err = cudaMallocManaged(&y1, sizeof(float) * N);
    err = cudaMallocManaged(&res, sizeof(float));
    err = cudaStreamCreate(&s2);
}

void Benchmark11::init()
{
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
}

void Benchmark11::reset()
{
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0 / (i + 1);
        y[i] = 2.0 / (i + 1);
    }
    res[0] = 0.0;
}

void Benchmark11::execute_sync(int iter)
{
    squareMulti<<<num_blocks, block_size_1d>>>(x, x1, N);
    err = cudaDeviceSynchronize();
    squareMulti<<<num_blocks, block_size_1d>>>(y, y1, N);
    err = cudaDeviceSynchronize();
    reduceMulti<<<num_blocks, block_size_1d>>>(x1, y1, res, N);
    err = cudaDeviceSynchronize();
}


void Benchmark11::execute_async(int iter)
{


    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(x, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(x1, sizeof(float) * N, 0, s1);
        cudaMemPrefetchAsync(y, sizeof(float) * N, 1, s2);
        cudaMemPrefetchAsync(y1, sizeof(float) * N, 1, s2);
        cudaMemPrefetchAsync(res, sizeof(float), 0, s1);
    }

    cudaSetDevice(0);            // Set device 0 as current


    cudaStreamAttachMemAsync(s1, x, sizeof(float) * N);
    cudaStreamAttachMemAsync(s1, x1, sizeof(float) * N);
    squareMulti<<<num_blocks, block_size_1d, 0, s1>>>(x, x1, N);

    cudaSetDevice(1);            // Set device 1 as current
    cudaStreamAttachMemAsync(s2, y, sizeof(float) * N);
    cudaStreamAttachMemAsync(s2, y1, sizeof(float) * N);
    squareMulti<<<num_blocks, block_size_1d, 0, s2>>>(y, y1, N);

    // Stream 1 waits stream 2;
    cudaEvent_t e1;
    cudaEventCreate(&e1);
    cudaEventRecord(e1, s2);
    cudaStreamWaitEvent(s1, e1, 0);
    cudaSetDevice(0);

    
    cudaStreamAttachMemAsync(s1, y1, sizeof(float) * N);

    if (pascalGpu && do_prefetch) {
        cudaMemPrefetchAsync(y1, sizeof(float) * N, 0, s1);
    }
    reduceMulti<<<num_blocks, block_size_1d, 0, s1>>>(x1, y1, res, N);
    cudaStreamSynchronize(s1);




}





void Benchmark11::execute_cudagraph(int iter) {}

void Benchmark11::execute_cudagraph_manual(int iter) {}

void Benchmark11::execute_cudagraph_single(int iter) {}

std::string Benchmark11::print_result(bool short_form)
{
    return std::to_string(res[0]);
}
