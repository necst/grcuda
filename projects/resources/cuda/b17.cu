#include "b17.cuh"

//////////////////////////////
//////////////////////////////

#define WARP_SIZE 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

/////////////////////////////
/////////////////////////////

extern "C" __global__ void spmv_multi(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows; n += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = ptr[n]; i < ptr[n + 1]; i++) {
            sum += val[i] * vec[idx[i]];
        }
        res[n] = sum;
    }
}

extern "C" __global__ void spmv2_multi(const int *ptr, const int *idx, const int *val, const float *vec, float *res, int num_rows, int num_nnz) {
    // Thread ID in block
    int t = threadIdx.x;

    // Thread ID in warp
    int lane = t & (WARP_SIZE - 1);

    // Number of warps per block
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    // One row per warp
    int row = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);

    extern __shared__ volatile float vals[];

    if (row < num_rows) {
        int rowStart = ptr[row];
        int rowEnd = ptr[row + 1];
        float sum = 0;

        // Use all threads in a warp accumulate multiplied elements
        for (int j = rowStart + lane; j < rowEnd; j += WARP_SIZE) {
            int col = idx[j];
            sum += val[j] * vec[col];
        }
        vals[t] = sum;
        __syncthreads();

        // Reduce partial sums
        if (lane < 16) vals[t] += vals[t + 16];
        if (lane < 8) vals[t] += vals[t + 8];
        if (lane < 4) vals[t] += vals[t + 4];
        if (lane < 2) vals[t] += vals[t + 2];
        if (lane < 1) vals[t] += vals[t + 1];
        __syncthreads();

        // Write result
        if (lane == 0) {
            res[row] = vals[t];
        }
    }
}

extern "C" __global__ void spmv3_multi(int *cudaRowCounter, int *d_ptr, int *d_cols, int *d_val, float *d_vector, float *d_out, int N) {
    int i;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x % THREADS_PER_VECTOR;       //lane index in the vector
    int vectorId = threadIdx.x / THREADS_PER_VECTOR;     //vector index in the thread block
    int warpLaneId = threadIdx.x & 31;                   //lane index in the warp
    int warpVectorId = warpLaneId / THREADS_PER_VECTOR;  //vector index in the warp

    __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {
        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (THREADS_PER_VECTOR == 32) {
            // Ensure aligned memory access
            i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

            // Process the aligned part
            for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = sum;
        }

        // Get a new row index
        if (warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;
    }
}

__inline__ __device__ float warp_reduce_multi(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

extern "C" __global__ void sum_multi(const float *x, float *z, int N) {
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce_multi(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);                     // The first thread in the warp updates the output;
}

extern "C" __global__ void divide_multi(const float *x, float *y, float *val, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] / val[0];
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark17::alloc() {
    nnz = degree * N;
    ptr_tmp = (int *)malloc(sizeof(int) * (N + 1));
    ptr2_tmp = (int *)malloc(sizeof(int) * (N + 1));
    idx_tmp = (int *)malloc(sizeof(int) * nnz);
    idx2_tmp = (int *)malloc(sizeof(int) * nnz);
    val_tmp = (int *)malloc(sizeof(int) * nnz);
    val2_tmp = (int *)malloc(sizeof(int) * nnz);

    cudaSetDevice(0);
    err = cudaMallocManaged(&ptr2, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&idx2, sizeof(int) * nnz);
    err = cudaMallocManaged(&val2, sizeof(int) * nnz);
    err = cudaMallocManaged(&hub1, sizeof(float) * N);
    err = cudaMallocManaged(&auth2, sizeof(float) * N);
    err = cudaMallocManaged(&rowCounter1, sizeof(int));
    err = cudaMallocManaged(&auth_norm, sizeof(float));
    err = cudaStreamCreate(&s1);

    cudaSetDevice(1);
    err = cudaMallocManaged(&ptr, sizeof(int) * (N + 1));
    err = cudaMallocManaged(&idx, sizeof(int) * nnz);
    err = cudaMallocManaged(&val, sizeof(int) * nnz);
    err = cudaMallocManaged(&auth1, sizeof(float) * N);
    err = cudaMallocManaged(&hub2, sizeof(float) * N);
    err = cudaMallocManaged(&rowCounter2, sizeof(int));
    err = cudaMallocManaged(&hub_norm, sizeof(float));
    err = cudaStreamCreate(&s2);


    x = (int *)malloc(nnz * sizeof(int));
    y = (int *)malloc(nnz * sizeof(int));
    v = (int *)malloc(nnz * sizeof(int));

}

void Benchmark17::init() {
    random_coo(x, y, v, N, degree);
    // Create a CSR;
    coo2csr(ptr_tmp, idx_tmp, val_tmp, x, y, v, N, N, nnz);
    coo2csr(ptr2_tmp, idx2_tmp, val2_tmp, y, x, v, N, N, nnz);
}

void Benchmark17::reset() {
    cudaSetDevice(0);

    for (int j = 0; j < nnz; j++) {
        idx2[j] = idx2_tmp[j];
        val2[j] = val2_tmp[j];
    }
    for (int j = 0; j < N + 1; j++) {
        ptr2[j] = ptr2_tmp[j];
    }
    for (int i = 0; i < N; i++) {
        auth2[i] = 1;
        hub1[i] = 1;
    }
    auth_norm[0] = 0;
    rowCounter1[0] = 0;
    cudaSetDevice(1);

    for (int j = 0; j < nnz; j++) {
        idx[j] = idx_tmp[j];
        val[j] = val_tmp[j];
    }
    for (int j = 0; j < N + 1; j++) {
        ptr[j] = ptr_tmp[j];
    }
    for (int i = 0; i < N; i++) {
        auth1[i] = 1;
        hub2[i] = 1;
    }
    hub_norm[0] = 0;
    rowCounter2[0] = 0;
}

void Benchmark17::execute_sync(int iter) {
    for (int iter = 0; iter < iterations; iter++) {
        // cudaMemPrefetchAsync(auth1, N * sizeof(float), 0);
        // cudaMemPrefetchAsync(auth2, N * sizeof(float), 0);
        // cudaMemPrefetchAsync(hub1, N * sizeof(float), 0);
        // cudaMemPrefetchAsync(hub2, N * sizeof(float), 0);
        // cudaMemPrefetchAsync(auth_norm, sizeof(float), 0);
        // cudaMemPrefetchAsync(hub_norm, sizeof(float), 0);
        // cudaDeviceSynchronize();

        int nb = ceil(N / ((float)block_size_1d));

        // spmv<<<nb, block_size_1d>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
        spmv3_multi<<<nb, block_size_1d, block_size_1d * sizeof(float)>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);
        err = cudaDeviceSynchronize();

        // spmv<<<nb, block_size_1d>>>(ptr, idx, val, auth1, hub2, N, nnz);
        spmv3_multi<<<nb, block_size_1d, block_size_1d * sizeof(float)>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);
        err = cudaDeviceSynchronize();

        sum_multi<<<num_blocks, block_size_1d>>>(auth2, auth_norm, N);
        err = cudaDeviceSynchronize();

        sum_multi<<<num_blocks, block_size_1d>>>(hub2, hub_norm, N);
        err = cudaDeviceSynchronize();

        divide_multi<<<num_blocks, block_size_1d>>>(auth2, auth1, auth_norm, N);
        err = cudaDeviceSynchronize();

        divide_multi<<<num_blocks, block_size_1d>>>(hub2, hub1, hub_norm, N);
        err = cudaDeviceSynchronize();

        auth_norm[0] = 0;
        hub_norm[0] = 0;
        rowCounter1[0] = 0;
        rowCounter2[0] = 0;

        if (debug && err) std::cout << err << std::endl;
    }
}

void Benchmark17::execute_async(int iter) {
    for (int iter = 0; iter < iterations; iter++) {
        // cudaMemPrefetchAsync(auth1, N * sizeof(float), 0, s2);
        // cudaMemPrefetchAsync(auth2, N * sizeof(float), 0, s1);
        // cudaMemPrefetchAsync(hub1, N * sizeof(float), 0, s1);
        // cudaMemPrefetchAsync(hub2, N * sizeof(float), 0, s2);
        // cudaMemPrefetchAsync(auth_norm, sizeof(float), 0, s1);
        // cudaMemPrefetchAsync(hub_norm, sizeof(float), 0, s2);
        cudaSetDevice(0);
        cudaStreamAttachMemAsync(s1, ptr2, 0);
        cudaStreamAttachMemAsync(s1, idx2, 0);
        cudaStreamAttachMemAsync(s1, val2, 0);
        cudaStreamAttachMemAsync(s1, hub1, 0);
        cudaStreamAttachMemAsync(s1, auth2, 0);
        cudaStreamAttachMemAsync(s1, rowCounter1, 0);
        cudaStreamAttachMemAsync(s1, auth_norm, 0);
        cudaEvent_t e1;
        cudaEventCreate(&e1);

        cudaSetDevice(1);
        cudaStreamAttachMemAsync(s2, ptr, 0);
        cudaStreamAttachMemAsync(s2, idx, 0);
        cudaStreamAttachMemAsync(s2, val, 0);
        cudaStreamAttachMemAsync(s2, auth1, 0);
        cudaStreamAttachMemAsync(s2, hub2, 0);
        cudaStreamAttachMemAsync(s2, rowCounter2, 0);
        cudaStreamAttachMemAsync(s2, hub_norm, 0);
        cudaEvent_t e2;
        cudaEventCreate(&e2);


        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(auth1, N * sizeof(float), 1, s2);
            cudaMemPrefetchAsync(auth2, N * sizeof(float), 0, s1);
            cudaMemPrefetchAsync(hub1, N * sizeof(float), 0, s1);
            cudaMemPrefetchAsync(hub2, N * sizeof(float), 1, s2);
            cudaMemPrefetchAsync(auth_norm, sizeof(float), 0, s1);
            cudaMemPrefetchAsync(hub_norm, sizeof(float), 1, s2);
        }
        int nb = ceil(N / ((float)block_size_1d));
        cudaSetDevice(0);
        // spmv<<<nb, block_size_1d, 0, s1>>>(ptr2, idx2, val2, hub1, auth2, N, nnz);
        spmv3_multi<<<nb, block_size_1d, block_size_1d * sizeof(float), s1>>>(rowCounter1, ptr2, idx2, val2, hub1, auth2, N);
        err = cudaEventRecord(e1, s1);

        cudaSetDevice(1);
        // spmv<<<nb, block_size_1d, 0, s2>>>(ptr, idx, val, auth1, hub2, N, nnz);
        spmv3_multi<<<nb, block_size_1d, block_size_1d * sizeof(float), s2>>>(rowCounter2, ptr, idx, val, auth1, hub2, N);
        err = cudaEventRecord(e2, s2);

        cudaSetDevice(0);
        sum_multi<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth_norm, N);
        cudaSetDevice(1);
        sum_multi<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub_norm, N);

        // Stream 1 waits stream 2;
        cudaSetDevice(0);
        err = cudaStreamWaitEvent(s1, e2, 0);
        cudaStreamAttachMemAsync(s1, auth1, 0);

        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(auth1, N * sizeof(float), 1, s1);
        }
        divide_multi<<<num_blocks, block_size_1d, 0, s1>>>(auth2, auth1, auth_norm, N);
        // Stream 2 waits stream 1;
        cudaSetDevice(1);
        err = cudaStreamWaitEvent(s2, e1, 0);
        cudaStreamAttachMemAsync(s2, hub1, 0);
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(hub1, N * sizeof(float), 1, s2);
        }
        divide_multi<<<num_blocks, block_size_1d, 0, s2>>>(hub2, hub1, hub_norm, N);

        cudaSetDevice(0);
        err = cudaStreamSynchronize(s1);
        auth_norm[0] = 0;
        rowCounter1[0] = 0;

        cudaSetDevice(1);
        err = cudaStreamSynchronize(s2);
        hub_norm[0] = 0;
        rowCounter2[0] = 0;

        if (debug && err) std::cout << err << std::endl;
    }
}

void Benchmark17::execute_cudagraph(int iter) {}

void Benchmark17::execute_cudagraph_manual(int iter) {}

void Benchmark17::execute_cudagraph_single(int iter) {}
std::string Benchmark17::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(auth1[0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < 10; j++) {
            res += std::to_string(auth1[j]) + ", ";
        }
        return res + ", ...]";
    }
}