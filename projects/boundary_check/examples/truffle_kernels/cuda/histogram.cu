#define BIN_COUNT 1024
#define WARP_LOG_SIZE 5
#define WARP_N 3
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)

extern "C" __global__ void histogram(int *d_Result, float *d_Data, float minimum, float maximum, int dataN) {

    //Current global thread index
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    //Total number of threads in the compute grid
    const int numThreads = blockDim.x * gridDim.x;
    //WARP_LOG_SIZE higher bits of counter values are tagged
    //by lower WARP_LOG_SIZE threadID bits
    // Will correctly issue warning when compiling for debug (x<<32-0)
    const unsigned int threadTag = threadIdx.x << (32 - WARP_LOG_SIZE);
    //Shared memory cache for each warp in current thread block
    //Declare as volatile to prevent incorrect compiler optimizations in addPixel()
    volatile __shared__ unsigned int s_Hist[BLOCK_MEMORY];
    //Current warp shared memory frame
    const int warpBase = (threadIdx.x >> WARP_LOG_SIZE) * BIN_COUNT;

    //Clear shared memory buffer for current thread block before processing
    for (int pos = threadIdx.x; pos < BLOCK_MEMORY; pos += blockDim.x)
        s_Hist[pos] = 0;

    __syncthreads();
    //Cycle through the entire data set, update subhistograms for each warp
    //Since threads in warps always execute the same instruction,
    //we are safe with the addPixel trick
    for (int pos = globalTid; pos < dataN; pos += numThreads) {
        unsigned int data4 = ((d_Data[pos] - minimum) / (maximum - minimum)) * BIN_COUNT;
        unsigned int count;
        do {
            count = s_Hist[data4 & 0x3FFU + warpBase] & 0x07FFFFFFU;
            count = threadTag | (count + 1);
            s_Hist[data4 & 0x3FFU + warpBase] = count;
        } while (s_Hist[data4 & 0x3FFU + warpBase] != count);
    }

    __syncthreads();
    //Merge per-warp histograms into per-block and write to global memory
    for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x) {
        unsigned int sum = 0;

        for (int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT)
            sum += s_Hist[base + pos] & 0x07FFFFFFU;
        atomicAdd(&d_Result[pos], sum);
    }
}
