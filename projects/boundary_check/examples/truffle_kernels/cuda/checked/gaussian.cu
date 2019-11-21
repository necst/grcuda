extern "C" __global__ void gaussian(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t) {
    if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
        return;
    if (threadIdx.y + blockIdx.y * blockDim.y >= Size - t)
        return;

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    a_cuda[Size * (xidx + 1 + t) + (yidx + t)] -= m_cuda[Size * (xidx + 1 + t) + t] * a_cuda[Size * t + (yidx + t)];
    if (yidx == 0) {
        b_cuda[xidx + 1 + t] -= m_cuda[Size * (xidx + 1 + t) + (yidx + t)] * b_cuda[t];
    }
}