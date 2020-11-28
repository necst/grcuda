extern "C" __global__ void spmv(int *ptr, int *idx, float *res, float *vec, float *val) {

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float r = 0;
    for (int i = ptr[v]; i < ptr[v + 1]; i++) {
        r += val[i] * vec[idx[i]];
    }
    res[v] = r;
}