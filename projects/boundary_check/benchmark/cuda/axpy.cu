extern "C" __global__ void axpy(float *x, float *y, float a, float *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a * x[i] + y[i];
}
