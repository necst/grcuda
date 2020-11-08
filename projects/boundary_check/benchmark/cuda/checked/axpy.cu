extern "C" __global__ void axpy_checked(float *x, float *y, float a, int size, float *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        res[i] = a * x[i] + y[i];
    }
}
