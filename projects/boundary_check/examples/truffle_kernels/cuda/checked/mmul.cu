extern "C" __global__ void mmul_checked(float *X, float *Y, int X_dim_col, int X_dim_row, int Y_dim_row, float *Z) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < X_dim_col && c < Y_dim_row) {
        float res = 0;
        for (uint i = 0; i < X_dim_row; i++) {
            res += X[X_dim_row * r + i] * Y[Y_dim_row * i + c];
        }
        Z[Y_dim_row * r + c] = res;
    }
}
