extern "C" __global__ void nested(int N, int *x, int *y) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int a = 0; a < N; ++a) {
        for (int b = 0; b < N; ++b) {
            for (int c = 0; c < N; ++c) {
                for (int d = 0; d < N; ++d) {
                    for (int e = 0; e < N; ++e) {
                        y[i] = x[i];
                    }
                }
            }
        }
    }
}