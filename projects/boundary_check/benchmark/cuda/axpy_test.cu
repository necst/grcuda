#define N 128
extern "C" __global__ void axpy_test(float *x, float y[N], float a, float res[N], int check[3]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > N) 
		check[1]++;
    res[i] = a * x[i] + y[i];
	printf("accessed kernel from thread %d\n", i);
}
