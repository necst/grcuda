#include "b15.cuh"

//////////////////////////////
//////////////////////////////


__device__ inline double
cndGPUMulti(double d) {
    const double A1 = 0.31938153f;
    const double A2 = -0.356563782f;
    const double A3 = 1.781477937f;
    const double A4 = -1.821255978f;
    const double A5 = 1.330274429f;
    const double RSQRT2PI = 0.39894228040143267793994605993438f;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(-0.5f * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}
extern "C" __global__ void
bsMulti(const double *x, double *y, int N, double R, double V, double T, double K) {
    
    double sqrtT = 1.0 / rsqrt(T);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        double expRT;
        double d1, d2, CNDD1, CNDD2;
        d1 = (log(x[i] / K) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPUMulti(d1);
        CNDD2 = cndGPUMulti(d2);

        // Calculate Call and Put simultaneously
        expRT = exp(-R * T);
        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;
    }
}

//////////////////////////////
//////////////////////////////

void Benchmark15::alloc() {
    x = (double **)malloc(sizeof(double *) * M);
    y = (double **)malloc(sizeof(double *) * M);
    tmp_x = (double *)malloc(sizeof(double) * N);
    // cudaHostRegister(tmp_x, sizeof(double) * N, 0);

    for (int i = 0; i < M; i++) {
        if(i%2 == 0){
            cudaSetDevice(0);            // Set device 0 as current
        }else{
            cudaSetDevice(1);            // Set device 1 as current
        }

        cudaMallocManaged(&x[i], sizeof(double) * N);
        cudaMallocManaged(&y[i], sizeof(double) * N);
    }


}

void Benchmark15::init() {
    for (int j = 0; j < N; j++) {
        tmp_x[j] = 60 - 0.5 + (double)rand() / RAND_MAX;
        for (int i = 0; i < M; i++) {
            x[i][j] = tmp_x[j];
            // y[i][j] = 0;
        }
    }

    s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * M);
    for (int i = 0; i < M; i++) {
        if(i%2 == 0){
            cudaSetDevice(0);            // Set device 0 as current
        }else{
            cudaSetDevice(1);            // Set device 1 as current
        }
        err = cudaStreamCreate(&s[i]);
    }
}

void Benchmark15::reset() {
    for (int i = 0; i < M; i++) {
        // memcpy(x[i], y, sizeof(int) * N);
        // cudaMemcpy(x[i], y, sizeof(double) * N, cudaMemcpyDefault);

        // cudaMemcpyAsync(x[i], y, sizeof(int) * N, cudaMemcpyHostToDevice,
        // s[i]);
        for (int j = 0; j < N; j++) {
            x[i][j] = tmp_x[j];
        }
    }
    // cudaMemPrefetchAsync(x[0], sizeof(double) * N, 0, s[0]);
}

void Benchmark15::execute_sync(int iter) {
    double R = 0.08;
    double V = 0.3;
    double T = 1.0;
    double K = 60.0;
    for (int j = 0; j < M; j++) {
        bsMulti<<<num_blocks, block_size_1d>>>(x[j], y[j], N, R, V, T, K);
        err = cudaDeviceSynchronize();
    }
}

void Benchmark15::execute_async(int iter) {
    double R = 0.08;
    double V = 0.3;
    double T = 1.0;
    double K = 60.0;
    for (int j = 0; j < M; j++) {
        if(j%2 == 0){
            cudaSetDevice(0);            // Set device 0 as current
        }else{
            cudaSetDevice(1);            // Set device 1 as current
        }
        cudaStreamAttachMemAsync(s[j], x[j], sizeof(double) * N);
        cudaStreamAttachMemAsync(s[j], y[j], sizeof(double) * N);
        if (pascalGpu && do_prefetch) {
            cudaMemPrefetchAsync(x[j], sizeof(double) * N, j%2, s[j]);
            cudaMemPrefetchAsync(y[j], sizeof(double) * N, j%2, s[j]);
        }
        // if (j > 0) cudaMemPrefetchAsync(y[j - 1], sizeof(double) * N, cudaCpuDeviceId, s[j - 1]);
        bsMulti<<<num_blocks, block_size_1d, 0, s[j]>>>(x[j], y[j], N, R, V, T, K);
        // if (j < M - 1) cudaMemPrefetchAsync(x[j + 1], sizeof(double) * N, 0, s[j + 1]);
    }

    // Last tile;
    // cudaMemPrefetchAsync(y[M - 1], sizeof(double) * N, cudaCpuDeviceId, s[M - 1]);

    for (int j = 0; j < M; j++) {
        if(j%2 == 0){
            cudaSetDevice(0);            // Set device 0 as current
        }else{
            cudaSetDevice(1);            // Set device 1 as current
        }
        err = cudaStreamSynchronize(s[j]);
    }
}

void Benchmark15::execute_cudagraph(int iter) {
}

void Benchmark15::execute_cudagraph_manual(int iter) {
}
void Benchmark15::execute_cudagraph_single(int iter) {
}
std::string
Benchmark15::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(y[0][0]);
    } else {
        std::string res = "[";
        for (int j = 0; j < M; j++) {
            res += std::to_string(y[j][0]) + ", ";
        }
        return res + ", ...]";
    }
}