#define NUM_THREADS 128
#define DANGLING 0.85

extern "C" __global__ void pr(int *ptr, int *idx, float *pr, float *pr_old, int *outdegrees, int N, int E) {

    int v = blockIdx.x * NUM_THREADS + threadIdx.x;
    float sum = 0;
    for (int i = ptr[v]; i < ptr[v + 1]; i++) {
        sum += pr_old[idx[i]] / outdegrees[v];
    }
    pr[v] = (1 - DANGLING) / N + DANGLING * sum;
}
