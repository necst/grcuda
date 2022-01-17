package com.nvidia.grcuda.benchmark;

public class B12 extends Benchmark {

    private static final String AXPB_XTENDED = "\n" +
            "extern \"C\" __global__ void axpb_xtended(const float alpha, const float *x, const float *b, const float beta, const float *c, float *out, const int N, const int offset_x, const int offset_c) {\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for (int i = init; i < N; i += stride) {\n" +
            "        out[i] = alpha * x[i + offset_x] + b[i] + beta * c[i + offset_c];\n" +
            "    }\n" +
            "}\n";
    private static final String NORMALIZE = "\n" +
            "extern \"C\" __global__ void normalize(const float *d_v_in, const double denominator, float *d_v_out, const int N) {\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for (int i = init; i < N; i += stride) {\n" +
            "        d_v_out[i] = d_v_in[i] * denominator;\n" +
            "    }\n" +
            "}";

    private static final String COPY_PARTITION_TO_VEC = "\n" +
            "extern \"C\" __global__ void copy_partition_to_vec(const float *vec_in, float *vec_out, const int N, const int offset_out, const int offset_in){\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for(int i = init; i < N; ++i){\n" +
            "        vec_out[i + offset_out] = float(vec_in[i + offset_in]);\n" +
            "    }\n" +
            "}";

    private static final String SQUARE = "\n" +
            "extern \"C\" __global__ void square(const float *x, float *y, int N){\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for(int i = init; i < N; i += stride){\n" +
            "        float value = x[i];\n" +
            "        y[i] = value * value;\n" +
            "    }\n" +
            "}";

    private static final String SPMV = "\n" +
            "extern \"C\" __global__ void spmv(const int *x, const int *y, const float *val, const float *v_in, float *v_out, int num_nnz) {\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for (int i = init; i < num_nnz; i += stride) {\n" +
            "        v_out[y[i]] += float(float(v_in[x[i]]) * float(val[i]));\n" +
            "    }\n" +
            "}";

    private static final String SUBTRACT = "\n" +
            "extern \"C\" __global__ void subtract(float* v1, const float* v2, const float alpha, int N, int offset) {" +
            "    int init = threadIdx.x + blockIdx.x * blockDim.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for(int i = init; i < N; i += stride){\n" +
            "        v1[i] -= alpha * v2[i + offset];\n" +
            "    }\n" +
            "}";

    private static final String DOT_PRODUCT = "\n" +
            "extern \"C\" __global__ void dot_product_stage_one(const float* v1, const float* v2, float* temporaryOutputValues, int N, int offset) {\n" +
            "    extern __shared__ float cache[];\n" +
            "    int threadId = threadIdx.x + blockIdx.x * blockDim.x;\n" +
            "    int cacheIdx = threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    float temp = float(0.0);\n" +
            "    for(int i = threadId; i < N; i += stride){\n" +
            "        temp += float(float(v1[i]) * float(v2[i + offset]));\n" +
            "    }\n" +
            "    cache[cacheIdx] = temp;\n" +
            "    __syncthreads();\n" +
            "    for(int i = blockDim.x >> 1; i != 0; i >>= 1){\n" +
            "        if(cacheIdx < i){\n" +
            "            cache[cacheIdx] += cache[cacheIdx + 1];\n" +
            "        }\n" +
            "        __syncthreads();\n" +
            "    }\n" +
            "    if (cacheIdx == 0){\n" +
            "        temporaryOutputValues[blockIdx.x] = cache[0];\n" +
            "    }\n" +
            "}\n";

    @Override
    public void initializeTest(int iteration) {

    }

    @Override
    public void resetIteration(int iteration) {

    }

    @Override
    public void runTest(int iteration) {

    }

    @Override
    protected void cpuValidation() {

    }
}
