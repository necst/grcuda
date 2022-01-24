package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Value;

import static org.junit.Assert.assertEquals;

public class B1 extends Benchmark {

    private static final String SQUARE_KERNEL =
            "extern \"C\" __global__ void square(float* x, float* y, int n) { \n" +
                    "for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
                    "y[i] = x[i] * x[i];\n" +
                    "}\n" +
                    "}\n";

    private static final String DIFF_KERNEL = "" +
            "extern \"C\" __global__ void diff(const float* x, const float* y, float* z, int n) {\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        z[i] = x[i] - y[i];\n" +
            "    }\n" +
            "}";

    private static final String REDUCE_KERNEL = "" +
            "// From https://devblogs.nvidia.com/faster-parallel-reductions-kepler/\n" +
            "\n" +
            "__inline__ __device__ float warp_reduce(float val) {\n" +
            "    int warp_size = 32;\n" +
            "    for (int offset = warp_size / 2; offset > 0; offset /= 2)\n" +
            "        val += __shfl_down_sync(0xFFFFFFFF, val, offset);\n" +
            "    return val;\n" +
            "}\n" +
            "\n" +
            "__global__ void reduce(float *x, float *y, float* z, int N) {\n" +
            "    int warp_size = 32;\n" +
            "    float sum = float(0);\n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {\n" +
            "        sum += x[i] - y[i];\n" +
            "    }\n" +
            "    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;\n" +
            "    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster\n" +
            "        atomicAdd(z, sum); // The first thread in the warp updates the output;\n" +
            "}";


    private static final String BENCHMARK_NAME = "B1";
    static {
        BenchmarkResults.setBenchmark(BENCHMARK_NAME);
    }

    private Value squareKernelFunction;
    private Value diffKernelFunction;
    private Value reduceKernelFunction;

    private Value x, x1, y, y1, res;

    @Override
    public void initializeTest(int iteration) {
        // Context initialization
        Value buildKernel = this.getContext().eval("grcuda", "buildkernel");

        // Kernel build
        squareKernelFunction = buildKernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
        diffKernelFunction = buildKernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
        reduceKernelFunction = buildKernel.execute(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");

        // Array initialization
        Value deviceArray = this.getContext().eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", getTestSize());
        x1 = deviceArray.execute("float", getTestSize());
        y = deviceArray.execute("float", getTestSize());
        y1 = deviceArray.execute("float", getTestSize());
        res = deviceArray.execute("float", 1);

    }

    @Override
    public void resetIteration(int iteration) {
        assert (!config.randomInit);
        for (int i = 0; i < getTestSize(); i++) {
            x.setArrayElement(i, 1.0f / (i + 1));
            y.setArrayElement(i, 2.0f / (i + 1));
        }
        res.setArrayElement(0, 0.0f);
    }

    @Override
    public void runTest(int iteration) {

        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x, x1, getTestSize()); // Execute actual kernel

        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(y, y1, getTestSize()); // Execute actual kernel

        reduceKernelFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x1, y1, res, getTestSize()); // Execute actual kernel

    }


    @Override
    protected void cpuValidation() {
        assert (!config.randomInit);

        float[] xHost = new float[getTestSize()];
        float[] yHost = new float[getTestSize()];
        float[] resHostTmp = new float[getTestSize()];
        for (int i = 0; i < getTestSize(); i++) {
            xHost[i] = 1.0f / (i + 1);
            yHost[i] = 2.0f / (i + 1);
            resHostTmp[i] = 0.0f;
        }

        for (int i = 0; i < getTestSize(); i++) {
            float xHostTmp = xHost[i] * xHost[i];
            float yHostTmp = yHost[i] * yHost[i];
            resHostTmp[i] = xHostTmp - yHostTmp;
        }

        float acc = 0.0f;

        for (int i = 0; i < getTestSize(); i++) {
            acc += resHostTmp[i];
        }

        assertEquals(res.getArrayElement(0).asFloat(), acc, 1e-5);

    }

}
