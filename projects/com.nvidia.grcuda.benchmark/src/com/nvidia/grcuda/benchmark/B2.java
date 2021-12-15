package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Value;

import static org.junit.Assert.assertEquals;

public class B2 extends Benchmark {

    private static final String SQUARE_KERNEL = "" +
            "extern \"C\" __global__ void square(float* x, int n) {\n" +
            "        int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "        if (idx < n) {\n" +
            "            x[idx] = x[idx] * x[idx];\n" +
            "        }\n" +
            "    }";

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
            "        sum += x[i] + y[i];\n" +
            "    }\n" +
            "    sum = warp_reduce(sum); // Obtain the sum of values in the current warp;\n" +
            "    if ((threadIdx.x & (warp_size - 1)) == 0) // Same as (threadIdx.x % warp_size) == 0 but faster\n" +
            "        atomicAdd(z, sum); // The first thread in the warp updates the output;\n" +
            "}";


    private static final String ADD_TWO_KERNEL = "" +
            " extern \"C\" __global__ void addtwo(float* b, float* a, int n) {\n" +
            "        int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "        if (idx < n) {\n" +
            "            b[idx] = a[idx] + 2.0;\n" +
            "        }\n" +
            "    }";

    private Value squareKernelFunction;
    private Value diffKernelFunction;
    private Value reduceKernelFunction;
    private Value addTwoKernelFunction;

    private Value x, y, a, a1, z, res;


    @Override
    public void initializeTest(int iteration) {
        Value buildKernel = this.getContext().eval("grcuda", "buildkernel");
        // Kernel build
        squareKernelFunction = buildKernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
        diffKernelFunction = buildKernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
        reduceKernelFunction = buildKernel.execute(REDUCE_KERNEL, "reduce", "const pointer, const pointer, pointer, sint32");
        addTwoKernelFunction = buildKernel.execute(ADD_TWO_KERNEL, "addtwo", "pointer, const pointer, sint32");

        // Array initialization
        Value deviceArray = this.getContext().eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", getTestSize());
        y = deviceArray.execute("float", getTestSize());
        a = deviceArray.execute("float", getTestSize());
        a1 = deviceArray.execute("float", getTestSize());
        z = deviceArray.execute("float", getTestSize());

        res = deviceArray.execute("float", 1);
    }

    @Override
    public void resetIteration(int iteration) {
        assert (!config.randomInit);
        for (int i = 0; i < getTestSize(); i++) {
            x.setArrayElement(i, 1.0f / (i + 1));
            y.setArrayElement(i, 2.0f / (i + 1));
            a.setArrayElement(i, 4.0f / (i + 1));
        }
        res.setArrayElement(0, 0.0f);
    }

    @Override
    public void runTest(int iteration) {
        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock)
                .execute(x, getTestSize());

        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock)
                .execute(y, getTestSize());

        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock)
                .execute(a, getTestSize());

        diffKernelFunction
                .execute(config.blocks, config.threadsPerBlock)
                .execute(x, y, z, getTestSize());

        addTwoKernelFunction
                .execute(config.blocks, config.threadsPerBlock)
                .execute(a1, a, getTestSize());

        reduceKernelFunction
                .execute(config.blocks, config.threadsPerBlock)
                .execute(a1, z, res, getTestSize());
    }

    @Override
    protected void cpuValidation() {
        assert (!config.randomInit);

        float[] xHost = new float[getTestSize()];
        float[] yHost = new float[getTestSize()];
        float[] aHost = new float[getTestSize()];
        float[] resHostTmpA = new float[getTestSize()];
        float[] resHostTmpB = new float[getTestSize()];


        for (int i = 0; i < getTestSize(); i++) {
            xHost[i] = 1.0f / (i + 1);
            yHost[i] = 2.0f / (i + 1);
            aHost[i] = 4.0f / (i + 1);
            resHostTmpA[i] = 0.0f;
            resHostTmpB[i] = 0.0f;
        }

        for (int i = 0; i < getTestSize(); ++i) {
            xHost[i] = xHost[i] * xHost[i];
            yHost[i] = yHost[i] * yHost[i];
            aHost[i] = aHost[i] * aHost[i];
        }

        for (int i = 0; i < getTestSize(); ++i) {
            resHostTmpA[i] = xHost[i] - yHost[i];
            resHostTmpB[i] = aHost[i] + 2.0f;
        }

        float acc = 0.0f;
        for (int i = 0; i < getTestSize(); ++i) {
            acc += (resHostTmpA[i] + resHostTmpB[i]);
        }

        assertEquals(acc, res.getArrayElement(0).asFloat(), 1e-1);
    }
}
