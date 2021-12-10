package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Value;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertEquals;

@RunWith(Theories.class)
public class BenchmarkB1 extends Benchmark {

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


    protected final String benchmarkName = "B1";

    private Value squareKernelFunction;
    private Value diffKernelFunction;
    private Value reduceKernelFunction;

    private Value x, x1, y, y1, res;



    @DataPoints
    public static int[] iterations() {
        return Benchmark.iterations();
    }

    @Override
    public void init() {
        // Context initialization
        Value buildkernel = this.getGrcudaContext().eval("grcuda", "buildkernel");

        // Kernel build
        squareKernelFunction = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
        diffKernelFunction = buildkernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
        reduceKernelFunction = buildkernel.execute(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");

        // Array initialization
        Value deviceArray = this.getGrcudaContext().eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", TEST_SIZE);
        x1 = deviceArray.execute("float", TEST_SIZE);
        y = deviceArray.execute("float", TEST_SIZE);
        y1 = deviceArray.execute("float", TEST_SIZE);
        res = deviceArray.execute("float", 1);

    }

    @Override
    public void resetIteration() {
        assert(!randomInit);
        for (int i = 0; i < TEST_SIZE; i++) {
            x.setArrayElement(i, 1.0f / (i + 1));
            y.setArrayElement(i, 2.0f / (i + 1));
        }
        res.setArrayElement(0, 0.0f);
    }

    @Override
    public void runTest(int iteration) {

        squareKernelFunction
                .execute(NUM_BLOCKS, NUM_THREADS) // Set parameters
                .execute(x, x1, TEST_SIZE); // Execute actual kernel

        squareKernelFunction
                .execute(NUM_BLOCKS, NUM_THREADS) // Set parameters
                .execute(y, y1, TEST_SIZE); // Execute actual kernel

        reduceKernelFunction
                .execute(NUM_BLOCKS, NUM_THREADS) // Set parameters
                .execute(x1, y1, res, TEST_SIZE); // Execute actual kernel

    }



    @Override
    protected void cpuValidation() {
        assert(!randomInit);

        float[] xHost = new float[TEST_SIZE];
        float[] yHost = new float[TEST_SIZE];
        float[] resHostTmp = new float[TEST_SIZE];
        for (int i = 0; i < TEST_SIZE; i++) {
            xHost[i] = 1.0f / (i + 1);
            yHost[i] = 2.0f / (i + 1);
            resHostTmp[i] = 0.0f;
        }

        for (int i = 0; i < TEST_SIZE; i++) {
            float xHostTmp = xHost[i] * xHost[i];
            float yHostTmp = yHost[i] * yHost[i];
            resHostTmp[i] = xHostTmp - yHostTmp;
        }

        float acc = 0.0f;

        for(int i = 0; i < TEST_SIZE; i++){
            acc += resHostTmp[i];
        }

        assertEquals(res.getArrayElement(0).asFloat(), acc, 1e-5);

    }

    @Override
    public String getBenchmarkName() {
        return benchmarkName;
    }

}
