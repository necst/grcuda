package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
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

    private static Context grcudaContext;

    private static Value squareKernelFunction;
    private static Value diffKernelFunction;
    private static Value reduceKernelFunction;

    private static Value x, x1, y, y1, res;

    // The following variables should be read from a config file
    // For simplicity, I'm initializing them statically now
    private boolean randomInit = false;
    private static int NUM_BLOCKS = 8;
    private static int NUM_THREADS = 128;
    private boolean cpuValidate = false;
    private long executionTime;

    @DataPoints
    public static int[] iterations() {
        return Benchmark.iterations();
    }

    @BeforeClass
    public static void init() {
        // Context initialization
        grcudaContext = Benchmark.buildBenchmarkContext();
        Value buildkernel = grcudaContext.eval("grcuda", "buildkernel");

        // Kernel build
        squareKernelFunction = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
        diffKernelFunction = buildkernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
        reduceKernelFunction = buildkernel.execute(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");

        // Array initialization
        Value deviceArray = grcudaContext.eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", TEST_SIZE);
        x1 = deviceArray.execute("float", TEST_SIZE);
        y = deviceArray.execute("float", TEST_SIZE);
        y1 = deviceArray.execute("float", TEST_SIZE);
        res = deviceArray.execute("float", 1);

    }

    @Override
    @Before
    public void reset() {
        // Do not do random reset for now
        assert(!randomInit);
        for (int i = 0; i < TEST_SIZE; i++) {
            x.setArrayElement(i, 1.0f / (i + 1));
            y.setArrayElement(i, 2.0f / (i + 1));
        }
        res.setArrayElement(0, 0.0f);
    }

    @Theory
    public void run(int iteration) {
        long beginTime = System.nanoTime();

        squareKernelFunction
                .execute(NUM_BLOCKS, NUM_THREADS) // Set parameters
                .execute(x, x1, TEST_SIZE); // Execute actual kernel

        squareKernelFunction
                .execute(NUM_BLOCKS, NUM_THREADS) // Set parameters
                .execute(y, y1, TEST_SIZE); // Execute actual kernel

        reduceKernelFunction
                .execute(NUM_BLOCKS, NUM_THREADS) // Set parameters
                .execute(x1, y1, res, TEST_SIZE); // Execute actual kernel

        executionTime = System.nanoTime() - beginTime;

        if(cpuValidate) cpuValidation();

    }


    @Override
    protected void cpuValidation() {
        assert(!randomInit);

        float[] x_host = new float[TEST_SIZE];
        float[] y_host = new float[TEST_SIZE];
        float[] res_host_tmp = new float[TEST_SIZE];
        for (int i = 0; i < TEST_SIZE; i++) {
            x_host[i] = 1.0f / (i + 1);
            y_host[i] = 2.0f / (i + 1);
            res_host_tmp[i] = 0.0f;
        }

        for (int i = 0; i < TEST_SIZE; i++) {
            float x_host_tmp = x_host[i] * x_host[i];
            float y_host_tmp = y_host[i] * y_host[i];
            res_host_tmp[i] = x_host_tmp - y_host_tmp;
        }

        float acc = 0.0f;

        for(int i = 0; i < TEST_SIZE; i++){
            acc += res_host_tmp[i];
        }

        assertEquals(res.getArrayElement(0).asFloat(), acc, 1e-5);

    }

    @Override
    @After
    public void saveResults() {
        System.out.println("Benchmark 1 took " + this.executionTime + "ns");
    }
}
