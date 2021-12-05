package com.nvidia.grcuda.test.runtime.executioncontext;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.test.util.GrCUDATestOptionsStruct;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(Parameterized.class)
public class GrCUDAMultiGPUExecutionContextTest {

    // FIXME: add multi-gpu policies;

    /**
     * Tests are executed for each of the {@link AsyncGrCUDAExecutionContext} values;
     * @return the current stream policy
     */

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.getAllOptionCombinations();
    }

    private final GrCUDATestOptionsStruct options;

    public GrCUDAMultiGPUExecutionContextTest(GrCUDATestOptionsStruct options) {
        this.options = options;
    }

    private static final int NUM_THREADS_PER_BLOCK = 32;

    private static final String SQUARE_KERNEL =
            "extern \"C\" __global__ void square(float* x, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       x[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    private static final String SQUARE_2_KERNEL =
            "extern \"C\" __global__ void square(const float* x, float *y, int n) {\n" +
                    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (idx < n) {\n" +
                    "       y[idx] = x[idx] * x[idx];\n" +
                    "    }" +
                    "}\n";

    private static final String DIFF_KERNEL =
            "extern \"C\" __global__ void diff(float* x, float* y, float* z, int n) {\n" +
                    "   int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "   if (idx < n) {\n" +
                    "      z[idx] = x[idx] - y[idx];\n" +
                    "   }\n" +
                    "}";

    private static final String REDUCE_KERNEL =
            "extern \"C\" __global__ void reduce(float *x, float *res, int n) {\n" +
                    "    __shared__ float cache[" + NUM_THREADS_PER_BLOCK + "];\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < n) {\n" +
                    "       cache[threadIdx.x] = x[i];\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +
                    "    i = " + NUM_THREADS_PER_BLOCK + " / 2;\n" +
                    "    while (i > 0) {\n" +
                    "       if (threadIdx.x < i) {\n" +
                    "            cache[threadIdx.x] += cache[threadIdx.x + i];\n" +
                    "        }\n" +
                    "        __syncthreads();\n" +
                    "        i /= 2;\n" +
                    "    }\n" +
                    "    if (threadIdx.x == 0) {\n" +
                    "        atomicAdd(res, cache[0]);\n" +
                    "    }\n" +
                    "}";

    /**
     * Execute 2 independent kernels, 2 times in a row;
     */
    @Test
    public void dependency2KernelsSimpleTest() {

        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options, 2)) {

            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            assertNotNull(squareKernel);

            // init arrays with values
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);

            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredSquareKernel.execute(y, numElements);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);
            configuredSquareKernel.execute(y, numElements);

            // Verify the output;
            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    /**
     * Test with 3 kernels: kernel0 does not have dependencies.
     * kernel1 is the parent of kernek2;
     */
    @Test
    public void dependencyKernelsTestA() {

        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options, 2)) {

            final int numElements = 1000000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            assertNotNull(squareKernel);

            // init arrays with values
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);
            }

            Value configuredK0 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredK1 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredK2 = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredK0.execute(x, numElements);
            configuredK1.execute(y, numElements);

            // Perform the computation;
            configuredK2.execute(y, numElements);
            // Verify the output;
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    /**
     * Test a join pattern (x) & (y) -> (z), with data in x and y being copied from other arrays;
     */
    @Test
    public void dependencyPipelineWithArrayCopyTest() {

        try (Context context = GrCUDATestUtil.createContextFromOptions(this.options, 2)) {

            final int numElements = 100000;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);
            Value x2 = deviceArrayConstructor.execute("float", numElements);
            Value y2 = deviceArrayConstructor.execute("float", numElements);
            Value res = deviceArrayConstructor.execute("float", 1);
            Value res2 = deviceArrayConstructor.execute("float", 1);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 1.0 / (i + 1));
                y.setArrayElement(i, 2.0 / (i + 1));
            }
            res.setArrayElement(0, 0.0);

            x2.invokeMember("copyFrom", x, numElements);
            y2.invokeMember("copyFrom", y, numElements);

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            Value diffKernel = buildkernel.execute(DIFF_KERNEL, "diff", "const pointer, const pointer, pointer, sint32");
            Value reduceKernel = buildkernel.execute(REDUCE_KERNEL, "reduce", "const pointer, pointer, sint32");
            assertNotNull(squareKernel);
            assertNotNull(diffKernel);
            assertNotNull(reduceKernel);

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredDiffKernel = diffKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);
            Value configuredReduceKernel = reduceKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x2, numElements);
            configuredSquareKernel.execute(y2, numElements);
            configuredDiffKernel.execute(x2, y2, z, numElements);
            configuredReduceKernel.execute(z, res, numElements);

            res.invokeMember("copyTo", res2, 1);

            // Verify the output;
            float resScalar = res2.getArrayElement(0).asFloat();
            assertEquals(-4.93, resScalar, 0.01);
        }
    }
}