package com.nvidia.grcuda.test.gpu.executioncontext;
//import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.gpu.ComplexExecutionDAGTest;
import com.oracle.truffle.api.TruffleLanguage;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(Parameterized.class)
public class ExecutionWithMultipleGPU {

    /**
     * Tests are executed for each of the {@link com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext} values;
     * @return the current stream policy
     */
//    @Parameterized.Parameters
//    public static Collection<Object[]> data() {
//        return Arrays.asList(new Object[][]{
//                {"sync"},
//                {"default"}
//        });
//    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return ComplexExecutionDAGTest.crossProduct(Arrays.asList(new Object[][]{
                {"default"},
                {false},
                {"multi_disjoint"}
        }));
    }

    private final String policy;
    private final boolean inputPrefetch;
    private final String streamPolicy;
    public ExecutionWithMultipleGPU(String policy, boolean inputPrefetch, String streamPolicy) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.streamPolicy = streamPolicy;
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

    @Test
    public void dependencyKernelSimpleTest() {

        try (Context context = Context.newBuilder().option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {
            final int numElements = 10;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");

            assertNotNull(squareKernel);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, numElements);

            // Verify the output;
            assertEquals(4.0, x.getArrayElement(1).asFloat(), 0.1);
        }
    }

    @Test
    public void dependency2KernelsSimpleTest() {

        Map<String,String> policy = new HashMap<>();
        policy.put("grcuda.RetrieveParentStreamPolicy", String.valueOf(this.streamPolicy));
        policy.put("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch));
        try (Context context = Context.newBuilder().option("grcuda.ExecutionPolicy", this.policy).options(policy).allowAllAccess(true).build()) {


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

//          // Verify the output;
            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);

        }
    }

    /**
     * Test with 3 kernels in multi-GPUs system, K0 without dependency neither backward nor forward. K1 is the parent of K2 with dependency
     * */
    @Test
    public void dependencyKernelsTestA() {

        Map<String,String> policy = new HashMap<>();
        policy.put("grcuda.RetrieveParentStreamPolicy", String.valueOf(this.streamPolicy));
        policy.put("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch));
        try (Context context = Context.newBuilder().option("grcuda.ExecutionPolicy", this.policy).options(policy).allowAllAccess(true).build()) {


            final int numElements = 100000000;
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

//          // Verify the output;
            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);

        }
    }


    /**
     * The read on "y" has to sync on the stream where the kernel is running, although that kernel doesn't use "y".
     * This is due to the pre-Pascal limitations on managed memory accesses,
     * and the inability to access an array while it is being used by a running kernel;
     */
    @Test
    public void dependencyPipelineSimple3Test() {

        try (Context context = Context.newBuilder().option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {

            final int numElements = 100;
            final int numBlocks = (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
            Value deviceArrayConstructor = context.eval("grcuda", "DeviceArray");
            Value x = deviceArrayConstructor.execute("float", numElements);
            Value y = deviceArrayConstructor.execute("float", numElements);
            Value z = deviceArrayConstructor.execute("float", numElements);

            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
            }

            Value buildkernel = context.eval("grcuda", "buildkernel");
            Value squareKernel = buildkernel.execute(SQUARE_2_KERNEL, "square", "const pointer, pointer, sint32");
            assertNotNull(squareKernel);

            Value configuredSquareKernel = squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            configuredSquareKernel.execute(x, z, numElements);

            // Verify the output;
            assertEquals(4.0, z.getArrayElement(0).asFloat(), 0.1);
            assertEquals(0.0, y.getArrayElement(0).asFloat(), 0.1);
        }
    }

    @Test
    public void dependencyPipelineWithArrayCopyTest() {

        try (Context context = Context.newBuilder().option("grcuda.ExecutionPolicy", this.policy)
                .option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(true).build()) {

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