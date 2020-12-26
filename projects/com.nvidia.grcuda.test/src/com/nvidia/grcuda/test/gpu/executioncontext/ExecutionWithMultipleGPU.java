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

            // Build kernel on device 0
            Value firstDevice_buildkernel = context.eval("grcuda", "buildkernel");
            Value firstDevice_squareKernel = firstDevice_buildkernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");
            assertNotNull(firstDevice_squareKernel);

            // init arrays with values
            for (int i = 0; i < numElements; ++i) {
                x.setArrayElement(i, 2.0);
                y.setArrayElement(i, 4.0);

            }

            Value firstconfiguredSquareKernel = firstDevice_squareKernel.execute(numBlocks, NUM_THREADS_PER_BLOCK);

            // Perform the computation;
            firstconfiguredSquareKernel.execute(x, numElements);
            firstconfiguredSquareKernel.execute(y, numElements);


            // Verify the output;
//            assertEquals(4.0, x.getArrayElement(0).asFloat(), 0.1);
//            assertEquals(16.0, y.getArrayElement(0).asFloat(), 0.1);
            context.eval("grcuda","cudaDeviceSynchronize");

            // Perform the computation;
            firstconfiguredSquareKernel.execute(x, numElements);
            firstconfiguredSquareKernel.execute(y, numElements);

//                    // Verify the output;
//            assertEquals(16.0, x.getArrayElement(0).asFloat(), 0.1);
//            assertEquals(256.0, y.getArrayElement(0).asFloat(), 0.1);

        }
    }


}