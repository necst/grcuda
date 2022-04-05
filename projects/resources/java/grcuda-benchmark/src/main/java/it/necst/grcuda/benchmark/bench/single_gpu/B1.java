/*
 * Copyright (c) 2022 NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package it.necst.grcuda.benchmark.bench.single_gpu;

import static org.junit.Assert.assertEquals;

import org.graalvm.polyglot.Value;

import it.necst.grcuda.benchmark.Benchmark;
import it.necst.grcuda.benchmark.BenchmarkConfig;



public class B1 extends Benchmark {

    private static final String SQUARE_KERNEL =
            "extern \"C\" __global__ void square(float* x, float* y, int n) { \n" +
                    "for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
                    "y[i] = x[i] * x[i];\n" +
                    "}\n" +
                    "}\n";

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

    public B1(BenchmarkConfig config){
        super(config);
    }

    private Value squareKernelFunction;
    private Value reduceKernelFunction;
    private Value x, x1, y, y1, res;

    @Override
    public void initializeTest(int iteration) {
        // Context initialization
        Value buildKernel = context.eval("grcuda", "buildkernel");

        // Kernel build
        squareKernelFunction = buildKernel.execute(SQUARE_KERNEL, "square", "pointer, pointer, sint32");
        reduceKernelFunction = buildKernel.execute(REDUCE_KERNEL, "reduce", "pointer, pointer, pointer, sint32");

        // Array initialization
        Value deviceArray = context.eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", config.testSize);
        x1 = deviceArray.execute("float", config.testSize);
        y = deviceArray.execute("float", config.testSize);
        y1 = deviceArray.execute("float", config.testSize);
        res = deviceArray.execute("float", 1);

    }

    @Override
    public void resetIteration(int iteration) {
        assert (!config.randomInit);
        for (int i = 0; i < config.testSize; i++) {
            x.setArrayElement(i, 1.0f / (i + 1));
            y.setArrayElement(i, 2.0f / (i + 1));
        }
        res.setArrayElement(0, 0.0f);
    }

    @Override
    public void runTest(int iteration) {

        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x, x1, config.testSize); // Execute actual kernel

        squareKernelFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(y, y1, config.testSize); // Execute actual kernel

        reduceKernelFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x1, y1, res, config.testSize); // Execute actual kernel

        // sync step to measure the real computation time
        benchmarkResults.gpu_result = res.getArrayElement(0).asFloat();
    }

    @Override
    public void cpuValidation() {
        assert (!config.randomInit);

        float[] xHost = new float[config.testSize];
        float[] yHost = new float[config.testSize];
        float[] resHostTmp = new float[config.testSize];
        for (int i = 0; i < config.testSize; i++) {
            xHost[i] = 1.0f / (i + 1);
            yHost[i] = 2.0f / (i + 1);
            resHostTmp[i] = 0.0f;
        }

        for (int i = 0; i < config.testSize; i++) {
            float xHostTmp = xHost[i] * xHost[i];
            float yHostTmp = yHost[i] * yHost[i];
            resHostTmp[i] = xHostTmp - yHostTmp;
        }

        float acc = 0.0f;

        for (int i = 0; i < config.testSize; i++) {
            acc += resHostTmp[i];
        }

        assertEquals(benchmarkResults.gpu_result, acc, 1e-5);
    }

}