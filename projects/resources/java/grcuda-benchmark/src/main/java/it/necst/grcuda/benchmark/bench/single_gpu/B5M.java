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

import it.necst.grcuda.benchmark.Benchmark;
import it.necst.grcuda.benchmark.BenchmarkConfig;
import org.graalvm.polyglot.Value;

import static org.junit.Assert.assertEquals;

public class B5M extends Benchmark {
    /*
     *  Black & Scholes equation benchmark, executed concurrently on different input vectors;
     */

    private static final String BS_KERNEL = "" + "__device__ inline double cndGPU(double d) {\n" + "    const double       A1 = 0.31938153f;\n" + "    const double       A2 = -0.356563782f;\n" + "    const double       A3 = 1.781477937f;\n" + "    const double       A4 = -1.821255978f;\n" + "    const double       A5 = 1.330274429f;\n" + "    const double RSQRT2PI = 0.39894228040143267793994605993438f;\n" + "\n" + "    double\n" + "    K = 1.0 / (1.0 + 0.2316419 * fabs(d));\n" + "\n" + "    double\n" + "    cnd = RSQRT2PI * exp(- 0.5f * d * d) *\n" + "          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));\n" + "\n" + "    if (d > 0)\n" + "        cnd = 1.0 - cnd;\n" + "\n" + "    return cnd;\n" + "}\n" + "\n" + "extern \"C\" __global__ void bs(const double *x, double *y, int N, double R, double V, double T, double K) {\n" + "\n" + "    double sqrtT = 1.0 / rsqrt(T);\n" + "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {\n" + "        double expRT;\n" + "        double d1, d2, CNDD1, CNDD2;\n" + "        d1 = (log(x[i] / K) + (R + 0.5 * V * V) * T) / (V * sqrtT);\n" + "        d2 = d1 - V * sqrtT;\n" + "\n" + "        CNDD1 = cndGPU(d1);\n" + "        CNDD2 = cndGPU(d2);\n" + "\n" + "        //Calculate Call and Put simultaneously\n" + "        expRT = exp(-R * T);\n" + "        y[i] = x[i] * CNDD1 - K * expRT * CNDD2;\n" + "    }\n" + "}";

    private Value bs_kernelFunction;
    private final Value[] x, y;
    private float[] x_tmp;
    private static final float R = 0.08f;
    private static final float V = 0.3f;
    private static final float T = 1.0f;
    private static final float K1 = 60.0f;
    private final int K2;

    public B5M(BenchmarkConfig currentConfig) {
        super(currentConfig);

        this.K2 = 24;
        this.x = new Value[this.K2];
        this.x_tmp = null;
        this.y = new Value[this.K2];

        this.bs_kernelFunction = null;
    }

    @Override
    public void initializeTest(int iteration) {
        // Context initialization
        Value buildKernel = context.eval("grcuda", "buildkernel");

        // Alloc vectors
        Value deviceArray = context.eval("grcuda", "DeviceArray");
        for (int i = 0; i < this.K2; i++) {
            this.x[i] = deviceArray.execute("float", this.config.size);
            this.y[i] = deviceArray.execute("float", this.config.size);
        }

        // Build the kernels
        bs_kernelFunction = buildKernel.execute(BS_KERNEL, "bs", "const pointer, pointer, sint32, double, double, double, double");
    }

    @Override
    public void resetIteration(int iteration) {
        // Initialization
        this.x_tmp = new float[this.config.size];

        assert (!config.randomInit);
        for (int i = 0; i < this.config.size; i++) this.x_tmp[i] = K1;

        // Reset result
        for (int i = 0; i < this.K2; i++) {
            for (int j = 0; j < this.config.size; j++) {
                this.x[i].setArrayElement(j, this.x_tmp[j]);
            }
        }
    }

    @Override
    public void runTest(int iteration) {
        System.out.println("    INSIDE runTest() - " + iteration);

        for (int i = 0; i < this.K2; i++) {
            bs_kernelFunction.execute(config.numBlocks, config.blockSize1D) // Set parameters
                    .execute(this.x[i], this.y[i], this.config.size, R, V, T, K1); // Execute actual kernel
        }

        Value[] result = new Value[this.K2];
        for (int i = 0; i < this.K2; i++) result[i] = this.y[i].getArrayElement(0);

        benchmarkResults.gpu_result = result[0].asFloat();
    }

    @Override
    public void cpuValidation() {
        float[] res;
        res = BS(this.x_tmp, R, V, T, K1);

        assertEquals(benchmarkResults.gpu_result, res[0], 1e-3);
    }

    private float[] CND(float[] X) {
        /*
         *  Cumulative normal distribution.
         *  Helper function used by BS(...).
         */

        float a1 = 0.31938153f;
        float a2 = -0.356563782f;
        float a3 = 1.781477937f;
        float a4 = -1.821255978f;
        float a5 = 1.330274429f;
        float[] L = new float[X.length];
        float[] K = new float[X.length];
        float[] w = new float[X.length];

        for (int i = 0; i < X.length; i++) L[i] = Math.abs(X[i]);
        for (int i = 0; i < X.length; i++) K[i] = (float) ((1.0f) / (1.0 + 0.2316419 * L[i]));
        for (int i = 0; i < X.length; i++)
            w[i] = (float) (1.0 - 1.0 / Math.sqrt(2 * Math.PI) * Math.exp(-L[i] * L[i] / 2.) * (a1 * K[i] + a2 * (Math.pow(K[i], 2)) + a3 * (Math.pow(K[i], 3)) + a4 * (Math.pow(K[i], 4)) + a5 * (Math.pow(K[i], 5))));
        for (int i = 0; i < X.length; i++) w[i] = (w[i] < 0) ? (float) (1.0 - w[i]) : w[i];

        return w;
    }

    private float[] BS(float[] X, float R, float V, float T, float K) {
        /*
         *  Black Scholes Function.
         */
        float[] d1_arr = new float[X.length];
        float[] d2_arr = new float[X.length];
        float[] result = new float[X.length];
        float[] w_arr;
        float[] w2_arr;

        for (int i = 0; i < X.length; i++)
            d1_arr[i] = (float) ((Math.log(X[i] / K) + (R + V * V / 2.) * T) / (V * Math.sqrt(T)));
        for (int i = 0; i < X.length; i++) d2_arr[i] = (float) (d1_arr[i] - V * Math.sqrt(T));
        w_arr = CND(d1_arr);
        w2_arr = CND(d2_arr);

        for (int i = 0; i < X.length; i++) result[i] = (float) (X[i] * w_arr[i] - X[i] * Math.exp(-R * T) * w2_arr[i]);
        return result;
    }
}