package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Value;
import org.junit.experimental.theories.Theories;
import org.junit.runner.RunWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@RunWith(Theories.class)
public class B6 extends Benchmark {

    private static final String NB_KERNEL =

            "extern \"C\" __global__ void nb_1(const int* x, float* y, float* z, int size, int n_feat, int n_classes) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {\n" +
                    "            for (int j = 0; j < n_classes; j++) {\n" +
                    "                for (int q = 0; q < n_feat; q++) {\n" +
                    "                    z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];\n" +
                    "                } \n" +
                    "            }\n" +
                    "        }\n" +
                    "    }\n" +
                    "    \n" +
                    "    extern \"C\" __global__ void nb_2(float* x, float* y, int n_row_x, int n_col_x) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
                    "            float curr_max = x[i * n_col_x];\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                curr_max = fmaxf(curr_max, x[i * n_col_x + j]); \n" +
                    "            }\n" +
                    "            y[i] = curr_max;\n" +
                    "        }\n" +
                    "    }\n" +
                    "    \n" +
                    "    extern \"C\" __global__ void nb_3(float* x, float* y, float* z, int n_row_x, int n_col_x) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
                    "            float sum = 0;\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                sum += expf(x[i * n_col_x + j] - y[i]);\n" +
                    "            }\n" +
                    "            z[i] = logf(sum) + y[i];\n" +
                    "        }\n" +
                    "    }\n" +
                    "    \n" +
                    "    extern \"C\" __global__ void nb_4(float* x, float* y, int n_row_x, int n_col_x) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }";

    private static final String RR_KERNEL =

            "extern \"C\" __global__ void rr_1(const int* x, float *y, int n_row_x, int n_col_x) {\n" +
                    "        for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x; j += blockDim.x * gridDim.x) {\n" +
                    "            float feature_mean = 0;\n" +
                    "            float sum_sq = 0;\n" +
                    "            // Compute mean and variance;\n" +
                    "            for (int i = 0; i < n_row_x; i++) {\n" +
                    "                feature_mean += x[j * n_row_x + i];\n" +
                    "                sum_sq += x[j * n_row_x + i] * x[j * n_row_x + i];\n" +
                    "            }\n" +
                    "            feature_mean /= n_row_x;\n" +
                    "            float std = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);\n" +
                    "            \n" +
                    "            // Update values;\n" +
                    "            for (int i = 0; i < n_row_x; i++) {\n" +
                    "                y[j * n_row_x + i] = ((float) x[j * n_row_x + i] - feature_mean) / std;\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }\n" +
                    "    \n" +
                    "    extern \"C\" __global__ void rr_2(float* x, float* y, float* z, int size, int n_feat, int n_classes) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {\n" +
                    "            for (int j = 0; j < n_classes; j++) {\n" +
                    "                for (int q = 0; q < n_feat; q++) {\n" +
                    "                    z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];\n" +
                    "                }\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }\n" +
                    "\n" +
                    "    extern \"C\" __global__ void rr_3(float* x, float *y, int n_row_x, int n_col_x) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                x[i * n_col_x + j] += y[j];\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }";

    private static final String ENSEMBLE_KERNEL =
            "extern \"C\" __global__ void softmax(float *x, int n_row_x, int n_col_x) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
                    "            float row_exp_sum = 0;\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                row_exp_sum += expf( x[i * n_col_x + j]);\n" +
                    "            }\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                 x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;\n" +
                    "            }\n" +
                    "        }\n" +
                    "    }\n" +
                    "    \n" +
                    "    extern \"C\" __global__ void argmax(float *x, float *y, int *z, int n_row_x, int n_col_x) {\n" +
                    "        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x; i += blockDim.x * gridDim.x) {\n" +
                    "            int curr_best_index = 0;\n" +
                    "            float curr_best = x[i * n_col_x] + y[i * n_col_x];\n" +
                    "            for (int j = 0; j < n_col_x; j++) {\n" +
                    "                float curr = x[i * n_col_x + j] + y[i * n_col_x + j];\n" +
                    "                if (curr > curr_best) {\n" +
                    "                    curr_best = curr;\n" +
                    "                    curr_best_index = j;\n" +
                    "                }\n" +
                    "            }\n" +
                    "            z[i] = curr_best_index;\n" +
                    "        }\n" +
                    "    }";

    private static final String BENCHMARK_NAME = "B6";
    private Value nb1Function;
    private Value nb2Function;
    private Value nb3Function;
    private Value nb4Function;

    private Value rr1Function;
    private Value rr2Function;
    private Value rr3Function;

    private Value softMaxFunction;
    private Value argMaxFunction;

    private int[][] xCpu;
    private float[][] nbFeatLogProbCpu;
    private float[][] ridgeCoeffCpu;
    private float[] nbClassLogPriorCpu;
    private float[] ridgeInterceptCpu;
    private float[][] r1Cpu;
    private float[][] r2Cpu;

    private Value x, z, nbFeatLogProb, nbClassLogPrior, ridgeCoeff, ridgeIntercept, nbAMax, nbL, r1, r2, r;

    private int numClasses;
    private int numFeatures;

    @Override
    public void initializeTest(int iteration) {
        numClasses = 10;
        numFeatures = 200;
        int maxOccurrenceOfNgram = 10;

        // CPU Arrays
        createCPUArrays();

        // Kernels
        createKernels();

        // Fill cpu arrays with random values
        fillCPUArrays(maxOccurrenceOfNgram);

        // Array initialization
        createDeviceArrays();

        // Array copy
        copyToGPU();


    }

    private void createCPUArrays() {
        nbClassLogPriorCpu = new float[numClasses];
        nbFeatLogProbCpu = new float[numClasses][numFeatures];
        ridgeCoeffCpu = new float[numClasses][numFeatures];
        ridgeInterceptCpu = new float[numClasses];
        xCpu = new int[getTestSize()][numFeatures];
        r1Cpu = new float[getTestSize()][numClasses];
        r2Cpu = new float[getTestSize()][numClasses];
    }

    private void createKernels() {
        // Context initialization
        Value buildkernel = this.getContext().eval("grcuda", "buildkernel");

        //Kernel build
        nb1Function = buildkernel.execute(NB_KERNEL, "nb_1", "const pointer, pointer, pointer, sint32, sint32, sint32");
        nb2Function = buildkernel.execute(NB_KERNEL, "nb_2", "pointer, pointer, sint32, sint32");
        nb3Function = buildkernel.execute(NB_KERNEL, "nb_3", "pointer, pointer, pointer, sint32, sint32");
        nb4Function = buildkernel.execute(NB_KERNEL, "nb_4", "pointer, pointer, sint32, sint32");

        rr1Function = buildkernel.execute(RR_KERNEL, "rr_1", "const pointer, pointer, sint32, sint32");
        rr2Function = buildkernel.execute(RR_KERNEL, "rr_2", "pointer, pointer, pointer, sint32, sint32, sint32");
        rr3Function = buildkernel.execute(RR_KERNEL, "rr_3", "pointer, pointer, sint32, sint32");

        softMaxFunction = buildkernel.execute(ENSEMBLE_KERNEL, "softmax", "pointer, sint32, sint32");
        argMaxFunction = buildkernel.execute(ENSEMBLE_KERNEL, "argmax", "pointer, pointer, pointer, sint32, sint32");
    }

    private void fillCPUArrays(int maxOccurrenceOfNgram) {
        Random rng = new Random();

        for (int i = 0; i < getTestSize(); ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                xCpu[i][j] = rng.nextInt(maxOccurrenceOfNgram);
            }
        }

        for (int i = 0; i < getTestSize(); i++) {
            for (int j = 0; j < numClasses; j++) {
                r1Cpu[i][j] = nbClassLogPriorCpu[j];
            }
        }

        for (int i = 0; i < numClasses; i++)
            for (int j = 0; j < numFeatures; j++) {
                nbFeatLogProbCpu[i][j] = rng.nextFloat();
            }
        for (int i = 0; i < numClasses; i++)
            for (int j = 0; j < numFeatures; j++) {
                ridgeCoeffCpu[i][j] = rng.nextFloat();
            }

        for (int i = 0; i < numClasses; i++) {
            nbClassLogPriorCpu[i] = rng.nextFloat();
            ridgeInterceptCpu[i] = rng.nextFloat();
        }
    }

    private void createDeviceArrays() {
        Value deviceArray = this.getContext().eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", getTestSize(), numFeatures);
        z = deviceArray.execute("float", getTestSize(), numFeatures);
        nbFeatLogProb = deviceArray.execute("float", numClasses, numFeatures);
        nbClassLogPrior = deviceArray.execute("float", numClasses);
        ridgeCoeff = deviceArray.execute("float", numClasses, numFeatures);
        ridgeIntercept = deviceArray.execute("float", numClasses);
        nbAMax = deviceArray.execute("float", getTestSize());
        nbL = deviceArray.execute("float", getTestSize());
        r1 = deviceArray.execute("float", getTestSize(), numClasses);
        r2 = deviceArray.execute("float", getTestSize(), numClasses);
        r = deviceArray.execute("float", getTestSize());
    }

    private void copyToGPU() {
        for (int i = 0; i < getTestSize(); ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                x.getArrayElement(i).setArrayElement(j, xCpu[i][j]);
            }
        }

        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numFeatures; j++) {
                nbFeatLogProb.getArrayElement(i).setArrayElement(j, nbFeatLogProbCpu[i][j]);
                ridgeCoeff.getArrayElement(i).setArrayElement(j, ridgeCoeffCpu[i][j]);
            }
        }


        for (int i = 0; i < getTestSize(); i++) {
            for (int j = 0; j < numClasses; ++j) {
                r1.getArrayElement(i).setArrayElement(j, r1Cpu[i][j]);
                r2.getArrayElement(i).setArrayElement(j, r2Cpu[i][j]);
            }
        }

        nbClassLogPrior.invokeMember("copyFrom", nbClassLogPriorCpu);
        ridgeIntercept.invokeMember("copyFrom", nbClassLogPriorCpu);
    }

    @Override
    public void resetIteration(int iteration) {
        assert (!config.randomInit);
        for (int i = 0; i < getTestSize(); i++) {
            for (int j = 0; j < numClasses; j++) {
                r1.getArrayElement(i).setArrayElement(j, nbClassLogPrior.getArrayElement(j).asFloat());
                r2.getArrayElement(j).setArrayElement(j, 0.0f);
            }
        }
    }

    @Override
    public void runTest(int iteration) {

        //RR - 1
        rr1Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x, z, getTestSize(), numFeatures); // Execute actual kernel

        //NB - 1
        nb1Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x, nbFeatLogProb, r1, getTestSize(), numFeatures, numClasses); // Execute actual kernel

        //RR - 2
        rr2Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, ridgeCoeff, r2, getTestSize(), numFeatures, numClasses); // Execute actual kernel

        //NB - 2
        nb2Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, nbAMax, getTestSize(), numClasses); // Execute actual kernel

        //RR - 3
        rr3Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r2, ridgeIntercept, getTestSize(), numClasses); // Execute actual kernel

        //NB - 3
        nb3Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, nbAMax, nbL, getTestSize(), numClasses); // Execute actual kernel

        //NB - 4
        nb4Function
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, nbL, getTestSize(), numClasses); // Execute actual kernel

        //Ensemble results;

        //Softmax normalization;
        softMaxFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, getTestSize(), numClasses); // Execute actual kernel
        softMaxFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r2, getTestSize(), numClasses); // Execute actual kernel

        //Prediction;
        argMaxFunction
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, r2, r, getTestSize(), numClasses); // Execute actual kernel

    }

    private Float[] exp(Float[] x) {
        Float[] expX = new Float[x.length];
        for (int i = 0; i < x.length; ++i) {
            expX[i] = (float) Math.exp(x[i]);

        }
        return expX;
    }

    private float sum(Float[] x) {
        float acc = 0.0f;
        for (float el : x)
            acc += el;
        return acc;
    }

    private Float[] softmax(Float[] x) {
        Float[] expX = exp(x);
        Float sumExpX = sum(expX);


        for (int i = 0; i < x.length; ++i) {
            expX[i] /= sumExpX;
        }

        return expX;

    }


    private float[] log(float[] x) {
        float[] logX = new float[x.length];
        for (int i = 0; i < x.length; ++i) {
            logX[i] = (float) Math.log(x[i]);
        }
        return logX;
    }

    private float log(float x) {
        return (float) Math.log(x);
    }

    private <T, V> float dot(T[] x, V[] y) {
        float acc = 0.0f;
        for (int i = 0; i < x.length; ++i)
            acc += ((float) x[i]) * ((float) y[i]);

        return acc;
    }

    private <T, V> Float[][] matmul(T[][] x, V[][] y) {
        Float[][] res = new Float[x.length][y[0].length];
        for (int i = 0; i < res.length; i++) {
            for (int j = 0; j < res[i].length; j++) {
                res[i][j] = dot(x[i], y[j]);
            }
        }
        return res;
    }

    private <T> Float[][] transpose(T[][] x) {
        Float[][] transposed = new Float[x.length][x[0].length];
        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[i].length; ++j) {
                transposed[j][i] = (Float) x[i][j];
            }
        }
        return transposed;
    }

    private float logSumExp(Float[] x) {
        return log(sum(exp(x)));
    }

    private float logSumExp(Float[][] x) {
        return log(sum(exp(x)));
    }

    private Float[][] exp(Float[][] x) {
        Float[][] ret = new Float[x.length][x[0].length];
        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[i].length; ++j) {
                ret[i][j] = (float) Math.exp(x[i][j]);
            }
        }
        return ret;
    }

    private float sum(Float[][] x) {
        float ret = 0.0f;
        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[i].length; ++j) {
                ret += x[i][j];
            }
        }
        return ret;
    }

    private Float[] amax(Float[][] x) {
        Float[] res = new Float[x[0].length];
        for (int i = 0; i < x[0].length; ++i) {
            res[i] = -Float.MAX_VALUE;
        }

        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[0].length; ++j) {
                if (res[j] < x[i][j])
                    res[j] = x[i][j];
            }
        }

        return res;
    }

    private Float[] naive_bayes_predict(Integer[][] x, Float[][] featureLogProb, Float[] logClassPrior) {
        Float[] ret = new Float[x[0].length];
        Float[][] tmp = matmul(x, transpose(featureLogProb));
        Float[][] jll = new Float[tmp.length][tmp[0].length];
        for (int i = 0; i < tmp.length; ++i) {
            for (int j = 0; j < logClassPrior.length; ++j) {
                jll[i][j] = tmp[i][j] + logClassPrior[j];
            }
        }
        Float[] amax = amax(jll);
        Float[][] l = new Float[jll.length][jll[0].length];
        for (int i = 0; i < jll.length; ++i) {
            for (int j = 0; j < jll[i].length; ++j) {
                l[i][j] = jll[i][j] - amax[j];
            }
        }

        Float tmp2 = logSumExp(l);
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = tmp2 + amax[i];
        }


        return ret;
    }

    private int[][] normalize(int[][] x) {
        int[][] ret = new int[x.length][x[0].length];

        // compute mean of each column
        float[] mean = new float[x.length];
        float squared = 0.0f;

        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < x[i].length; ++j) {
                mean[i] += x[i][j];
            }
        }

        for (int i = 0; i < x.length; ++i) {
            mean[i] /= mean.length;
        }

        // subtract the mean from every column
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                ret[i][j] = (int) (x[i][j] - mean[i]);
                squared += ret[i][j] * ret[i][j];
            }
        }

        float std = (float) Math.sqrt(squared);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                ret[i][j] = (int)((float) ret[i][j] / std);
            }
        }


        return ret;
    }


    @Override
    protected void cpuValidation() {
        // TODO: do ridge_predict and this
    }
}
