package com.nvidia.grcuda.benchmark;

import com.sun.jdi.Value;
import org.graalvm.polyglot.Value;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;            //TODO: verify that they are actually useful
import org.junit.runner.RunWith;

import static org.junit.Assert.assertEquals;

@RunWith(Theories.class)
public class BenchmarkB6 extends Benchmark{

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

    protected final String benchmarkName = "B6";

    private Value nb1;
    private Value nb2;
    private Value nb3;
    private Value nb4;

    private Value rr1;
    private Value rr2;
    private Value rr3;

    private Value softMax;
    private Value argMax;

    private int xCpu;
    private float[][] nbFeatLogProbCpu;
    private float[][] ridgeCoeffCpu;
    private float[] nbClassLogPriorCpu;
    private float[] ridgeInterceptCpu;
    private Value r1Cpu;
    private Value r2Cpu;

    private Value x, z, nbFeatLogProb, nbClassLogPrior, ridgeCoeff, ridgeIntercept, nbAMax, nbL, r1, r2, r;

    private int numClasses;
    private int numFeatures;

    @DataPoints
    public static int[] iterations() {
        return Benchmark.iterations();
    }

    @Override
    public void init() {

        numClasses = 10;
        numFeatures = 200;
        nbClassLogPriorCpu = new float[numClasses];
        nbFeatLogProbCpu = new float[numClasses][numFeatures];
        ridgeCoeffCpu = new float[numClasses][numFeatures];
        ridgeInterceptCpu = new float[numClasses];

        // Context initialization
        Value buildkernel = this.getGrcudaContext().eval("grcuda", "buildkernel");

        //Kernel build
        nb1 = buildkernel.execute(NB_KERNEL, "nb1", "const pointer, pointer, pointer, sint32, sint32, sint32");
        nb2 = buildkernel.execute(NB_KERNEL, "nb2", "pointer, pointer, sint32, sint32");
        nb3 = buildkernel.execute(NB_KERNEL, "nb3", "pointer, pointer, pointer, sint32, sint32");
        nb4 = buildkernel.execute(NB_KERNEL, "nb4", "pointer, pointer, sint32, sint32");

        rr1 = buildkernel.execute(RR_KERNEL, "rr1", "const pointer, pointer, sint32, sint32");
        rr2 = buildkernel.execute(RR_KERNEL, "rr2", "pointer, pointer, pointer, sint32, sint32, sint32");
        rr3 = buildkernel.execute(RR_KERNEL, "rr3", "pointer, pointer, sint32, sint32");

        softMax = buildkernel.execute(ENSEMBLE_KERNEL, "softmax", "pointer, sint32, sint32");
        argMax = buildkernel.execute(ENSEMBLE_KERNEL, "argmax", "pointer, pointer, pointer, sint32, sint32");

        // Create a random input
        int maxOccurrenceOfNgram = 10;
        int minOccurrenceOfNgram = 0;
        double min = 0.0;
        double max = 1.0;

        xCpu = (int)(Math.random()*(maxOccurrenceOfNgram-minOccurrenceOfNgram+1)+minOccurrenceOfNgram);

        for (int i = 0;i < numClasses; i++)
            for (int j = 0;j < numFeatures; j++){
                nbFeatLogProbCpu[i][j] = (float) Math.random();
            }
        for (int i = 0;i < numClasses; i++)
            for (int j = 0;j < numFeatures; j++){
                ridgeCoeffCpu[i][j] = (float) Math.random();
            }

        for (int i = 0; i < numClasses; i++) {
            nbClassLogPriorCpu[i] = (float) Math.random();
        }

        for (int i = 0; i < numClasses; i++) {
            ridgeInterceptCpu[i] = (float) Math.random();
        }

        float[][] r1Cpu = new float[getTestSize()][numClasses];
        for (int i = 0; i < getTestSize(); i++)
            for (int j = 0; j < numClasses; j++) {
                r1Cpu[i][j] = nbClassLogPriorCpu[j];
            }

        int[][] r2Cpu = new int[getTestSize()][numClasses];

        // Array initialization
        Value deviceArray = this.getGrcudaContext().eval("grcuda", "DeviceArray");
        x = deviceArray.execute("float", (getTestSize() * numFeatures));
        z = deviceArray.execute("float", (getTestSize() * numFeatures));

        nbFeatLogProb = deviceArray.execute("float", numClasses * numFeatures);
        nbClassLogPrior = deviceArray.execute("float", numClasses);
        ridgeCoeff = deviceArray.execute("float", numClasses * numFeatures);
        ridgeIntercept = deviceArray.execute("float", numClasses);

        nbAMax = deviceArray.execute("float",getTestSize());
        nbL = deviceArray.execute("float",getTestSize());

        r1 = deviceArray.execute("float",getTestSize() * numClasses);
        r2 = deviceArray.execute("float",getTestSize() * numClasses);
        r = deviceArray.execute("float",getTestSize());
    }

    @Override
    public void resetIteration(int iteration) {
        assert (!config.randomInit);
        for (int i = 0; i < getTestSize(); i++) {
            for (int j = 0; j < numClasses; j++) {
                r1.setArrayElement(i * numClasses + j, nbClassLogPrior.getArrayElement(j).asFloat());                r2.setArrayElement(i * numClasses +j, 0);
            }
        }
    }

    @Override
    public void runTest(int iteration) {

        //RR - 1
        rr1
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x, z, getTestSize(), numFeatures); // Execute actual kernel

        //NB - 1
        nb1
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(x, nbFeatLogProb, r1, getTestSize(), numFeatures, numClasses); // Execute actual kernel

        //RR - 2
        rr2
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, ridgeCoeff, r2, getTestSize(), numFeatures, numClasses); // Execute actual kernel

        //NB - 2
        nb2
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, nbAMax, getTestSize(), numClasses); // Execute actual kernel

        //RR - 3
        rr3
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r2, ridgeIntercept, getTestSize(), numClasses); // Execute actual kernel

        //NB - 3
        nb3
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, nbAMax, nbL, getTestSize(), numClasses); // Execute actual kernel

        //NB - 4
        nb4
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, nbL, getTestSize(), numClasses); // Execute actual kernel

        //Ensemble results;

        //Softmax normalization;
        softMax
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, getTestSize(), numClasses); // Execute actual kernel
        softMax
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r2, getTestSize(), numClasses); // Execute actual kernel

        //Prediction;
        argMax
                .execute(config.blocks, config.threadsPerBlock) // Set parameters
                .execute(r1, r2, r, getTestSize(), numClasses); // Execute actual kernel

    }
}
