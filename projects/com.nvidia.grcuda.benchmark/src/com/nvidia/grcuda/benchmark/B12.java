package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Value;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class B12 extends Benchmark {

    private static final int NUM_PARTITIONS = 4;
    private static final int NUM_EIGEN = 8;

    private float[][] vecIn;
    private float[][] vecOut;
    private COOMatrix matrix;
    private COOMatrix[] partitions;
    private float alpha = 0.0f;
    private float beta = 0.0f;

    // Device Vectors
    private Value[] deviceVecIn = new Value[NUM_PARTITIONS];
    private Value[] deviceVecOut = new Value[NUM_PARTITIONS];
    private Value[] deviceIntermediateDotProductValues = new Value[NUM_PARTITIONS];
    private Value[] alphaIntermediate = new Value[NUM_PARTITIONS];
    private Value[] betaIntermediate = new Value[NUM_PARTITIONS];
    private Value[] deviceVecOutSpmv = new Value[NUM_PARTITIONS];
    private Value[] deviceCooX = new Value[NUM_PARTITIONS];
    private Value[] deviceCooY = new Value[NUM_PARTITIONS];
    private Value[] deviceCooVal = new Value[NUM_PARTITIONS];
    private Value[] deviceVecNext = new Value[NUM_PARTITIONS];
    private Value[] deviceNormalizedOut = new Value[NUM_PARTITIONS];
    private Value[] deviceLanczosVectors = new Value[NUM_PARTITIONS];

    // Kernels
    private Value spmv;
    private Value axpbXtended;
    private Value normalize;
    private Value subtract;
    private Value copyPartitionToVec;


    // TODO: i'm missing DOT_PRODUCT AND L2 NORM

    private static final String AXPB_XTENDED = "\n" +
            "extern \"C\" __global__ void axpb_xtended(const float alpha, const float *x, const float *b, const float beta, const float *c, float *out, const int N, const int offset_x, const int offset_c) {\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for (int i = init; i < N; i += stride) {\n" +
            "        out[i] = alpha * x[i + offset_x] + b[i] + beta * c[i + offset_c];\n" +
            "    }\n" +
            "}\n";

    private static final String NORMALIZE = "\n" +
            "extern \"C\" __global__ void normalize(const float *d_v_in, const double denominator, float *d_v_out, const int N) {\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for (int i = init; i < N; i += stride) {\n" +
            "        d_v_out[i] = d_v_in[i] * denominator;\n" +
            "    }\n" +
            "}";

    private static final String COPY_PARTITION_TO_VEC = "\n" +
            "extern \"C\" __global__ void copy_partition_to_vec(const float *vec_in, float *vec_out, const int N, const int offset_out, const int offset_in){\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for(int i = init; i < N; ++i){\n" +
            "        vec_out[i + offset_out] = float(vec_in[i + offset_in]);\n" +
            "    }\n" +
            "}";

    private static final String SPMV = "\n" +
            "extern \"C\" __global__ void spmv(const int *x, const int *y, const float *val, const float *v_in, float *v_out, int num_nnz) {\n" +
            "    int init = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for (int i = init; i < num_nnz; i += stride) {\n" +
            "        v_out[y[i]] += float(float(v_in[x[i]]) * float(val[i]));\n" +
            "    }\n" +
            "}";

    private static final String SUBTRACT = "\n" +
            "extern \"C\" __global__ void subtract(float* v1, const float* v2, const float alpha, int N, int offset) {" +
            "    int init = threadIdx.x + blockIdx.x * blockDim.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    for(int i = init; i < N; i += stride){\n" +
            "        v1[i] -= alpha * v2[i + offset];\n" +
            "    }\n" +
            "}";

    private static final String DOT_PRODUCT = "\n" +
            "extern \"C\" __global__ void dot_product_stage_one(const float* v1, const float* v2, float* temporaryOutputValues, int N, int offset) {\n" +
            "    extern __shared__ float cache[];\n" +
            "    int threadId = threadIdx.x + blockIdx.x * blockDim.x;\n" +
            "    int cacheIdx = threadIdx.x;\n" +
            "    int stride = blockDim.x * gridDim.x;\n" +
            "    float temp = float(0.0);\n" +
            "    for(int i = threadId; i < N; i += stride){\n" +
            "        temp += float(float(v1[i]) * float(v2[i + offset]));\n" +
            "    }\n" +
            "    cache[cacheIdx] = temp;\n" +
            "    __syncthreads();\n" +
            "    for(int i = blockDim.x >> 1; i != 0; i >>= 1){\n" +
            "        if(cacheIdx < i){\n" +
            "            cache[cacheIdx] += cache[cacheIdx + 1];\n" +
            "        }\n" +
            "        __syncthreads();\n" +
            "    }\n" +
            "    if (cacheIdx == 0){\n" +
            "        temporaryOutputValues[blockIdx.x] = cache[0];\n" +
            "    }\n" +
            "}\n";

    @Override
    public void initializeTest(int iteration) {
        // TODO: read CooMatrix
        readMatrix();

        // initialize CPU vectors
        initCPUVectors();

        // Create device buffers
        createGPUVectors();

        // Transfer from CPU to device
        transferToGPU();

        // Create Kernels
        createKernels();

    }

    private void createKernels() {
        Value buildKernel = this.getContext().eval("grcuda", "buildkernel");

        spmv = buildKernel.execute(SPMV, "spmv", "const pointer, const pointer, const pointer, const pointer, pointer, const sint32");
        // TODO: dot product and l2-norm
        axpbXtended = buildKernel.execute(AXPB_XTENDED, "axpb_xtended", "const float, const pointer, const pointer, const float, const pointer, pointer, const sint32, const sint32, const sint32");
        normalize = buildKernel.execute(NORMALIZE, "normalize", "const pointer, const float, pointer, const sint32");
        subtract = buildKernel.execute(SUBTRACT, "subtract", "pointer, const pointer, float, const sint32, const sint32");
        copyPartitionToVec = buildKernel.execute(COPY_PARTITION_TO_VEC, "copy_partition_to_vec", "const pointer, pointer, const sint32, const sint32, const sint32");

    }

    private void transferToGPU() {

        for (int i = 0; i < NUM_PARTITIONS; i++) {
            COOMatrix partition = partitions[i];
            deviceVecIn[i].invokeMember("copyFrom", (Object) vecIn[i], matrix.getN());
            deviceCooX[i].invokeMember("copyFrom", (Object) partition.getX(), partition.getNnz());
            deviceCooY[i].invokeMember("copyFrom", (Object) partition.getY(), partition.getNnz());
            deviceCooVal[i].invokeMember("copyFrom", (Object) partition.getVal(), partition.getNnz());

            for (int j = 0; j < config.blocks; j++) {
                deviceIntermediateDotProductValues[i].getArrayElement(j).setArrayElement(0, 0.0f);
            }
        }
    }

    private void createGPUVectors() {
        Value deviceArray = this.getContext().eval("grcuda", "DeviceArray");

        for (int i = 0; i < NUM_PARTITIONS; i++) {
            COOMatrix partition = this.partitions[i];
            deviceVecIn[i] = deviceArray.execute("float", matrix.getN());
            deviceIntermediateDotProductValues[i] = deviceArray.execute("float", config.blocks);
            alphaIntermediate[i] = deviceArray.execute("float", 1);
            betaIntermediate[i] = deviceArray.execute("float", 1);
            deviceVecOutSpmv[i] = deviceArray.execute("float", partition.getN());
            deviceCooX[i] = deviceArray.execute("float", partition.getNnz());
            deviceCooY[i] = deviceArray.execute("float", partition.getNnz());
            deviceCooVal[i] = deviceArray.execute("float", partition.getNnz());
            deviceVecNext[i] = deviceArray.execute("float", partition.getN());
            deviceNormalizedOut[i] = deviceArray.execute("float", partition.getN());
            deviceLanczosVectors[i] = deviceArray.execute("float", partition.getN() * NUM_EIGEN);
        }

    }

    private void initCPUVectors() {
        vecIn = new float[NUM_PARTITIONS][matrix.getN()];

        // Do this hack since it is not possible to create
        // matrices with different row length in java
        int maxN = -1;
        for (int i = 0; i < NUM_PARTITIONS; i++) {
            if (maxN < partitions[i].getN())
                maxN = partitions[i].getN();
        }

        vecOut = new float[NUM_PARTITIONS][maxN];

        // Generate a random vector
        float norm = 0.0f;
        for (int i = 0; i < matrix.getN(); ++i) {
            vecIn[0][i] = (float) Math.random();
            norm += vecIn[0][i] * vecIn[0][i];
        }

        norm = (float) Math.sqrt(norm);

        // l2-normalize it
        for (int i = 0; i < matrix.getN(); i++) {
            vecIn[0][i] /= norm;
        }

        // mirror the created array to the other partitions
        for (int i = 1; i < NUM_PARTITIONS; i++) {
            System.arraycopy(vecIn[0], 0, vecIn[i], 0, matrix.getN());
        }


    }

    private void readMatrix() {
        // TODO: for the moment being read it from an environmental variable called $COO_MATRIX
        String matrixPath = System.getenv().get("COO_MATRIX");
        this.matrix = COOMatrix.readMatix(matrixPath);
        this.partitions = this.matrix.asPartitions(NUM_PARTITIONS);
    }

    @Override
    public void resetIteration(int iteration) {

    }

    @Override
    public void runTest(int iteration) {

    }

    @Override
    protected void cpuValidation() {

    }


    private static class COOMatrix {
        private int[] x;
        private int[] y;
        private float[] val;
        private int N;
        private int M;
        private int nnz;

        private COOMatrix(int[] x, int[] y, float[] val, int N, int M) {
            this.x = x;
            this.y = y;
            this.val = val;
            this.N = N;
            this.M = M;
            this.nnz = x.length;
        }

        public int[] getX() {
            return x;
        }

        public int[] getY() {
            return y;
        }

        public float[] getVal() {
            return val;
        }

        public int getN() {
            return N;
        }

        public int getNnz() {
            return nnz;
        }

        static COOMatrix readMatix(String path) {

            try (BufferedReader reader = new BufferedReader(new FileReader(path));){

                String currentLine = reader.readLine();

                // Skip comments
                while(currentLine.contains("#"))
                    currentLine = reader.readLine();

                // Read header
                int N, M, nnz;
                String[] headerValues = currentLine.split(" ");

                N = Integer.parseInt(headerValues[0]);
                M = Integer.parseInt(headerValues[1]);
                nnz = Integer.parseInt(headerValues[2]);

                currentLine = reader.readLine();

                int curIdx = 0;

                int[] xVec = new int[nnz];
                int[] yVec = new int[nnz];
                float[] valVec = new float[nnz];

                while (currentLine != null) {
                    String[] values = currentLine.split(" ");

                    int x = Integer.parseInt(values[0]);
                    int y = Integer.parseInt(values[1]);
                    float val = values.length == 3 ? Float.parseFloat(values[2]) : 1.0f;
                    xVec[curIdx] = x;
                    yVec[curIdx] = y;
                    valVec[curIdx] = val;

                    curIdx++;
                    currentLine = reader.readLine();
                }

                return new COOMatrix(xVec, yVec, valVec, N, M);

            } catch (FileNotFoundException e) {
                System.err.println("Invalid path " + path + ". File not found.");
                System.exit(-1);
            } catch (IOException e) {
                System.err.println(e.getMessage());
                System.exit(-1);
            }
            // TODO: assert never reachable....
            return null;
        }

        public COOMatrix[] asPartitions(int numPartitions) {
            COOMatrix[] partitions = new COOMatrix[numPartitions];

            int chunkSize = (int) ((float) (this.nnz + numPartitions - 1))/ numPartitions;
            int begin = 0;
            int end = chunkSize;

            for(int i = 0; i < numPartitions - 1; ++i){
                int[] xChunk = new int[nnz];
                int[] yChunk = new int[nnz];
                float[] valChunk = new float[nnz];



            }

            // TODO:
            return partitions;
        }

    }

}
