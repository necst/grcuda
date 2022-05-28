package it.necst.grcuda.benchmark;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

public class SimpleBenchProcess{
    private static final String SQUARE_KERNEL = "" +
            "extern \"C\" __global__ void square(float* x, int n) { \n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        x[i] = x[i] * x[i];\n" +
            "    }\n" +
            "}\n";

    private static Context createContext(BenchmarkConfig benchConfig){
        return  Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                .option("grcuda.ExecutionPolicy", benchConfig.executionPolicy)
                .option("grcuda.InputPrefetch", String.valueOf(benchConfig.inputPrefetch))
                .option("grcuda.RetrieveNewStreamPolicy", benchConfig.retrieveNewStreamPolicy)
                .option("grcuda.RetrieveParentStreamPolicy", benchConfig.retrieveParentStreamPolicy)
                .option("grcuda.DependencyPolicy", benchConfig.dependencyPolicy)
                .option("grcuda.DeviceSelectionPolicy", benchConfig.deviceSelectionPolicy)
                .option("grcuda.ForceStreamAttach", String.valueOf(benchConfig.forceStreamAttach))
                .option("grcuda.EnableComputationTimers", String.valueOf(benchConfig.enableComputationTimers))
                .option("grcuda.MemAdvisePolicy", benchConfig.memAdvisePolicy)
                .option("grcuda.NumberOfGPUs", String.valueOf(benchConfig.numGpus))
                .option("grcuda.BandwidthMatrix", benchConfig.bandwidthMatrix)
                .build();
    }

    public static void main(String[] args) {
        BenchmarkConfig benchConfig;
        benchConfig = new BenchmarkConfig();
        benchConfig.benchmarkName = "B1";
        benchConfig.setupId = "";
        benchConfig.totIter = 1;
        benchConfig.currentIter = 0;
        benchConfig.randomSeed = 42;
        benchConfig.size = Integer.parseInt(args[0]);
        benchConfig.blockSize1D = 32;
        benchConfig.blockSize2D = 8;
        benchConfig.timePhases = false;
        benchConfig.numBlocks = 32;
        benchConfig.randomInit = false;
        benchConfig.reInit = false;
        benchConfig.reAlloc = false;
        benchConfig.cpuValidate = true;
        benchConfig.executionPolicy = "sync";
        benchConfig.inputPrefetch = false;
        benchConfig.retrieveNewStreamPolicy = "always-new";
        benchConfig.retrieveParentStreamPolicy = "disjoint";
        benchConfig.dependencyPolicy = "with-const";
        benchConfig.deviceSelectionPolicy = "round-robin";
        benchConfig.forceStreamAttach = false;
        benchConfig.numGpus = 2;
        benchConfig.memAdvisePolicy = "none";
        benchConfig.bandwidthMatrix="/home/users/ian.didio/grcuda/projects/resources/connection_graph/datasets/connection_graph_2_v100.csv";

        try(Context context = createContext(benchConfig)){
            // Create array
            Value deviceArray = context.eval("grcuda", "DeviceArray");
            Value x = deviceArray.execute("float", benchConfig.size); // 1GB array
            for (int i = 0; i < benchConfig.size; i++)
                x.setArrayElement(i, 3.0f);

            // Build kernel
            Value buildKernel = context.eval("grcuda", "buildkernel");
            Value squareKernelFunction = buildKernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");

            // Execute kernel
            squareKernelFunction.execute(benchConfig.numBlocks, benchConfig.blockSize1D) // Set parameters
                    .execute(x, benchConfig.size); // Execute actual kernel

            // Sync step to measure the real computation time
            Float res = x.getArrayElement(0).asFloat();
            System.out.println("RES: "+res.toString());
        }
    }
}