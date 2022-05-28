package it.necst.grcuda.benchmark;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class DebugTest {
    BenchmarkConfig benchConfig;

    @Before
    public void init(){
        benchConfig = new BenchmarkConfig();
        benchConfig.benchmarkName = "B1";
        benchConfig.setupId = "";
        benchConfig.totIter = 1;
        benchConfig.currentIter = 0;
        benchConfig.randomSeed = 42;
        benchConfig.size = 60000000;
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
    }

    @Test
    public void multiContext_SimpleBench() throws InterruptedException {
        SimpleBench bench;

        System.out.println("BENCH (1) - START");
        bench = new SimpleBench(benchConfig);
        bench.run();
        System.out.println("BENCH (1) - END");

        System.out.println("#################################");
        TimeUnit.SECONDS.sleep(10);
        System.out.println("#################################");

        System.out.println("BENCH (2) - START");
        bench = new SimpleBench(benchConfig);
        bench.run();
        System.out.println("BENCH (2) - END");

    }

    @Test
    public void multiContext_SimpleBench_thread() throws InterruptedException {
        Thread thread;

        thread = new Thread(() -> {
            SimpleBench bench;
            System.out.println("BENCH (1) - START");
            bench = new SimpleBench(benchConfig);
            bench.run();
            System.out.println("BENCH (1) - END");
        });
        thread.start();
        thread.join();

        System.out.println("#################################");
        TimeUnit.SECONDS.sleep(10);
        System.out.println("#################################");


        thread = new Thread(() -> {
            SimpleBench bench;
            System.out.println("BENCH (2) - START");
            bench = new SimpleBench(benchConfig);
            bench.run();
            System.out.println("BENCH (2) - END");
        });
        thread.start();
        thread.join();
    }

    @Test
    public void multiContext_SimpleBenchProcess() throws InterruptedException, IOException {
        LinkedList<String> args = new LinkedList<>();
        args.add(Integer.toString(250000000)); // ==> 1GB array

        System.out.println("BENCH (1) - START");
        JavaProcess.exec(SimpleBenchProcess.class, args);
        System.out.println("BENCH (1) - END");

        System.out.println("#################################");
        TimeUnit.SECONDS.sleep(10); // timer to see in nvtop the free of GPU mem
        System.out.println("#################################");


        System.out.println("BENCH (2) - START");
        JavaProcess.exec(SimpleBenchProcess.class, args);
        System.out.println("BENCH (2) - END");

    }

}

final class JavaProcess {
    // taken from https://stackoverflow.com/questions/636367/executing-a-java-application-in-a-separate-process

    private JavaProcess() {}

    public static int exec(Class klass, List<String> args) throws IOException,
            InterruptedException {
        String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +
                File.separator + "bin" +
                File.separator + "java";
        String classpath = System.getProperty("java.class.path");
        String className = klass.getName();

        List<String> command = new LinkedList<String>();
        command.add(javaBin);
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        if (args != null) {
            command.addAll(args);
        }

        ProcessBuilder builder = new ProcessBuilder(command);

        Process process = builder.inheritIO().start();
        process.waitFor();
        return process.exitValue();
    }

}

class SimpleBench{
    private static final String SQUARE_KERNEL = "" +
            "extern \"C\" __global__ void square(float* x, int n) { \n" +
            "    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n" +
            "        x[i] = x[i] * x[i];\n" +
            "    }\n" +
            "}\n";
    private  final BenchmarkConfig benchConfig;

    SimpleBench(BenchmarkConfig benchConfig){
        this.benchConfig = benchConfig;
    }

    private Context createContext(){
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


    public void run() {
        try(Context context = createContext()){
            // Create array
            Value deviceArray = context.eval("grcuda", "DeviceArray");
            Value x = deviceArray.execute("float", 250000000); // 1GB array
            for (int i = 0; i < benchConfig.size; i++)
                x.setArrayElement(i, 3.0f);

            // Build kernel
            Value buildKernel = context.eval("grcuda", "buildkernel");
            Value squareKernelFunction = buildKernel.execute(SQUARE_KERNEL, "square", "pointer, sint32");

            // Execute kernel
            squareKernelFunction.execute(benchConfig.numBlocks, benchConfig.blockSize1D) // Set parameters
                    .execute(x, 250000000); // Execute actual kernel

            // Sync step to measure the real computation time
            Float res = x.getArrayElement(0).asFloat();
            System.out.println("RES: "+res.toString());

            x.invokeMember("free"); // this actually free mem
        }
    }
}