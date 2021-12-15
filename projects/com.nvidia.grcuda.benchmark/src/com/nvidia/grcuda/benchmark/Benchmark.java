package com.nvidia.grcuda.benchmark;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import com.nvidia.grcuda.benchmark.config.BenchmarkConfig;
import com.nvidia.grcuda.benchmark.config.Options;
import org.graalvm.polyglot.Context;
import org.junit.After;
import org.junit.Test;
import org.junit.experimental.theories.Theories;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@RunWith(Theories.class)
abstract class Benchmark {

    private final Context context;

    protected BenchmarkConfig config;
    private final List<Long> executionTimes = new ArrayList<>();

    private final static String DEFAULT_CONFIG_PATH = "$HOME/grcuda/grcuda-data/config.json";
    // The following variables should be read from a config file
    // For simplicity, I'm initializing them statically now

    private boolean shouldSkipTest;


    public void readConfig() {
        String configPath;
        Gson gson = new Gson();

        try {
            configPath = System.getProperty("benchmarkConfigPath");
        } catch (IllegalArgumentException ignored) {
            configPath = DEFAULT_CONFIG_PATH;
        }

        File configFile = new File(configPath);

        try (JsonReader reader = new JsonReader(new FileReader(configFile))) {
            Options options = gson.fromJson(reader, Options.class);

            for (BenchmarkConfig config : options.benchmarks) {
                if (this.getBenchmarkName().equalsIgnoreCase(config.name))
                    this.config = config;
            }

            this.shouldSkipTest = this.config == null;

        } catch (IOException e) {
            throw new RuntimeException("File not found, aborting...");
        }

    }

    Benchmark() {
        context = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                .build();
        // Parse options from a file
        this.readConfig();

        if (!this.shouldSkipTest) {
            this.init();
        } else {
            System.out.println("Skipping " + this.getBenchmarkName());
        }
    }


    @Test
    public void run() {

        if (this.shouldSkipTest) return;

        for (int i = 0; i < getIterationsCount(); ++i) {

            if (config.reInit) this.init();
            this.resetIteration();

            long beginTime = System.nanoTime();
            this.runTest(i);
            executionTimes.add(System.nanoTime() - beginTime);
            if (config.cpuValidate) cpuValidation();

        }


    }

    /**
     * Save the results in a file or print them
     */
    @After
    public void saveResults() {
        if (this.shouldSkipTest) return;
        double avgExecutionTime = this.executionTimes.stream().mapToDouble(v -> v).average().getAsDouble();
        System.out.println("Benchmark " + this.getBenchmarkName() + " took " + avgExecutionTime + "ns over " + this.executionTimes.size() + " runs");
    }

    public Context getContext() {
        return context;
    }

    public String getBenchmarkName() {
        return this.getClass().getSimpleName();
    }


    /**
     * Here goes the read of the test parameters,
     * the initialization of the necessary arrays
     * and the creation of the kernels (if applicable)
     */
    public abstract void init();

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     */
    public abstract void resetIteration();


    /**
     * Run the actual test
     */
    public abstract void runTest(int iteration);


    /**
     * (numerically) validate results against CPU
     */
    protected abstract void cpuValidation();

    /**
     * Override it to reduce or increase the number of repetitions
     */
    protected int getIterationsCount() {
        return config.iterations;
    }

    protected int getTestSize() {
        return config.testSize;
    }


}