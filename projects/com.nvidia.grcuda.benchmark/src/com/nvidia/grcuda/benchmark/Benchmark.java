package com.nvidia.grcuda.benchmark;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import com.nvidia.grcuda.benchmark.config.BenchmarkConfig;
import com.nvidia.grcuda.benchmark.config.Options;
import org.graalvm.polyglot.Context;
import org.junit.After;
import org.junit.Test;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

abstract class Benchmark {

    private final Context context;

    protected BenchmarkConfig config;
    private final List<Long> executionTimes = new ArrayList<>();
    private final BenchmarkResults benchmarkResults = new BenchmarkResults();

    private final static String DEFAULT_CONFIG_PATH = "$HOME/grcuda/grcuda-data/config.json";
    private boolean shouldSkipTest;


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
            this.init(0);
        } else {
            System.out.println("Skipping " + this.getBenchmarkName());
        }
    }


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

        } catch (IOException | NullPointerException ignored) {
            throw new RuntimeException("File not found, aborting...");
        }

    }

    private void time(int iteration, String phaseName, Consumer<Integer> functionToTime){
        long begin = System.nanoTime();
        functionToTime.accept(iteration);
        this.benchmarkResults.addPhase(phaseName, System.nanoTime() - begin);
    }

    public void reset(int iteration){
        time(iteration, "reset", this::resetIteration);
    }

    @Test
    public void run() {

        if (this.shouldSkipTest) return;

        for (int i = 0; i < getIterationsCount(); ++i) {
            if (config.reInit || i == 0) this.init(i);
            this.reset(i);
            time(i, "execution", this::runTest);
            if (config.cpuValidate) cpuValidation();
        }
    }

    /**
     * Save the results in a file or print them
     */
    @After
    public void saveResults() {
        if (this.shouldSkipTest) return;
        System.out.println(this.benchmarkResults);
        //System.out.println(this.benchmarkResults.filter("init"));
        //System.out.println(this.benchmarkResults.filter("execution"));
        //System.out.println(this.benchmarkResults.filter("reset"));

    }

    public Context getContext() {
        return context;
    }

    public String getBenchmarkName() {
        return this.getClass().getSimpleName();
    }

    public void init(int iteration){
        time(iteration, "init", this::initializeTest);
    }

    /**
     * Here goes the read of the test parameters,
     * the initialization of the necessary arrays
     * and the creation of the kernels (if applicable)
     * @param iteration
     */
    public abstract void initializeTest(int iteration);

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     */
    public abstract void resetIteration(int iteration);


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