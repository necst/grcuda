package com.nvidia.grcuda.benchmark;

import org.graalvm.polyglot.Context;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.experimental.theories.DataPoint;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

@RunWith(Theories.class)
abstract class Benchmark {
    private static final int MAX_ITERATIONS = 100;
    private final boolean reInit = false;
    private final Context grcudaContext;
    // The following variables should be read from a config file
    // For simplicity, I'm initializing them statically now
    protected final int TEST_SIZE = 1000;
    protected final boolean randomInit = false;
    protected final int NUM_BLOCKS = 8;
    protected final int NUM_THREADS = 128;
    protected final boolean cpuValidate = false;
    protected long executionTime;

    Benchmark(){
        // Parse options from a file
        this.grcudaContext = this.buildBenchmarkContext();
        this.init();
    }

    public static int[] iterations(){
        int[] iterations = new int[MAX_ITERATIONS];
        for(int i = 0; i < MAX_ITERATIONS; ++i){
            iterations[i] = i;
        }
        return iterations;
    }

    /**
     * Utility function to build the GrCUDA Context
     */
    private Context buildBenchmarkContext(){
        return Context.newBuilder().allowAllAccess(true).allowExperimentalOptions(true)
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                .build();
    }

    /**
     * Here goes the read of the test parameters,
     * the initialization of the necessary arrays
     * and the creation of the kernels (if applicable)
     */
    public abstract void init();

    public abstract void resetIteration();

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     */
    @Before
    public void reset(){
        if (this.reInit) this.init();
        this.resetIteration();
    }

    /**
     * Run the actual test
     */
    @Theory
    public abstract void run(int iteration);

    /**
     * Save the results in a file or print them
     */
    @After
    public abstract void saveResults();

    protected abstract void cpuValidation();

    public Context getGrcudaContext() {
        return grcudaContext;
    }



}