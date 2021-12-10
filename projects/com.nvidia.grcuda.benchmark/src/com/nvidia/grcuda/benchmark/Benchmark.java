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
    protected static final int TEST_SIZE = 1000;

    public static int[] iterations(){
        int[] iterations = new int[MAX_ITERATIONS];
        for(int i = 0; i < MAX_ITERATIONS; ++i){
            iterations[i] = i;
        }
        return iterations;
    }

    /**
     * This is a direct copy of GrCUDATestUtil::buildTestContext
     * for the moment being it stays here as I could not find how to import things
     * from com.nvidia.grcuda.test
     * @return
     */
    public static Context buildBenchmarkContext(){
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
    @BeforeClass
    public static void init() {
        throw new RuntimeException("This shouldn't ever be called");
    }

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     */
    @Before
    public abstract void reset();

    /**
     * Run the actual test
     */
    @Theory
    public abstract void run(int iteration);

    /**
     * Save the results in a file
     */
    @After
    public abstract void saveResults();

    protected abstract void cpuValidation();


}