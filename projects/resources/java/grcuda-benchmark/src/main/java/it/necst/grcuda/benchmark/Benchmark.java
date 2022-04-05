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

package it.necst.grcuda.benchmark;

import java.util.function.Consumer;
import org.graalvm.polyglot.Context;

public abstract class Benchmark {
    public final Context context;
    public final BenchmarkConfig config;
    public final BenchmarkResults benchmarkResults;

    public Benchmark(BenchmarkConfig currentConfig) {
        context = Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                .build();
        config = currentConfig;
        benchmarkResults = new BenchmarkResults(config.name, config.setupId);
    }

    /**
     * This method is used to time the function passed to it.
     * It will add the timing and the phase name to the benchmarkResult attribute.
     * @param iteration the current iteration of the benchmark
     * @param phaseName the current phase of the benchmark
     * @param functionToTime the function to time passed like "class::funName"
     */
    private void time(int iteration, String phaseName, Consumer<Integer> functionToTime){
        long begin = System.nanoTime();
        functionToTime.accept(iteration);
        this.benchmarkResults.addPhase(phaseName, System.nanoTime() - begin, iteration);
    }

    /**
     * This method is used to run the current benchmark.
     * It will use the information stored in the config attribute to decide whether to do an additional initialization phase and the cpuValidation.
     */
    public void run() {
        for (int i = 0; i < config.iterations; ++i) {
            if (config.reInit || i == 0)
                time(i, "init", this::initializeTest);

            time(i, "reset", this::resetIteration);

            time(i, "execution", this::runTest);
            
            if (config.cpuValidate) 
                cpuValidation();
        }
    }

    /**
     * This function is tasked with saving a json file with the results of the current benchmark, computing the needed statistics.
     */
    public void saveResults() {
        System.out.println(this.benchmarkResults);
        /*  TODO:
                - Compute the various statistics as in the Python version of the benchmarks
                - Store the statistics in a json object
                - Save the json object to a file
         */
    }



    /*
        ###################################################################################
                        METHODS TO BE IMPLEMENTED IN THE BENCHMARKS
        ###################################################################################
    */
    
    /**
     * Here goes the read of the test parameters,
     * the initialization of the necessary arrays
     * and the creation of the kernels (if applicable)
     * @param iteration the current number of the iteration
     */
    public abstract void initializeTest(int iteration);

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     * @param iteration the current number of the iteration
     */
    public abstract void resetIteration(int iteration);

    /**
     * Run the actual test
     * @param iteration the current number of the iteration
     */
    public abstract void runTest(int iteration);

    /**
     * (numerically) validate results against CPU
     */
    public abstract void cpuValidation();

}