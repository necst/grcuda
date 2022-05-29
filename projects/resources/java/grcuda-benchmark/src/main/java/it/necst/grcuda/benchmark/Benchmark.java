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

import java.util.ArrayList;
import java.util.function.Consumer;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

public abstract class Benchmark {
    public Context context;
    public final BenchmarkConfig config;
    public final BenchmarkResults benchmarkResults;
    public ArrayList<Value> deviceArrayList = new ArrayList<>(); // used to store all the arrays to be freed at the end of the benchmark
    public Value deviceArray;

    public Benchmark(BenchmarkConfig currentConfig) {
        this.config = currentConfig;
        this.benchmarkResults = new BenchmarkResults(currentConfig);
        this.context = createContext(currentConfig);
        this.deviceArray = context.eval("grcuda", "DeviceArray");
    }

    /**
     * This method is used to run the current benchmark.
     * It will use the information stored in the config attribute to decide whether to do an additional initialization phase and
     the cpuValidation.
     */
    public void run() {
        if(config.debug)
            System.out.println("INSIDE run()");

        for (int i = 0; i < config.totIter; ++i) {

            // Start a timer to monitor the total GPU execution time
            long overall_gpu_start = System.nanoTime();

            // Allocate memory for the benchmark
            if (config.reAlloc || i == 0) time(i, "alloc", this::allocateTest);

            // Initialize memory for the benchmark
            if (config.reInit || i == 0) time(i, "init", this::initializeTest);

            // Reset the result
            time(i, "reset", this::resetIteration);

            //TODO: add nvprof_profile step

            // Execute the benchmark
            time(i, "execution", this::runTest);

            //TODO: stop nvprof_profile step

            // Stop the timer
            long overall_gpu_end = System.nanoTime();

            // Perform validation on CPU
            if (config.cpuValidate)
                cpuValidation();

            if(config.debug)
                System.out.println("VALIDATION \nCPU: " + benchmarkResults.cpu_result +"\nGPU: " + benchmarkResults.gpu_result);
        }

        // Save the benchmark results
        saveResults();

        // Free the allocated arrays
        deallocDeviceArrays();

        //  Gracefully close the current context
        context.close();
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
     * This function is tasked with saving a json file with the results of the current benchmark, computing the needed
     statistics.
     */
    private void saveResults() {
        if(config.debug)
            System.out.println(this.benchmarkResults);
        /*  TODO:
                - Compute the various statistics as in the Python version of the benchmarks
                - Store the statistics in a json object
                - Save the json object to a file
         */
    }

    protected void deallocDeviceArrays(){
        for(Value v : deviceArrayList)
            v.invokeMember("free");
    }

    protected Value requestArray(String type, int size){
        Value vector = deviceArray.execute(type, size);
        deviceArrayList.add(vector);
        return vector;
    }

    private Context createContext(BenchmarkConfig config){
        return Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                //logging settings
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                //GrCUDA env settings
                .option("grcuda.ExecutionPolicy", config.executionPolicy)
                .option("grcuda.InputPrefetch", String.valueOf(config.inputPrefetch))
                .option("grcuda.RetrieveNewStreamPolicy", config.retrieveNewStreamPolicy)
                .option("grcuda.RetrieveParentStreamPolicy", config.retrieveParentStreamPolicy)
                .option("grcuda.DependencyPolicy", config.dependencyPolicy)
                .option("grcuda.DeviceSelectionPolicy", config.deviceSelectionPolicy)
                .option("grcuda.ForceStreamAttach", String.valueOf(config.forceStreamAttach))
                .option("grcuda.EnableComputationTimers", String.valueOf(config.enableComputationTimers))
                .option("grcuda.MemAdvisePolicy", config.memAdvisePolicy)
                .option("grcuda.NumberOfGPUs", String.valueOf(config.numGpus))
                .option("grcuda.BandwidthMatrix", config.bandwidthMatrix)
                .build();
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
    protected abstract void initializeTest(int iteration);

    /**
     * Allocate new memory on GPU used for the benchmark
     * @param iteration the current number of the iteration
     */
    protected abstract void allocateTest(int iteration);

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     * @param iteration the current number of the iteration
     */
    protected abstract void resetIteration(int iteration);

    /**
     * Run the actual test
     * @param iteration the current number of the iteration
     */
    protected abstract void runTest(int iteration);

    /**
     * (numerically) validate results against CPU
     */
    protected abstract void cpuValidation();

}
