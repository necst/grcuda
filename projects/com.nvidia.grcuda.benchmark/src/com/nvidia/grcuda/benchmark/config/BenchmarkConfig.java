package com.nvidia.grcuda.benchmark.config;

public class BenchmarkConfig {
    /**
     * Default parameters
     */
    public String name = "";
    public int testSize = 100;
    public int threadsPerBlock = 128;
    public int blocks = 8;
    public boolean randomInit = false;
    public boolean reInit = false;
    public int iterations = 100;
    public boolean cpuValidate = true;
}
