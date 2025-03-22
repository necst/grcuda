#!/bin/bash

######## setup_env ########
source env.sh

######## compute interconnect. matrix (only needed one time) ########
cd $GRCUDA_HOME/projects/resources/connection_graph
./run.sh 

######## sanity_check (run a small unittest) ########
cd $GRCUDA_HOME
mx unittest com.nvidia.grcuda.test.BuildKernelTest#testBuildKernelwithNFILegacytSignature

######## check GPUs ########
# The java workloads will automatically perform the full evaluation based on the configurations files for the current GPU architecture.
# Within "$GRCUDA_HOME/projects/resources/java/grcuda-benchmark/src/test", you'll find e.g., "config_A100.json". 
# In the same folder, the "TestBenchmarks.java" will try to map the current detected GPU in the system with one of the configuration files.
# If no configuration is found, it will fall back to "config_fallback.json".
# This command will print the GPU name of the currently available GPUs in the system so that you can change TestBenchmarks.java accordingly. 
nvidia-smi --query-gpu=gpu_name --format=csv

######## run benchmark ########
export JAVA_HOME=$GRAAL_HOME
cd $GRCUDA_HOME/projects/resources/java/grcuda-benchmark
mvn test --offline