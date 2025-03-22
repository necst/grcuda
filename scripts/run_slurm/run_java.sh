#!/bin/bash
#SBATCH -A XXX
#SBATCH -p XXX
#SBATCH --qos XXX
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --gres=gpu:2        
#SBATCH --mem=64000          
#SBATCH --job-name=grcuda_java
#SBATCH --error=java_%j.err            # standard error file
#SBATCH --output=java_%j.out           # standard output file

######## setup_env ########
source env.sh

######## install ########
cd $GRCUDA_HOME/projects/resources/connection_graph
./run.sh 
cd $GRCUDA_HOME

######## sanity_check ########
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