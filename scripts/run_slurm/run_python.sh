#!/bin/bash
#SBATCH -A IscrC_GrOUT
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --gres=gpu:2        
#SBATCH --mem=64000          
#SBATCH --job-name=grcuda_python
#SBATCH --error=python_%j.err            # standard error file
#SBATCH --output=python_%j.out           # standard output file

######## setup_env ########
source env.sh

######## build ########
mx build;

######## install ########
mkdir -p $GRAAL_HOME/languages/grcuda;
cp $GRCUDA_HOME/mxbuild/dists/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
cd $GRCUDA_HOME/projects/resources/connection_graph
./run.sh 
cd $GRCUDA_HOME

######## sanity_check ########
mx unittest com.nvidia.grcuda.test.BuildKernelTest#testBuildKernelwithNFILegacytSignature

######## check GPUs ########
nvidia-smi --query-gpu=gpu_name --format=csv

######## run benchmark ########
export JAVA_HOME=$GRAAL_HOME
source $INSTALL_DIR/graalpython_venv/bin/activate
cd $GRCUDA_HOME/projects/resources/python/benchmark
graalpython --jvm --polyglot benchmark_wrapper_custom.py -d -i 1