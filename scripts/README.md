# Useful scripts

This folder contains useful scripts to build GrCUDA and run both the Java and Python. 
The scripts are expected, unless otherwise specified, to be run from the scripts' folder (i.e., `cd` to the folder containing the script)

### Generic setup

The scripts in this folder are intended for a generic/custom setup.
The only requirements that are not automatically satisfied by the scripts are:

- `cuda`: a properly configured version of cuda (tested: 11.4 - 11.7 - 12.1)
- `maven`: necessary to run the Java workloads

The installation proceeds by modifying accordingly `env.sh`, and then running `bash generic_setup/setup.sh` and `bash generic_setup/install.sh`.

The file `env.sh` contains the env variables that needs to be properly set to build and run GrCUDA. 

##### Expected folder structure

Users should clone grcuda's repository in its own directory, named `grcuda`.
After you have run the generic setup, the content of `$INSTALL_DIR` (specified in `env.sh`) should be the following:

- grcuda (clone of grcuda's repository)
- graal
- graalvm-ce-java11.22.1.0
- mx
- labsjdk-ce-11.0.15-jvmci-22.1-b01
- graalpython_venv




### OCI setup

This folder contains the scripts necessary for a full install of GrCUDA on OCI resources as detailed by the outermost README of this repository. 

### run_local

The scripts within this folder are used to run the java workloads on a simple local multi-GPU machine.
The file `env.sh` contains the env variables that needs to be properly set to build and run GrCUDA. 

### run_slurm

The scripts within this folder are used to run the python and java workloads within a slurm setup. The file `env.sh` contains the env variables that needs to be properly set to build and run GrCUDA. 
