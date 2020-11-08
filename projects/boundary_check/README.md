# CUDA Boundary Check 

## Introduction

* See paper at https://www.overleaf.com/read/dpvjqgstyfcy

## Current limitations

* For maximum compatibility, input IR should be generated with `-O0` level optimizations
    * Higher optimizations might still work, depending on the complexity of the kernel
    * Further optimizations can be applied after our transformations, see the `Makefile` in `examples/truffle_kernels`
    * Applying optimizations after transformations doesn't decrease performance, according to our tests

***

## Setup

### Building LLVM and adding custom Transformation passes

* As a general reference, the procedure is similar to https://github.com/upenn-acg/gpudrano-static-analysis_v1.0
	* Note that for installing LLVM it's better to follow the official GitHub repo: https://github.com/llvm/llvm-project
* Get LLVM and clang from https://github.com/llvm/llvm-project
	* `cd llvm-project`
	* `mkdir build; cd build`
	* `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS='clang;libcxx;libcxxabi' ../llvm`, this will build LLVM together with Clang, libcxx, and libcxxabi, with `-O3` optimizations, using the Ninja build generator (the default)
	* GCC 5 is required (at least), on Centos enable it as `scl enable devtoolset-7 bash`
	* After CMake, build with `cmake --build .  -j16 --target check-all`, this will build LLVM with Ninja using 16 cores, and rull all integration tests (`check-all` target). Use `--target ninja` to skip integrations tests.
* LLVM build notes:
	* CMake 3.13 is required for installing LLVM
	* Create an alias as `alias cmake=cmake3` or follow `https://stackoverflow.com/questions/48831131/cmake-on-linux-centos-7-how-to-force-the-system-to-use-cmake3`
	* Install the Ninja build tool, `https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages`, on Centos it is `sudo yum install ninja-build`
	* Specify `CLANG_DIR` as the path where the Clang executable is, e.g. `llvm-project/build/bin`, and optionally add it to PATH. Also specify `LLVM_BUILD_DIR` as `llvm-project/build`
* The CUDA pass can be added to the `Transforms` folder with a symlink (e.g. `ln -s /path/to/grcuda/projects/boundary_check/fix_oob/src CudaFixOoBPass`, executed from `llvm/lib/Transforms`)
	* `cd llvm-project/llvm/lib/Transforms`
	* `ln -s /path/to/grcuda/projects/boundary_check/src/add_sizes CudaAddSizePass`
	* `ln -s /path/to/grcuda/projects/boundary_check/src/fix_oob CudaFixOoBPass`
* We must registers our custom passes to LLVM inside CMake. Add `add_subdirectory(CudaAddSizePass)` and `add_subdirectory(CudaFixOoBPass)` to `llvm-project/llvm/lib/Transforms/CMakeLists.txt`
* Now rebuild LLVM, starting from CMake (no need to `cmake clean` though)

* LLVM can be built as release with: `cmake -DCMAKE_BUILD_TYPE=Release  ../llvm`
    * Using `Release` builds LLVM with `-O3` level optimizations

### Setting up grCUDA

* Main repository: https://ol-bitbucket.us.oracle.com/users/lukas.stadler_oracle.com/repos/trufflecuda/browse
* Download GraalVM: https://github.com/oracle/graal/releases
* Setup GraalVM: https://www.graalvm.org/docs/getting-started
    * The correct path is something like `JAVA_HOME=~/graalvm-ce-19.1.1`
    * Install the Python component: `gu install python`
    * Install the `mx` tool: https://github.com/oracle/graal/tree/92f0b657a5f4892f7380517a714bf93d13db2530/tools
    * Follow the main grCUDA README to build it
    
***


## Usage

### Compile CUDA code with Clang (for reference)

* Refer to https://llvm.org/docs/CompileCudaWithLLVM.html
* For example, run:

`CLANG_DIR/clang++ axpy.cu -o axpy --cuda-gpu-arch=sm_60 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -std=c++11`

### Generate LLVM IR

* To generate the original LLVM IR for a single kernel (in this case `axpy.cu`), run:

`CLANG_DIR/clang++  axpy.cu  --cuda-gpu-arch=sm_60  -pthread -std=c++11  -S -emit-llvm`

* This is useful if you want to manually apply the transformation passes to the IR, for debugging purposes.

### Run transformation passes using a standalone executable (for debugging)

* Build the transformations:

```
cd boundary_check
make
```

* Make sure that you have LLVM code to give as input! 
    * See [Generate LLVM IR](#generate-llvm-ir)

* Run the pass that adds sizes to the kernel signature

`build/add_sizes_cuda_pass --input_file examples/axpy/axpy.ll --kernel axpy --debug --dump_updated_kernel`

* Run the pass that adds boundary checks
    * Make sure that the input was processed with the `add_sizes` pass first!   
    * Either run that pass with `opt` or copy the result obtained from the previous command 

`build/fix_oob_cuda_pass --input_file examples/axpy/axpy_with_sizes.ll --kernel axpy --debug --dump_updated_kernel --lower_bounds`


### Run passes with `opt`  

* Follow [this guide](http://llvm.org/docs/WritingAnLLVMPass.html#running-a-pass-with-opt) to add the transformation pass to a working LLVM build.
* This allows to integrate the transformation passes in LLVM compilation pipelines, and save the results of the transformations.

* Run the `add_sizes_cuda_pass`

`CLANG_DIR/opt --debug-pass=Structure -S -o examples/truffle_kernels/llvm/added_size/O0/no_simplification/axpy.ll -load CLANG_DIR/../lib/LLVMCudaAddSizePass.so -add_sizes_cuda_pass < examples/truffle_kernels/llvm/original/axpy.ll --kernel axpy`

* Run the `fix_oob_cuda_pass`

`CLANG_DIR/opt --debug-pass=Structure -S -oexamples/truffle_kernels/llvm/added_size/O0/no_simplification/axpy.ll -load CLANG_DIR/../lib/LLVMCudaFixOoBPass.so -fix_oob_cuda_pass < examples/truffle_kernels/llvm/original/axpy.ll --kernel axpy`


### Compiling a CUDA kernel from source to binary, with our transformations

* Refer to the `Makefile` in `benchmark`
* The function `compile_kernel` takes CUDA code as input, applies our transformations, and outputs a binary file
* For example, you can define the following command

```
axpy: cuda/axpy.cu
	$(call compile_kernel,$^,axpy)
```

***

## Testing

* This repository includes 15 CUDA kernels that can be used to test our transformation passes
* Kernels are located in `benchmark/cuda`
* Running `./build_all.sh` will compile all the kernels with transformation passes, and also compile manually modified and unmodified kernels that can be used for validation
* Generated LLVM code is stored inside `examples/truffle_kernels/llvm`. It is kept for debugging and manual inspection, but it's not required for execution
* `benchmark/run_kernels/python` contains a number of scripts that show how to launch CUDA kernels using grCUDA. In each script, we compare the manually modified or original kernel to the one that we automatically modified, and check results and execution time. 
* To run all the Python examples and store results, run the following

```
cd examples/truffle_kernels/run_kernels/python
python3 run.py
```

* Results are stored in `data/results`
* `resources/python` contains a number of scripts to generate plots and summary tables from these results

* Short guide to run all the examples from scratch:

```
Download LLVM, GraalVM & grCUDA
Create symlink (or copy source code of transformations) inside LLVM Transforms folder
Build LLVM
Build GraalVM
Build grCUDA

cd examples/truffle_kernels
./build_all.sh
cd run_kernels/python
python3 run.py
```
