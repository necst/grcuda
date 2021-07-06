import polyglot
import time
import numpy as np
from random import random, randint, seed

from benchmark import Benchmark, time_phase, DEFAULT_BLOCK_SIZE_1D, DEFAULT_NUM_BLOCKS
from benchmark_result import BenchmarkResult
from java.lang import System
import math


##############################
##############################

JACOBI_KERNEL = """
extern "C" __global__ void JacobiIter(int n, float *a, float *x, int offset, int max,float*b, float*x_result){
    float buf = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + offset; idx < max; idx += blockDim.x * gridDim.x){
        for (int idy = blockIdx.y * blockDim.y + threadIdx.y; idy < n; idy += blockDim.y * gridDim.y){
            if(idx != idy){
                buf += a[idx*n + idy] * x[idy];
            }
        }
        x_result[idx - offset] = (b[idx] - buf)/a[idx + idx*n];
    }
}
"""

MERGE_KERNEL = """
extern "C" __global__ void mergeResults(int n, int nGPU, int offset, float *x, float *x_result_0, float *x_result_1){

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n/nGPU; idx += blockDim.x * gridDim.x){
        x[idx] = x_result_0[idx];
        x[idx + offset] = x_result_1[idx];
    }
}

"""

class Benchmark11(Benchmark):
    def __init__(self, benchmark: BenchmarkResult, nvprof_profile: bool = False):
        super().__init__("b11", benchmark, nvprof_profile)
        self.size = 0

        # self.num_blocks = DEFAULT_NUM_BLOCKS
        self.sum_kernel = None
        self.cpu_result = 0
        self.block_size = DEFAULT_BLOCK_SIZE_1D

        self.NGPU = 2
        self.ITER = 50

        self.x_result_d = [[]] * self.NGPU
        self.a_d = None
        self.b_d = None
        self.x_d = None


        self.JacobiIter_kernel = None
        self.merge_kernel = None

    @time_phase("allocation")
    def alloc(self, size: int, block_size: dict = None) -> None:
        self.size = size
        self.block_size = block_size["block_size_1d"]
        self.partition_size = self.size/self.NGPU
        # print("partition: "+self.partition_size)
        # print("size: "+size)
        for i in range(self.NGPU):
            self.x_result_d[i] = polyglot.eval(language="grcuda", string=f"float[{int(self.partition_size)}]")
        self.a_d = polyglot.eval(language="grcuda", string=f"float[{size*size}]")
        self.b_d = polyglot.eval(language="grcuda", string=f"float[{size}]")
        self.x_d = polyglot.eval(language="grcuda", string=f"float[{size}]")

        # Build the kernels;
        build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
        self.jacobi_kernel = build_kernel(JACOBI_KERNEL, "JacobiIter", "sint32, const pointer, const pointer, sint32, sint32, const pointer, pointer")
        self.merge_kernel = build_kernel(MERGE_KERNEL, "mergeResults","sint32, sint32, sint32,  pointer, const pointer, const pointer")


    @time_phase("initialization")
    def init(self):
        for i in range(self.size):
            self.b_d[i] = 3.0
            self.x_d[i] = 0.0
            
            for j in range(self.size):
                if (i == j-1):
                    self.a_d[i+j*self.size] = -1.0
                elif ( j == i):
                    self.a_d[i+j*self.size] = 2.0
                elif ( j == i+1):
                    self.a_d[i+j*self.size] = -1.0
                else:
                    self.a_d[i+j*self.size] = 0.0
        self.b_d[self.size-1] = self.size + 1



    @time_phase("reset_result")
    def reset_result(self) -> None:
        for i in range(self.size):
            self.x_d[i] = 0.0

    def execute(self) -> object:

        # Call the kernels;
        start_comp = System.nanoTime()
        start = System.nanoTime()

        for it in range(self.ITER):
            offset = 0
            section = 0
            for g in range(self.NGPU):
                offset = self.size/self.NGPU * g
                section = self.size/self.NGPU * (g+1)
                self.execute_phase(f"jacobiIter_{g}", self.jacobi_kernel(1024,32), self.size, self.a_d, self.x_d, int(offset), int(section), self.b_d, self.x_result_d[g])
            
            offset = self.size/self.NGPU
            self.execute_phase(f"mergeResult", self.merge_kernel(1024, 32), self.size, self.NGPU, int(offset), self.x_d, self.x_result_d[0], self.x_result_d[1])
        if self.time_phases:
            start = System.nanoTime()
        
        result_execution = self.x_d[0]


        end = System.nanoTime()
        if self.time_phases:
            self.benchmark.add_phase({"name": "sync", "time_sec": (end - start) / 1_000_000_000})
        self.benchmark.add_computation_time((end - start_comp) / 1_000_000_000)

        self.benchmark.add_to_benchmark("gpu_result", self.x_d[0])
        if self.benchmark.debug:
            BenchmarkResult.log_message(f"\tgpu result: {self.x_d[0]}")




    def cpu_validation(self, gpu_result: object, reinit: bool) -> None:
        
        x = [0.0 for x in range(self.size)]
        b = [3.0 for x in range(self.size)]
        b[self.size-1] = self.size+1
        x_res = [0.0 for x in range(self.size)]

        for it in range(self.ITER):
            for i in range(self.size):
                sigma = 0
                for j in range(self.size):
                    if (j!=i) :
                        sigma = sigma + self.a_d[i*self.size + j] * x[j]
            
                x_res[i] = (b[i]-sigma)/self.a_d[i*self.size + i]

            for k in range(self.size):
                x[k] = x_res[k]

        for i in range(self.size):
            if(x[i] != self.x_d[i]):
                print("x :" + str(x[i])+" x_d:" + str(self.x_d[i]))

