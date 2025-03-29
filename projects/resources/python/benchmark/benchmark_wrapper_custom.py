# Copyright (c) 2025 NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import subprocess
import time
import os
from datetime import datetime
from benchmark_result import BenchmarkResult
from pathlib import Path

##############################
##############################
GPU = "GPU_NAME" ## not relevant, it is just used for the output files
BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph.csv"


# Benchmark settings;
DEFAULT_NUM_BLOCKS = 640

HEAP_SIZE = 470

benchmarks = [
    "b1m",
    "b5m",
    "b6m",
    "b9m",
    "b11m",
]

num_elem= {
    "b1m": [
      160000,
      500000,
      950000
    ],
    "b5m": [
      10000,
      21000,
      35000
    ],
    "b6m": [
      1000,
      1400,
      1800
    ],
    "b9m": [
      2000,
      4000,
      6000
    ],
    "b11m": [
      2000,
      4000,
      6000
    ]
}


block_dim_dict = {
    "b1m": 64,
    "b5m": 64,
    "b6m": 64,
    "b9m": 64,
    "b11m": 64,
}


exec_policies = ["async"]

dependency_policies = ["with-const"]  #, "no-const"]

new_stream_policies = ["always-new"]  #, "reuse"]

parent_stream_policies = ["multigpu-disjoint"]  # ["same-as-parent", "disjoint", "multigpu-early-disjoint", "multigpu-disjoint"]

choose_device_policies = ["round-robin", "stream-aware", "min-transfer-size", "minmax-transfer-time"]  # ["single-gpu", "round-robin", "stream-aware", "min-transfer-size", "minmin-transfer-time", "minmax-transfer-time"]

memory_advise = ["none"]

prefetch = ["false"]

stream_attach =  [False]

time_computation = [False]

num_gpus = [2]

block_sizes1d_dict = {
    "b1m": 32,
    "b5m": 1024,
    "b6m": 32,
    "b9m": 32,
    "b11m": 256,
}

block_sizes2d_dict = {
    "b1m": 8,
    "b5m": 8,
    "b6m": 8,
    "b9m": 8,
    "b11m": 8,
}

##############################
##############################

GRAALPYTHON_CMD = "graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --experimental-options " \
                  "--grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveNewStreamPolicy={} " \
                  "--grcuda.NumberOfGPUs={} --grcuda.RetrieveParentStreamPolicy={} " \
                  "--grcuda.DeviceSelectionPolicy={} --grcuda.MemAdvisePolicy={} --grcuda.InputPrefetch={} --grcuda.BandwidthMatrix={} {} {} " \
                  "benchmark_main.py -i {} -n {} -g {} --number_of_gpus {} --reinit false --realloc false " \
                  "-b {} --block_size_1d {} --block_size_2d {} --execution_policy {} --dependency_policy {} --new_stream {} "\
                  "--parent_stream {} --device_selection {} --memory_advise_policy {} --prefetch {} --no_cpu_validation {} {} {} {} -o {}"

def execute_grcuda_benchmark(benchmark, size, num_gpus, block_sizes, exec_policy, dependency_policy, new_stream_policy,
                      parent_stream_policy, choose_device_policy, memory_advise, prefetch, num_iter, bandwidth_matrix, time_phases, debug, stream_attach=False,
                      time_computation=False, num_blocks=DEFAULT_NUM_BLOCKS, output_date=None, mock=False):
    if debug:
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={benchmark}, size={n},"
                                    f"gpus={num_gpus}, "
                                    f"block-sizes={block_sizes}, "
                                    f"num-blocks={num_blocks}, "
                                    f"exec-policy={exec_policy}, "
                                    f"dependency-policy={dependency_policy}, "
                                    f"new-stream-policy={new_stream_policy}, "
                                    f"parent-stream-policy={parent_stream_policy}, "
                                    f"choose-device-policy={choose_device_policy}, "
                                    f"mem-advise={memory_advise}, "
                                    f"prefetch={prefetch}, "
                                    f"stream-attachment={stream_attach}, "
                                    f"time-computation={time_computation}, "
                                    f"bandwidth-matrix={bandwidth_matrix}, "
                                    f"time-phases={time_phases}")
        BenchmarkResult.log_message("")

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{output_date}_{benchmark}_{size}_{num_gpus}_{num_blocks}_{exec_policy}_{dependency_policy}_" \
                f"{new_stream_policy}_{parent_stream_policy}_{choose_device_policy}_" \
                f"{memory_advise}_{prefetch}_{stream_attach}.json"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_grcuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        if not mock:
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder_path, file_name)
    b1d_size = " ".join([str(b['block_size_1d']) for b in block_sizes])
    b2d_size = " ".join([str(b['block_size_2d']) for b in block_sizes])

    benchmark_cmd = GRAALPYTHON_CMD.format(HEAP_SIZE, exec_policy, dependency_policy, new_stream_policy,
                                           num_gpus, parent_stream_policy, choose_device_policy, memory_advise, prefetch, bandwidth_matrix,
                                           "--grcuda.ForceStreamAttach" if stream_attach else "", 
                                           "--grcuda.EnableComputationTimers" if time_computation else "",
                                           num_iter, size, num_blocks, num_gpus, benchmark, b1d_size, b2d_size, exec_policy, dependency_policy,
                                           new_stream_policy, parent_stream_policy, choose_device_policy, memory_advise, prefetch,
                                           "-d" if debug else "",
                                           "-p" if time_phases else "", 
                                           "--force_stream_attach" if stream_attach else "", 
                                           "--timing" if time_computation else "",
                                           output_path)    
    if debug:
        BenchmarkResult.log_message(benchmark_cmd)
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
    if not mock:
        start = time.time()
        result = subprocess.run(benchmark_cmd,
                                shell=True,
                                stdout=None, #subprocess.STDOUT,
                                cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/python/benchmark")
        result.check_returncode()
        end = time.time()
        if debug:
            BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start):.2f} seconds")

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrap the GrCUDA benchmark to specify additional settings")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-c", "--cuda_test", action="store_true",
                        help="If present, run performance tests using CUDA")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-g", "--num_blocks", metavar="N", type=int,
                        help="Number of blocks in each kernel, when applicable")
    parser.add_argument("-p", "--time_phases", action="store_true",
                        help="Measure the execution time of each phase of the benchmark;"
                             " note that this introduces overheads, and might influence the total execution time")
    parser.add_argument("-m", "--mock", action="store_true",
                        help="If present, simply print the benchmark CMD without executing it")
    parser.add_argument("--gpus", metavar="N", type=int, nargs="*",
                        help="Specify the maximum number of GPUs to use in the computation")

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    use_cuda = args.cuda_test
    time_phases = args.time_phases
    num_blocks = args.num_blocks
    mock = args.mock
    gpus = args.gpus

    if gpus is not None:
        num_gpus = gpus

    if debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes1d_dict} {block_sizes2d_dict}; using low-level CUDA benchmarks: {use_cuda}")

    def tot_benchmark_count():
        tot = 0
        if use_cuda:
            for b in benchmarks:
                for e in cuda_exec_policies:
                    if e == "sync":
                        tot += len(num_elem[b]) * len(prefetch) * len(stream_attach) 
                    else:
                        tot += len(num_elem[b]) * len(prefetch) * len(num_gpus) * len(stream_attach) 
        else:
            for b in benchmarks:
                for e in exec_policies:
                    if e == "sync":
                        tot += len(num_elem[b]) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation)
                    else:
                        for n in num_gpus:
                            if n == 1:
                                tot += len(num_elem[b]) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation)
                            else:
                                tot += len(num_elem[b]) * len(dependency_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(choose_device_policies) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation)
        return tot

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            for exec_policy in exec_policies:            # GrCUDA Benchmarks;
                if exec_policy == "sync":
                    dp = [dependency_policies[0]]
                    nsp = [new_stream_policies[0]]
                    psp = [parent_stream_policies[0]]
                    cdp = [choose_device_policies[0]]
                    ng = [1]
                else:
                    dp = dependency_policies
                    nsp = new_stream_policies
                    psp = parent_stream_policies
                    cdp = choose_device_policies
                    ng = num_gpus
                for num_gpu in ng:
                    if exec_policy == "async" and num_gpu == 1:
                        dp = [dependency_policies[0]]
                        nsp = [new_stream_policies[0]]
                        psp = [parent_stream_policies[0]]
                        cdp = [choose_device_policies[0]]
                    else:
                        dp = dependency_policies
                        nsp = new_stream_policies
                        psp = parent_stream_policies
                        cdp = choose_device_policies
                    for m in memory_advise:
                        for p in prefetch:
                            for s in stream_attach:
                                for t in time_computation:
                                    # Select the correct connection graph;
                                    BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph.csv"
                                    for dependency_policy in dp:
                                        for new_stream_policy in nsp:
                                            for parent_stream_policy in psp:
                                                for choose_device_policy in cdp:
                                                    nb = num_blocks if num_blocks else block_dim_dict[b]
                                                    block_sizes = BenchmarkResult.create_block_size_list([block_sizes1d_dict[b]], [block_sizes2d_dict[b]])
                                                    execute_grcuda_benchmark(b, n, num_gpu, block_sizes,
                                                        exec_policy, dependency_policy, new_stream_policy, parent_stream_policy, choose_device_policy, 
                                                        m, p, num_iter, BANDWIDTH_MATRIX, time_phases, debug, s, t, nb, output_date=output_date, mock=mock)
                                                    i += 1 

