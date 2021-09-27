import argparse
import subprocess
import time
import os
from datetime import datetime
from benchmark_result import BenchmarkResult
from benchmark_main import create_block_size_list
from java.lang import System

##############################
##############################

DEFAULT_NUM_BLOCKS = 32  # GTX 960, 8 SM
#DEFAULT_NUM_BLOCKS = 448  # P100, 56 SM
#DEFAULT_NUM_BLOCKS = 176  # GTX 1660 Super, 22 SM

HEAP_SIZE = 26 
#HEAP_SIZE = 140 # P100

# Benchmark settings;
benchmarks = [
    # "b1",
    # "b5",
    # "b6",
    # "b7",
    # "b8",
    # "b10",
    "b11",
]

# GTX 960
num_elem = {
    "b1": [1000],#[20_000_000, 60_000_000, 80_000_000, 100_000_000, 120_000_000],
    "b5": [2000],#[2_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000],
    "b6": [200],#[200_000, 500_000, 800_000, 1_000_000, 1_200_000],
    "b7": [4000],#[4_000_000, 7_000_000, 10_000_000, 15_000_000, 20_000_000], 
    "b8": [800],#[1600, 2400, 3200, 4000, 4800],
    "b10": [300],#[3000, 4000, 5000, 6000, 7000],
    "b11": [1000],
}

# P100
#num_elem = {
#     "b1": [120_000_000, 200_000_000, 500_000_000, 600_000_000, 700_000_000],
#     "b5": [12_000_000, 20_000_000, 50_000_000, 60_000_000, 70_000_000],
#     "b6": [1_200_000, 2_000_000, 4_000_000, 5_000_000, 6_000_000],
#     "b7": [20_000_000, 40_000_000, 60_000_000, 100_000_000, 140_000_000],
#     "b8": [4800, 8000, 10000, 12000, 16000],
#     "b10": [7000, 10000, 12000, 14000, 16000],
#}

# GTX 1660 Super
# num_elem = {
#     "b1": [60_000_000, 80_000_000, 100_000_000, 120_000_000, 200_000_000],
#     "b5": [6_000_000, 8_000_000, 10_000_000, 12_000_000, 20_000_000],
#     "b6": [500_000, 800_000, 1_000_000, 1_200_000, 2_000_000],
#     "b7": [7_000_000, 10_000_000, 15_000_000, 20_000_000, 40_000_000],
#     "b8": [3200, 4000, 4800, 8000, 10000],
#     "b10": [6000, 7000, 10000, 12000, 14000],
# }

cuda_exec_policies = ["default", "sync", "cudagraph", "cudagraphmanual", "cudagraphsingle"]

exec_policies = ["default"]#, "sync"]

dependency_policies = ["default"]#, "with_const"]

new_stream_policies = ["always_new"]#, "fifo"]

parent_stream_policies = ["data_aware"]#, "default", "disjoint", "disjoint_data_aware", "stream_aware"]

choose_device_heuristics = ["data_locality"]#, "best_transfer_time_min", "best_transfer_time_max"]

memAdvisers = ["none"]#, "read_mostly", "preferred"]

prefetches = ["none"]#, "default", "sync"]

streamAttachs =  [False]#, True]

timeComputes = [False]#, True]

numGPUs = [1]#, 2]

block_sizes_1d = [32]#[32, 128, 256, 1024]
block_sizes_2d = [8]#[8, 8, 8, 8]

# 960
block_dim_dict = {
    "b1": DEFAULT_NUM_BLOCKS,
    "b5": DEFAULT_NUM_BLOCKS,
    "b6": 32,
    "b7": DEFAULT_NUM_BLOCKS,
    "b8": 12,
    "b10": 16,
    "b11": DEFAULT_NUM_BLOCKS,
}

# P100
# block_dim_dict = {
#     "b1": DEFAULT_NUM_BLOCKS,
#     "b5": DEFAULT_NUM_BLOCKS,
#     "b6": 64,
#     "b7": DEFAULT_NUM_BLOCKS,
#     "b8": 32,
#     "b10": DEFAULT_NUM_BLOCKS,
# }

# 1660
# block_dim_dict = {
#     "b1": DEFAULT_NUM_BLOCKS,
#     "b5": DEFAULT_NUM_BLOCKS,
#     "b6": 32,
#     "b7": DEFAULT_NUM_BLOCKS,
#     "b8": 16,
#     "b10": DEFAULT_NUM_BLOCKS,
# }

##############################
##############################

CUDA_CMD = "./b -k {} -p {} -n {} -b {} -c {} -t {} -g {} {} {} | tee {}"


def execute_cuda_benchmark(benchmark, size, block_size, exec_policy, num_iter, debug, prefetch=False, num_blocks=DEFAULT_NUM_BLOCKS, output_date=None):
    if debug:
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                    f" block size={block_size}, "
                                    f" prefetch={prefetch}, "
                                    f" num blocks={num_blocks}, "
                                    f" exec policy={exec_policy}")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"cuda_{output_date}_{benchmark}_{exec_policy}_{size}_{block_size['block_size_1d']}_{block_size['block_size_2d']}_{prefetch}_{num_iter}_{num_blocks}.csv"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_cuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        os.mkdir(output_folder_path)
    output_path = os.path.join(output_folder_path, file_name)

    benchmark_cmd = CUDA_CMD.format(benchmark, exec_policy, size, block_size["block_size_1d"],
                                    block_size["block_size_2d"], num_iter, num_blocks, "-r" if prefetch else "", "-a", output_path)
    start = System.nanoTime()
    result = subprocess.run(benchmark_cmd,
                            shell=True,
                            stdout=None,
                            cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/cuda/bin")
    result.check_returncode()
    end = System.nanoTime()
    if debug:
        BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start) / 1_000_000_000:.2f} seconds")


##############################
##############################

GRAALPYTHON_CMD = "graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --experimental-options " \
                  "--grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveNewStreamPolicy={} " \
                  "--grcuda.NumberOfGPUs={} --grcuda.RetrieveParentStreamPolicy={} " \
                  "--grcuda.ChooseDeviceHeuristic={} --grcuda.memAdviseOption={} --grcuda.InputPrefetch={} {} {} " \
                  "benchmark_main.py -i {} -n {} -g {} --reinit false --realloc false " \
                  "-b {} --block_size_1d {} --block_size_2d {} --no_cpu_validation {} {} -o {}"


def execute_grcuda_benchmark(benchmark, size, numGPUs, block_sizes, exec_policy, dependency_policy, new_stream_policy,
                      parent_stream_policy, choose_device_heuristic, memAdviser, prefetch, num_iter, debug, time_phases, streamAttach=False,
                      timeCompute=False, num_blocks=DEFAULT_NUM_BLOCKS, output_date=None):
    if debug:
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={benchmark}, size={n},"
                                    f"num GPUs={numGPUs}, "
                                    f"block sizes={block_sizes}, "
                                    f"num blocks={num_blocks}, "
                                    f"exec policy={exec_policy}, "
                                    f"dependency policy={dependency_policy}, "
                                    f"new stream policy={new_stream_policy}, "
                                    f"parent stream policy={parent_stream_policy}, "
                                    f"choose-device heuristic={choose_device_heuristic}, "
                                    f"mem-advise option={memAdviser}, "
                                    f"prefetch={prefetch}, "
                                    f"stream attachment={streamAttach}, "
                                    f"time computation={timeCompute}"
                                    f"time_phases={time_phases}")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{output_date}_{benchmark}_sz-{size}_{numGPUs}GPUs_nb-{num_blocks}_exe-{exec_policy}_dep-{dependency_policy}_" \
                f"new-{new_stream_policy}_par-{parent_stream_policy}_heu-{choose_device_heuristic}_" \
                f"adv-{memAdviser}_prf-{prefetch}_att-{streamAttach}.json"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_grcuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        os.mkdir(output_folder_path)
    output_path = os.path.join(output_folder_path, file_name)
    b1d_size = " ".join([str(b['block_size_1d']) for b in block_sizes])
    b2d_size = " ".join([str(b['block_size_2d']) for b in block_sizes])

    benchmark_cmd = GRAALPYTHON_CMD.format(HEAP_SIZE, exec_policy, dependency_policy, new_stream_policy,
                                           numGPUs, parent_stream_policy, choose_device_heuristic, memAdviser, prefetch,
                                           "--grcuda.ForceStreamAttach" if streamAttach else "", "--grcuda.TimeComputation" if timeCompute else "",
                                           num_iter, size, num_blocks, benchmark, b1d_size, b2d_size,
                                           "-d" if debug else "",  "-p" if time_phases else "", output_path)
    print(benchmark_cmd)
    start = System.nanoTime()
    result = subprocess.run(benchmark_cmd,
                            shell=True,
                            stdout=None, #subprocess.STDOUT,
                            cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/python/benchmark")
    result.check_returncode()
    end = System.nanoTime()
    if debug:
        BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start) / 1_000_000_000:.2f} seconds")

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

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    use_cuda = args.cuda_test
    time_phases = args.time_phases
    num_blocks = args.num_blocks

    # Setup the block size for each benchmark;
    block_sizes = create_block_size_list(block_sizes_1d, block_sizes_2d)
    if debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes}; using low-level CUDA benchmarks: {use_cuda}")

    def tot_benchmark_count():
        tot = 0
        if use_cuda:
            for b in benchmarks:
                tot += len(num_elem[b]) * len(block_sizes) * len(cuda_exec_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(dependency_policies) * len(prefetches)
        else:
            for b in benchmarks:
                tot += len(num_elem[b]) * len(numGPUs) * len(exec_policies) * len(dependency_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(choose_device_heuristics) * len(memAdvisers) * len(prefetches) * len(streamAttachs) * len(timeComputes)
        return tot

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            if use_cuda:
                # CUDA Benchmarks;
                for exec_policy in cuda_exec_policies:
                    for block_size in block_sizes:
                        for p in prefetch:
                            nb = num_blocks if num_blocks else block_dim_dict[b]
                            execute_cuda_benchmark(b, n, block_size, exec_policy, num_iter, debug, num_blocks=nb, prefetch=p, output_date=output_date)
                            i += 1
            # GrCUDA Benchmarks;
            else:
                for numGPU in numGPUs:
                    for exec_policy in exec_policies:
                        for dependency_policy in dependency_policies:
                            for new_stream_policy in new_stream_policies:
                                for parent_stream_policy in parent_stream_policies:
                                    for choose_device_heuristic in choose_device_heuristics:
                                        for memAdviser in memAdvisers:
                                            for prefetch in prefetches:
                                                for streamAttach in streamAttachs:
                                                    for timeCompute in timeComputes:
                                                        nb = num_blocks if num_blocks else block_dim_dict[b]
                                                        execute_grcuda_benchmark(b, n, numGPU, block_sizes, exec_policy, dependency_policy, new_stream_policy,
                                                                             parent_stream_policy, choose_device_heuristic, memAdviser, prefetch, num_iter, debug, time_phases, streamAttach,
                                                                             timeCompute, nb, output_date=output_date)
                                                        i += 1                                        


                                                        
