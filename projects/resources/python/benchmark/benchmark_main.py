import argparse
from distutils.util import strtobool

from bench.single_gpu.bench_1 import Benchmark1
from bench.single_gpu.bench_2 import Benchmark2
from bench.single_gpu.bench_3 import Benchmark3
from bench.single_gpu.bench_4 import Benchmark4
from bench.single_gpu.bench_5 import Benchmark5
from bench.single_gpu.bench_6 import Benchmark6
from bench.single_gpu.bench_72 import Benchmark7
from bench.single_gpu.bench_8 import Benchmark8
from bench.single_gpu.bench_9 import Benchmark9
from bench.single_gpu.bench_10 import Benchmark10
from bench.single_gpu.bench_11 import Benchmark11
from bench.multi_gpu.bench_1 import Benchmark1M
from bench.multi_gpu.bench_5 import Benchmark5M
from bench.multi_gpu.bench_6 import Benchmark6M
from bench.multi_gpu.bench_9 import Benchmark9M
from bench.multi_gpu.bench_11 import Benchmark11M
from benchmark_result import BenchmarkResult

##############################
##############################

# Benchmark settings;
benchmarks = {
    # Single GPU;
    "b1": Benchmark1,
    "b2": Benchmark2,
    "b3": Benchmark3,
    "b4": Benchmark4,
    "b5": Benchmark5,
    "b6": Benchmark6,
    "b7": Benchmark7,
    "b8": Benchmark8,
    "b9": Benchmark9,
    "b10": Benchmark10,
    "b11": Benchmark11,
    # Multi GPU;
    "b1m": Benchmark1M,
    "b5m": Benchmark5M,
    "b6m": Benchmark6M,
    "b9m": Benchmark9M,
    "b11m": Benchmark11M,
}

num_elem = {
    # Single GPU;
    "b1": [100],
    "b2": [100],
    "b3": [100],
    "b4": [100],
    "b5": [100],
    "b6": [100],
    "b7": [100],
    "b8": [100],
    "b9": [100],
    "b10": [100],
    "b11": [100],
    # Multi GPU;
    "b1m": [100],
    "b5m": [100],
    "b6m": [100],
    "b9m": [100],
    "b11m": [100],
}

policies = {
    # Single GPU;
    "b1": ["default"],
    "b2": ["default"],
    "b3": ["default"],
    "b4": ["default"],
    "b5": ["default"],
    "b6": ["default"],
    "b7": ["default"],
    "b8": ["default"],
    "b9": ["default"],
    "b10": ["default"],
    "b11": ["default"],
    # Multi GPU;
    "b1m": ["default"],
    "b5m": ["default"],
    "b6m": ["default"],
    "b9m": ["default"],
    "b11m": ["default"],
}

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="measure GrCUDA execution time")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-o", "--output_path", metavar="path/to/output.json",
                        help="Path to the file where results will be stored")
    parser.add_argument("--realloc", metavar="[True|False]", type=lambda x: bool(strtobool(x)), nargs="*",
                        help="If True, allocate new memory and rebuild the GPU code at each iteration")
    parser.add_argument("--reinit", metavar="[True|False]", type=lambda x: bool(strtobool(x)), nargs="*",
                        help="If True, re-initialize the values used in each benchmark at each iteration")
    parser.add_argument("-c", "--cpu_validation", action="store_true", dest="cpu_validation",
                        help="Validate the result of each benchmark using the CPU")
    parser.add_argument("--no_cpu_validation", action="store_false", dest="cpu_validation",
                        help="Validate the result of each benchmark using the CPU")
    parser.add_argument("-b", "--benchmark", nargs="*",
                        help="If present, run the benchmark only for the specified kernel")
    parser.add_argument("--execP",
                        help="If present, run the benchmark only with the selected execution policy")
    parser.add_argument("--depeP",
                        help="If present, run the benchmark only with the selected dependency policy")
    parser.add_argument("--new_stream",
                        help="If present, run the benchmark only with the selected new stream policy")
    parser.add_argument("--parent_stream",
                        help="If present, run the benchmark only with the selected parent stream policy")
    parser.add_argument("--heuristic",
                        help="If present and parent policy is data aware, run the benchmark only with the selected heuristic")
    parser.add_argument("--memAdviser",
                        help="If present run the benchmark only with the selected memory adviser")
    parser.add_argument("--prefetch",
                        help="If present run the benchmark only with the selected prefetcher")
    parser.add_argument("-n", "--size", metavar="N", type=int, nargs="*",
                        help="Override the input data size used for the benchmarks")
    parser.add_argument("--numGPU", metavar="N", type=int, nargs="*",
                        help="Number of GPU employed for computation")
    parser.add_argument("--block_size_1d", metavar="N", type=int, nargs="*",
                        help="Number of threads per block when using 1D kernels")
    parser.add_argument("--block_size_2d", metavar="N", type=int, nargs="*",
                        help="Number of threads per block when using 2D kernels")
    parser.add_argument("-g", "--number_of_blocks", metavar="N", type=int, nargs="?",
                        help="Number of blocks in the computation")
    parser.add_argument("-r", "--random", action="store_true",
                        help="Initialize benchmarks randomly whenever possible")
    parser.add_argument("--strAttach", action="store_true",
                        help="Force stream attachment")
    parser.add_argument("--timing", action="store_true",
                        help="Measure the execution time of each kernel")
    parser.add_argument("-p", "--time_phases", action="store_true",
                        help="Measure the execution time of each phase of the benchmark;"
                             " note that this introduces overheads, and might influence the total execution time")
    parser.add_argument("--nvprof", action="store_true",
                        help="If present, enable profiling when using nvprof."
                             " For this option to have effect, run graalpython using nvprof, with flag '--profile-from-start off'")
    parser.set_defaults(cpu_validation=BenchmarkResult.DEFAULT_CPU_VALIDATION)

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    output_path = args.output_path if args.output_path else ""
    realloc = args.realloc if args.realloc else [BenchmarkResult.DEFAULT_REALLOC]
    reinit = args.reinit if args.reinit else [BenchmarkResult.DEFAULT_REINIT]
    random_init = args.random if args.random else BenchmarkResult.DEFAULT_RANDOM_INIT
    cpu_validation = args.cpu_validation
    time_phases = args.time_phases
    nvprof_profile = args.nvprof
    timing = args.timing
    prefetch = args.prefetch 
    str_attach = args.strAttach
    nstr_policy = args.new_stream
    pstr_policy = args.parent_stream
    heuristic = args.heuristic
    numGPU = args.numGPU if args.numGPU else [BenchmarkResult.DEFAULT_NUM_GPU]
    exec_policy = args.execP if args.execP else BenchmarkResult.DEFAULT_EXEC_POLICY
    dep_policy = args.depeP if args.depeP else BenchmarkResult.DEFAULT_DEPE_POLICY
    mem_advise = args.memAdviser if args.memAdviser else BenchmarkResult.DEFAULT_MEM_ADVISE
    
    # Create a new benchmark result instance;
    benchmark_res = BenchmarkResult(debug=debug, num_iterations=num_iter, output_path=output_path,
                                    cpu_validation=cpu_validation, random_init=random_init)
    if benchmark_res.debug:
        BenchmarkResult.log_message(f"using CPU validation: {cpu_validation}")

    if args.benchmark:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only benchmark: {args.benchmark}")
        benchmarks = {b: benchmarks[b] for b in args.benchmark}

    # if args.policy:
    #     if benchmark_res.debug:
    #         BenchmarkResult.log_message(f"using only type: {args.policy}")
    #     policies = {n: [args.policy] for n in policies.keys()}

    if args.size:
        if benchmark_res.debug:
            BenchmarkResult.log_message(f"using only size: {args.size}")
        num_elem = {n: args.size for n in num_elem.keys()}

    # Setup the block size for each benchmark;
    block_sizes = BenchmarkResult.create_block_size_list(args.block_size_1d, args.block_size_2d)
    number_of_blocks = args.number_of_blocks
    if (args.block_size_1d or args.block_size_2d) and benchmark_res.debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes}")
    if number_of_blocks:
        BenchmarkResult.log_message(f"using number of blocks: {number_of_blocks}")

    # Execute each test;
    for b_name, b in benchmarks.items():
        benchmark = b(benchmark_res, nvprof_profile=nvprof_profile)
        for p in policies[b_name]:
            for n in num_elem[b_name]:
                prevent_reinit = False
                for re in realloc:
                    for ri in reinit:
                        for block_size in block_sizes:
                            for i in range(num_iter):
                                benchmark.run(num_iter=i, size=n, numGPU=numGPU[0], block_size=block_size, exec_policy=exec_policy,
                                          dep_policy=dep_policy, nstr_policy=nstr_policy, pstr_policy=pstr_policy, heuristic=heuristic,
                                          mem_advise=mem_advise, prefetch=prefetch, str_attach=str_attach, timing=timing,
                                          realloc=re, reinit=ri, time_phases=time_phases, prevent_reinit=prevent_reinit,
                                          number_of_blocks=number_of_blocks)
                                prevent_reinit = True
                            # Print the summary of this block;
                            if benchmark_res.debug:
                                benchmark_res.print_current_summary(name=b_name, size=n, numGPU=numGPU[0], block_size=block_size, exec_policy=exec_policy,
                                          dep_policy=dep_policy, nstr_policy=nstr_policy, pstr_policy=pstr_policy, heuristic=heuristic,
                                          mem_advise=mem_advise, prefetch=prefetch, str_attach=str_attach, timing=timing,
                                          realloc=re, reinit=ri, time_phases=time_phases, num_blocks=number_of_blocks, skip=3)
