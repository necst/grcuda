# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

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

import os
from datetime import datetime
import json
import numpy as np

class BenchmarkResult:

    DEFAULT_RES_FOLDER = "../../../../grcuda-data/results/scheduling_multi_gpu"
    DEFAULT_NUM_ITER = 20
    DEFAULT_DEBUG = True
    DEFAULT_CPU_VALIDATION = False
    DEFAULT_REALLOC = False
    DEFAULT_REINIT = True
    DEFAULT_RANDOM_INIT = False

    def __init__(self,
                 num_iterations: int = DEFAULT_NUM_ITER,
                 cpu_validation: bool = DEFAULT_CPU_VALIDATION,
                 debug: bool = DEFAULT_DEBUG,
                 random_init: bool = DEFAULT_RANDOM_INIT,
                 output_path: str = "",
                 ):
        self.debug = debug
        self.random_init = random_init
        self.num_iterations = num_iterations
        self.cpu_validation = cpu_validation
        self._results = {"num_iterations": num_iterations,
                         "cpu_validation": cpu_validation,
                         "random_init": random_init,
                         "benchmarks": {}}
        # Used to store the results of the benchmark currently being executed;
        self._dict_current = {}

        # If true, use the provided output path as it is, without adding extensions or creating folders;
        self._output_path = output_path if output_path else self.default_output_file_name()
        output_folder = os.path.dirname(output_path) if output_path else self.DEFAULT_RES_FOLDER
        if not os.path.exists(output_folder):
            if self.debug:
                BenchmarkResult.log_message(f"creating result folder: {output_folder}")
                os.makedirs(output_folder)
        if self.debug:
            BenchmarkResult.log_message(f"storing results in {self._output_path}")

    @staticmethod
    def create_block_size_key(block_size: dict) -> str:
        return f"{block_size['block_size_1d']},{block_size['block_size_2d']}"

    def default_output_file_name(self) -> str:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"{output_date}_{self.num_iterations}.json"
        return os.path.join(self.DEFAULT_RES_FOLDER, file_name)

    def start_new_benchmark(self, name: str, size: int, numGPU: int,
                            block_size: dict, num_blocks: int, exec_policy: str,
                            dep_policy: str, nstr_policy: str, pstr_policy: str,
                            heuristic: str, mem_advise: str, prefetch: str,
                            str_attach: str, timing: bool, iteration: int, 
                            time_phases: bool, realloc: bool, reinit: bool) -> None:
        """
        Benchmark results are stored in a nested dictionary with the following structure.
        self.results["benchmarks"]->{name}->{size}->{numGPU}->{num_blocks}->{exec_policy}->{dep_policy}->
        {nstr_policy}->{pstr_policy}->{heuristic}->{prefetch}->{str_attach}->{timing}->{realloc}->{reinit}->{block_size}_{actual result}

        :param name: name of the benchmark
        :param size: size of the input data
        :param numGPU: number of GPU used in the benchmark
        :param num_blocks: number of GPU thread blocks 
        :param exec_policy: current execution policy used in the benchmark
        :param dep_policy: current dependency policy used in the benchmark
        :param nstr_policy: current new stream policy used in the benchmark
        :param pstr_policy: current parent stream policy used in the benchmark
        :param heuristic: current choose device heuristic used in the benchmark
        .param prefetch: current prefetcher used in the benchmark
        :param str_attach: if stream attachment are forced
        :param timing: if kernel timing is enabled
        :param realloc: if reallocation is performed
        :param reinit: if re-initialization is performed
        :param block_size: dictionary that specifies the number of threads per block
        :param iteration: current iteration
        :param time_phases: if True, measure the execution time of each phase of the benchmark.
         Note that this introduces overheads, and might influence the total execution time
        """

        # 1. Benchmark name;
        if name in self._results["benchmarks"]:
            dict_size = self._results["benchmarks"][name]
        else:
            dict_size = {}
            self._results["benchmarks"][name] = dict_size
        # 2. Input size;
        if size in dict_size:
            dict_nGPU = dict_size[size]
        else:
            dict_nGPU = {}
            dict_size[size] = dict_nGPU
        # 3. Number of GPUs;
        if numGPU in dict_nGPU:
            dict_nblock = dict_nGPU[numGPU]
        else:
            dict_nblock = {}
            dict_nGPU[numGPU] = dict_nblock
        # 4. Number of blocks; 
        if num_blocks in dict_nblock:
            dict_exeP = dict_nblock[num_blocks]
        else:
            dict_exeP = {}
            dict_nblock[num_blocks] = dict_exeP
        # 5. Execution policy
        if exec_policy in dict_exeP:
            dict_depP = dict_exeP[exec_policy]
        else:
            dict_depP = {}
            dict_exeP[exec_policy] = dict_depP
        # 6. Dependency policy
        if dep_policy in dict_depP:
            dict_nstr = dict_depP[dep_policy]
        else:
            dict_nstr = {}
            dict_depP[dep_policy] = dict_nstr
        # 7. New stream policy
        if nstr_policy in dict_nstr:
            dict_pstr = dict_nstr[nstr_policy]
        else:
            dict_pstr = {}
            dict_nstr[nstr_policy] = dict_pstr
        # 8. Parent stream policy
        if pstr_policy in dict_pstr:
            dict_heur = dict_pstr[pstr_policy]
        else:
            dict_heur = {}
            dict_pstr[pstr_policy] = dict_heur
        # 9. Choose Device Heuristic
        if heuristic in dict_heur:
            dict_prefetch = dict_heur[heuristic]
        else:
            dict_prefetch = {}
            dict_heur[heuristic] = dict_prefetch
        # 10. Prefetcher 
        if prefetch in dict_prefetch:
            dict_sAtt = dict_prefetch[prefetch]
        else:
            dict_sAtt = {}
            dict_prefetch[prefetch] = dict_sAtt
        # 11. Stream Attachment
        if str_attach in dict_sAtt:
            dict_time = dict_sAtt[str_attach]
        else:
            dict_time = {}
            dict_sAtt[str_attach] = dict_time
        # 12. Kernel timing
        if timing in dict_time:
            dict_realloc = dict_time[timing]
        else:
            dict_realloc = {}
            dict_time[timing] = dict_realloc
        # 13. Realloc options;
        if realloc in dict_realloc:
            dict_reinit = dict_realloc[realloc]
        else:
            dict_reinit = {}
            dict_realloc[realloc] = dict_reinit
        # 14. Reinit options;
        if reinit in dict_reinit:
            dict_block = dict_reinit[reinit]
        else:
            dict_block = {}
            dict_reinit[reinit] = dict_block
        # 15. Block size options;
        self._dict_current = {"phases": [], "iteration": iteration, "time_phases": time_phases}
        if BenchmarkResult.create_block_size_key(block_size) in dict_block:
            dict_block[BenchmarkResult.create_block_size_key(block_size)] += [self._dict_current]
        else:
            dict_block[BenchmarkResult.create_block_size_key(block_size)] = [self._dict_current]

        if self.debug:
            BenchmarkResult.log_message(
                f"starting benchmark={name}, iter={iteration + 1}/{self.num_iterations}, size={size}, numGPU={numGPU}, num_blocks={num_blocks}, "
                f"exec_policy={exec_policy}, dep_policy={dep_policy}, nstr_policy={nstr_policy}, pstr_policy={pstr_policy}, "
                f"heuristic={heuristic}, realloc={realloc}, reinit={reinit}, prefetch={prefetch}, str_attach={str_attach}, "
                f"block_size={BenchmarkResult.create_block_size_key(block_size)}, timing={timing}, time_phases={time_phases}")

    def add_to_benchmark(self, key: str, message: object) -> None:
        """
        Add an key-value pair in the current benchmark entry, e.g. ("allocation_time_ms", 10);
        :param key: the key used to identify the message, e.g. "allocation_time_ms"
        :param message: the value of the message, possibly a string, a number,
        or any object that can be represented as JSON
        """
        self._dict_current[key] = message

    def add_total_time(self, total_time: float) -> None:
        """
        Add to the current benchmark entry the execution time of a benchmark iteration,
         and compute the amount of overhead w.r.t. the single phases
        :param total_time: execution time of the benchmark iteration
        """
        self._dict_current["total_time_sec"] = total_time

        # Keep only phases related to GPU computation;
        blacklisted_phases = ["allocation", "initialization", "reset_result"]
        filtered_phases = [x for x in self._dict_current["phases"] if x["name"] not in blacklisted_phases]
        tot_time_phases = sum([x["time_sec"] if "time_sec" in x else 0 for x in filtered_phases])
        self._dict_current["overhead_sec"] = total_time - tot_time_phases
        self._dict_current["computation_sum_phases_sec"] = tot_time_phases
        if self.debug:
            BenchmarkResult.log_message(f"\ttotal execution time: {total_time:.4f} sec," +
                                        f" overhead: {total_time - tot_time_phases:.4f} sec, " +
                                        f" computation: {self._dict_current['computation_sec']:.4f} sec")

    def add_computation_time(self, computation_time: float) -> None:
        """
        Add to the current benchmark entry the GPU computation time of the benchmark iteration
        :param computation_time: execution time of the GPU computation in the benchmark iteration, in seconds
        """
        self._dict_current["computation_sec"] = computation_time

    def add_phase(self, phase: dict) -> None:
        """
        Add a dictionary that represents a phase of a benchmark, to provide fine-grained profiling;
        :param phase: a dictionary that contains information about a phase of the algorithm,
        with information such as name, duration, description, etc...
        """
        self._dict_current["phases"] += [phase]
        if self.debug and "name" in phase and "time_sec" in phase:
            BenchmarkResult.log_message(f"\t\t{phase['name']}: {phase['time_sec']:.4f} sec")

    def print_current_summary(self, name: str, size: int, numGPU: int,
                            num_blocks: int, exec_policy: str,
                            dep_policy: str, nstr_policy: str, pstr_policy: str,
                            heuristic: str, mem_advise: str, prefetch: str,
                            str_attach: str, timing: bool, block_size: dict,
                            time_phases: bool, realloc: bool, reinit: bool, skip: int = 0) -> None:
        """
        Print a summary of the benchmark with the provided settings;

        :param name: name of the benchmark
        :param size: size of the input data
        :param numGPU: number of GPU used in the benchmark
        :param num_blocks: number of GPU thread blocks 
        :param exec_policy: current execution policy used in the benchmark
        :param dep_policy: current dependency policy used in the benchmark
        :param nstr_policy: current new stream policy used in the benchmark
        :param pstr_policy: current parent stream policy used in the benchmark
        :param heuristic: current choose device heuristic used in the benchmark
        .param prefetch: current prefetcher used in the benchmark
        :param str_attach: if stream attachment are forced
        :param timing: if kernel timing is enabled
        :param realloc: if reallocation is performed
        :param reinit: if re-initialization is performed
        :param block_size: dictionary that specifies the number of threads per block
        :param time_phases: if True, measure the execution time of each phase of the benchmark.
        :param skip: skip the first N iterations when computing the summary statistics
        """
        try:
            results_filtered = self._results["benchmarks"][name][size][numGPU][num_blocks][exec_policy][dep_policy][nstr_policy][pstr_policy][heuristic][prefetch][str_attach][timing][realloc][reinit][BenchmarkResult.create_block_size_key(block_size)]
        except KeyError as e:
            results_filtered = []
            BenchmarkResult.log_message(f"WARNING: benchmark with signature"
                                        f" [{name}][{size}][{numGPU}][{num_blocks}][{exec_policy}][{dep_policy}][{nstr_policy}][{pstr_policy}][{heuristic}][{prefetch}][{str_attach}][{timing}][{realloc}][{reinit}][{BenchmarkResult.create_block_size_key(block_size)}] not found, exception {e}")
        # Retrieve execution times;
        exec_times = [x["total_time_sec"] for x in results_filtered][skip:]
        mean_time = np.mean(exec_times) if exec_times else np.nan
        std_time = np.std(exec_times) if exec_times else np.nan

        comp_exec_times = [x["computation_sec"] for x in results_filtered][skip:]
        comp_mean_time = np.mean(comp_exec_times) if comp_exec_times else np.nan
        comp_std_time = np.std(comp_exec_times) if comp_exec_times else np.nan

        BenchmarkResult.log_message(f"summary of benchmark={name}, size={size}, numGPU={numGPU}, " +
                                    f" num_blocks={num_blocks}, exec_policy={exec_policy}, dep_policy={dep_policy}, " +
                                    f" nstr_policy={nstr_policy}, pstr_policy={pstr_policy}, heuristic={heuristic}, " + 
                                    f" prefetch={prefetch}, str_attach={str_attach}, timing={timing}, " +
                                    f" realloc={realloc}, reinit={reinit}, block_size=({BenchmarkResult.create_block_size_key(block_size)});" +
                                    f" mean total time={mean_time:.4f}±{std_time:.4f} sec;" +
                                    f" mean computation time={comp_mean_time:.4f}±{comp_std_time:.4f} sec")

    def save_to_file(self) -> None:
        with open(self._output_path, "w+") as f:
            json_result = json.dumps(self._results, ensure_ascii=False, indent=4)
            f.write(json_result)

    @staticmethod
    def create_block_size_list(block_size_1d, block_size_2d) -> list:
        """
        Utility method used to create a list of dictionaries {"block_size_1d": N, "block_size_2d": N} to pass to the benchmark execution.
        The method ensures that the output is a valid list of tuples even if one list is missing or if they have different lengths
        """
        if (not block_size_1d) and block_size_2d:  # Only 2D block size;
            block_size = [{"block_size_2d": b} for b in block_size_2d]
        elif (not block_size_2d) and block_size_1d:  # Only 1D block size;
            block_size = [{"block_size_1d": b} for b in block_size_1d]
        elif block_size_1d and block_size_2d:  # Both 1D and 2D size;
            # Ensure they have the same size;
            if len(block_size_2d) > len(block_size_1d):
                block_size_1d = block_size_1d + [block_size_1d[-1]] * (len(block_size_2d) - len(block_size_1d))
            elif len(block_size_1d) > len(block_size_2d):
                block_size_2d = block_size_2d + [block_size_2d[-1]] * (len(block_size_1d) - len(block_size_2d))
            block_size = [{"block_size_1d": x[0], "block_size_2d": x[1]} for x in zip(block_size_1d, block_size_2d)]
        else:
            block_size = [{}]
        return block_size

    @staticmethod
    def log_message(message: str) -> None:
        date = datetime.now()
        date_str = date.strftime("%Y-%m-%d-%H-%M-%S-%f")
        print(f"[{date_str} grcuda-python] {message}")
