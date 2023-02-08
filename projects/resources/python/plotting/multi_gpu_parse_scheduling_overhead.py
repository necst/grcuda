# Copyright (c) 2020, 2021, 2022, NECSTLab, Politecnico di Milano. All rights reserved.

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

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS"" AND ANY
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5 13:29:26 2022

@author: guidowalter.didonato
"""

from copy import deepcopy
import pandas as pd
import json
import os

DEFAULT_OVERHEAD_DIR = "../../../../grcuda-data/results/scheduling_multi_gpu"
# PLOT_DIR = "../../../../grcuda-data/plots/multi_gpu"

ASYNC_POLICY_NAME = "async" 

BENCHMARK_NAMES = {
    "B1M": "VEC",
    "B5M": "B&S",
    "B6M": "ML",
    "B9M": "CG",
    "B11M": "MUL",
    }

POLICY_NAMES = {
    "sync": "SYNC",
    ASYNC_POLICY_NAME: "ASYNC",
    }

def load_overhead_java(input_files: list): #-> pd.DataFrame:
    dictionaries = []
    for file in input_files:
        dictionary = {}
        input_path = os.path.join(DEFAULT_OVERHEAD_DIR, file)
        with open(os.path.abspath(input_path), 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if (stripped_line.startswith('BenchmarkConfig')):
                    bench_config_str = stripped_line[len("BenchmarkConfig"):]
                    bench_config_str = bench_config_str.replace('=', ':')
                    bench_config_str = bench_config_str.replace("'", '"')
                    bench_config_couples = bench_config_str.strip('{}').split(',')
                    bench_config_str = []
                    for couple in bench_config_couples:
                        tmp = couple.split(':')
                        tmp[0] = '"' + tmp[0].strip() + '"'
                        tmp = ': '.join(tmp)
                        bench_config_str.append(tmp)
                    bench_config_str = '{' + ', '.join(bench_config_str) + '}'
                    dictionary = json.loads(bench_config_str)
                elif (stripped_line.startswith('[grcuda::com.nvidia.grcuda.runtime.computation]')):
                    overhead_sec = float(stripped_line.split(' ')[-2])/1000000000
                    dictionary['tot_overhead_sec'] = overhead_sec
                    dictionary['mean_overhead_sec'] = dictionary['tot_overhead_sec']/dictionary['totIter']
                    dictionaries.append(deepcopy(dictionary))
                    dictionary = {}
    df = pd.DataFrame.from_dict(dictionaries)
    df.rename(columns = {'benchmarkName':'benchmark', 'totIter':'total_iterations', 'numGpus':'gpus', 
                            'deviceSelectionPolicy':'device_selection_policy', 'executionPolicy':'exec_policy', 
                            'retrieveParentStreamPolicy':'parent_stream_policy'}, inplace = True)

    df["benchmark"] = df["benchmark"].replace(BENCHMARK_NAMES)
    df["benchmark"] = pd.Categorical(df["benchmark"], list(BENCHMARK_NAMES.values()))
    df["exec_policy"] = df["exec_policy"].replace(POLICY_NAMES)
    df["exec_policy"] = pd.Categorical(df["exec_policy"], list(POLICY_NAMES.values()))

    return df.loc[:,['benchmark', 'total_iterations', 'size', 'gpus', 'exec_policy', 'parent_stream_policy',
                     'device_selection_policy', 'tot_overhead_sec', 'mean_overhead_sec']]

if __name__ == "__main__":
    
    ## V100
    res_list_v100 = [
        "V100/overhead/V100_it.necst.grcuda.benchmark.TestBenchmarks-output.txt",
        ]

    ## A100
    res_list_a100 = [
        "A100/overhead/it.necst.grcuda.benchmark.TestBenchmarks-output_A100.txt",
        ]

    # V100
    scheduling_overhead_v100 = load_overhead_java(res_list_v100)
    # chosen_sizes = pd.DataFrame.from_dict({k: [v] for k, v in {"VEC": 160000000, "B&S": 10000000, "ML": 1000000, "CG": 20000, "MUL": 20000}.items()}, orient="index").reset_index().rename(columns={"index": "benchmark", 0: "size"})
    # scheduling_overhead_v100 = scheduling_overhead_v100.merge(chosen_sizes, how="inner", on=["benchmark", "size"])
    # scheduling_overhead_v100 = scheduling_overhead_v100.query(f"gpus > 3")
    scheduling_overhead_v100.to_csv(os.path.join(DEFAULT_OVERHEAD_DIR, "V100/overhead/scheduling_overhead_v100_default.csv"), index=False)

    # A100
    scheduling_overhead_a100 = load_overhead_java(res_list_a100)
    chosen_sizes = pd.DataFrame.from_dict({k: [v] for k, v in {"VEC": 950000000, "B&S": 35000000, "ML": 1200000, "CG": 50000, "MUL": 60000}.items()}, orient="index").reset_index().rename(columns={"index": "benchmark", 0: "size"})
    scheduling_overhead_a100 = scheduling_overhead_a100.merge(chosen_sizes, how="inner", on=["benchmark", "size"])
    # scheduling_overhead_a100 = scheduling_overhead_a100.query(f"gpus > 3")
    scheduling_overhead_a100.to_csv(os.path.join(DEFAULT_OVERHEAD_DIR, "A100/overhead/scheduling_overhead_a100_default.csv"), index=False)
