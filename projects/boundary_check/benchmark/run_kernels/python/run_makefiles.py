#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import argparse
import pandas as pd
import time
import numpy as np

##############################
##############################

RESULT_FOLDER = "../../../../data/results/compilation"
TRUFFLECUDA_jAR = "../../../../../../mxbuild/dists/jdk1.8/trufflecuda.jar"
NUM_TESTS = 2

##############################
##############################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="run trufflecuda kernels with boundary checks")
    parser.add_argument("-t", "--num_tests", metavar="N", type=int, default=NUM_TESTS,
                        help="Number of times each test is executed")
    
    args = parser.parse_args()
    
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    opt_levels = ["O0", "O2"]
    simplify = [False, True]
    
    kernels=[
             "axpy",
             "dot_product",
             "convolution",
             "hotspot",
             "mmul",
             "autocov",
             "backprop",
             "backprop2",
             "bfs",
             "gaussian",
             "pr",
             "hotspot3d",
             "histogram",
             "lud",
             "needle"
            ]
    
    # Define the output file;
    output_file = os.path.join(RESULT_FOLDER, f"makefile_{now}.csv")
    print(f"saving results to {output_file}")
    
    # Execute each test;
    results = []
    for k in kernels:            
        for o in opt_levels:
            for s in simplify:
                simplify_flag = "-s" if s else ""
                cmd_modified = f"make --directory ../.. -s {k} OPT_LEVEL={o}" + (" SIMPLIFY=y" if s else "")
                print(f"executing {cmd_modified}")
                cmd_unmodified = f"make --directory ../.. -s {k}_u OPT_LEVEL={o}" + (" SIMPLIFY=y" if s else "")
                print(f"executing {cmd_unmodified}")
                exec_times_m = []
                exec_times_u = []
                for n in range(args.num_tests):
                    
                    start_m = time.perf_counter()
                    os.system(cmd_modified)
                    end_m = time.perf_counter()
                    exec_time_m = (end_m - start_m)
                    exec_times_m += [exec_time_m]

                    start_u = time.perf_counter()
                    os.system(cmd_unmodified)
                    end_u = time.perf_counter()
                    exec_time_u = (end_u - start_u)
                    exec_times_u += [exec_time_u]
                    
                    results += [{
                            "kernel": k,
                            "iteration": n,
                            "opt_level": o,
                            "simplify": s,
                            "exec_time_m_s": exec_time_m,
                            "exec_time_u_s": exec_time_u
                            }]
                print(f"mean modified time: {np.mean(exec_times_m)}")
                print(f"mean unmodified time: {np.mean(exec_times_u)}")
    
    results_df = pd.DataFrame(results, columns=["kernel", "iteration", "opt_level", "simplify", "exec_time_m_s", "exec_time_u_s"])
    results_df.to_csv(output_file, index=False)
                    
        

