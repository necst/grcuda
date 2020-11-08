#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import argparse

##############################
##############################

RESULT_FOLDER = "../../../../data/results"
TRUFFLECUDA_jAR = "../../../../../../mxbuild/dists/jdk1.8/trufflecuda.jar"
NUM_TESTS=100

##############################
##############################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="run trufflecuda kernels with boundary checks")
    
    parser.add_argument("-d", "--debug", action='store_true',
                        help="If present, print debug messages")
    parser.add_argument("-t", "--num_tests", metavar="N", type=int, default=NUM_TESTS,
                        help="Number of times each test is executed")
    
    args = parser.parse_args()
    
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    debug = args.debug

    opt_levels = ["O0", "O2"]
    simplify = [False, True]
    
    kernels=[
            #"axpy",
            #"dot_product",
            #"convolution",
            #"hotspot",
            #"mmul",
            #"autocov",
            #"backprop",
            #"backprop2",
            #"bfs",
            #"gaussian",
            #"pr",
            #"hotspot3d",
            #"histogram",
            #"lud",
            #"needle",
            "nested",

            # USED FOR SCALABILITY TEST;
            #"mmul",
            #"autocov",
            #"backprop",
            #"needle"            
            ]
    
    num_elem = {
            #"axpy": [4000000],
            #"dot_product": [1000000],
            #"convolution": [1000000],
            #"hotspot": [600**2],
            #"mmul": [100000],
            #"autocov": [1000000],
            #"backprop": [400000],
            #"backprop2": [400000],
            #"bfs": [100000],
            #"gaussian": [4096],
            #"pr": [100000],
            #"hotspot3d": [128],
            #"histogram": [2000000],
            #"lud": [2400],
            #"needle": [2400],
            "nested": [100]

            # USED FOR SCALABILITY TEST;
            # "mmul": [20000, 100000, 200000, 400000, 800000],
            # "autocov": [20000, 40000, 60000, 80000, 100000],
            # "backprop": [10000, 20000, 40000, 60000, 80000],
            # "needle" : [1200, 1600, 2000, 2400, 2800]   
            }
    
    # Execute each test;
    for k in kernels:
        
        if not debug:
            # Define the output file;
            output_file = os.path.join(RESULT_FOLDER, f"{k}_{now}.csv")
            print(f"saving results to {output_file}")
            os.system(f"echo 'iteration, num_elements, opt_level, simplify, exec_time_u_k_us, exec_time_u_us, exec_time_m_k_us, exec_time_m_us, errors' > {output_file}")
            
        for o in opt_levels:
            for s in simplify:
                simplify_flag = "-s" if s else ""
                for n in num_elem[k]:
                    
                    cmd = f"graalpython --polyglot --jvm --vm.Dtruffle.class.path.append={TRUFFLECUDA_jAR} run_{k}_kernel.py -o {o} {simplify_flag} -n {n} -t {args.num_tests}"
                    
                    if debug:
                        cmd += " -d"
                    else:
                        cmd += f" >> {output_file}"
                        
                    print(f"executing {cmd}")
                    os.system(cmd)
                    
        
