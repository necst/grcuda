#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import wilcoxon
import scipy.stats as st
import os
import pandas as pd
import numpy as np

from plot_kernel_exec_time import load_data
from plot_kernel_scalability import remove_outliers
from scipy.stats.mstats import gmean


GPU = "GTX1660"
DATE = "2020_11_13_16_06_37"

RES_FOLDER = f"../../../../../data/oob/results/{GPU}/{DATE}"
PLOT_FOLDER = f"../../../../../data/oob/plots/{GPU}/{DATE}"

KERNELS =  ["axpy", "dot_product", "convolution", "mmul", "autocov", "hotspot", "hotspot3d",
            "backprop", "backprop2", "bfs", "pr", "nested", "gaussian",
            "histogram", "lud", "needle"]


if __name__ == "__main__":
    
    ##################################
    # Load data ######################
    ##################################
        
    # res_folder = "../../../data/results/with_lower_bounds/2019_10_23"
    # out_folder = "../../../data/tables/2019_10_23"
    
    # res_axpy = load_data(os.path.join(res_folder, "axpy_2019_10_23_16_55_54.csv"))
    # res_dp = load_data(os.path.join(res_folder, "dot_product_2019_10_23_16_55_54.csv"))
    # res_conv = load_data(os.path.join(res_folder, "convolution_2019_10_23_17_42_09.csv"))
    # res_mmul = load_data(os.path.join(res_folder, "mmul_2019_10_23_17_42_09.csv"))
    # res_autocov = load_data(os.path.join(res_folder, "autocov_2019_10_23_16_55_54.csv"))
    # res_hotspot = load_data(os.path.join(res_folder, "hotspot_2019_10_23_17_42_09.csv"))
    # res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d_2019_10_23_16_55_54.csv"))
    # res_bb = load_data(os.path.join(res_folder, "backprop_2019_10_23_16_55_54.csv"))
    # res_bb2 = load_data(os.path.join(res_folder, "backprop2_2019_10_23_16_55_54.csv"))
    # res_bfs = load_data(os.path.join(res_folder, "bfs_2019_10_23_16_55_54.csv"))
    # res_pr = load_data(os.path.join(res_folder, "pr_2019_10_23_16_55_54.csv"))  
    # res_gaussian = load_data(os.path.join(res_folder, "gaussian_2019_10_23_16_55_54.csv"))
    # res_histogram = load_data(os.path.join(res_folder, "histogram_2019_10_23_17_42_09.csv"))
    # res_lud = load_data(os.path.join(res_folder, "lud_2019_10_23_17_42_09.csv"))
    # res_needle = load_data(os.path.join(res_folder, "needle_2019_10_23_16_55_54.csv"))
    # res_nested = load_data(os.path.join(res_folder, "nested_2019_10_23_16_55_54.csv"))

    res_list = []
    for k in KERNELS:
        for f in os.listdir(RES_FOLDER):
            if k == "_".join(f.split("_")[:-6]):
                print(f, k)
                res_list += [load_data(os.path.join(RES_FOLDER, f))]

     
    # res_list = [res_axpy, res_dp, res_conv, res_autocov, res_hotspot3d, res_bb, res_bfs, res_pr, res_mmul, res_hotspot,
    #             res_bb2, res_gaussian,
    #             res_histogram, res_lud, res_needle, res_nested]
    
    res_list = [remove_outliers(res) for res in res_list]
    
    names = ["Axpy", "Dot Product", "Convolution 1D",  "Auto-covariance", "Hotspot - 3D", "NN - Forward Phase",
             "BFS", "PageRank", "Matrix Multiplication", "Hotspot", 
             "NN - Backpropagation",  "Gaussian Elimination",
             "Histogram", "LU Decomposition", "Needleman-Wunsch", "Nested Loops"]
    
    # Tests;
    test_res = []
    for o in ["O0", "O2"]:
        for s in ["simplify_accesses"]:
            for i, res in enumerate(res_list):            
                curr_res = res[(res["opt_level"] == o) & (res["simplify"] == s) & (res["num_elements"] == max(res["num_elements"]))]
                
                # Perform wilcoxon test on the kernel execution time differences;
                diff = curr_res["time_m_k_ms"]  / curr_res["time_u_k_ms"] - 1
                faster_kernel = "Manually Modified" if np.median(curr_res["time_u_k_ms"]) < np.median(curr_res["time_m_k_ms"]) else "Automatically Modified"
                pvalue = wilcoxon(diff).pvalue
                faster_kernel = faster_kernel if pvalue < 10e-4 else "-"
                
                print(f"-- {names[i]}, {o}")
                print(f"    {pvalue}, Faster Kernel: {faster_kernel}")
                
                test_res += [{"Kernel Name": names[i], "Optimization Level": o, "Wilcoxon Test p-value": pvalue, "Faster Kernel": faster_kernel}]
                
    # Compute also if overall modified kernels are faster. Obtain the median for each kernel, then compare medians;
    for o in ["O0", "O2"]:
        for s in ["simplify_accesses"]:
            speedups_mean_u = []
            speedups_mean_m = []
            for i, res in enumerate(res_list):     
                curr_res = res[(res["opt_level"] == o) & (res["simplify"] == s) & (res["num_elements"] == max(res["num_elements"]))]
                
                median_u = np.median(curr_res["time_u_k_ms"])
                speedups_mean_u += [gmean(curr_res["time_u_k_ms"] / median_u)]
                speedups_mean_m += [gmean(curr_res["time_m_k_ms"] / median_u)]
            diff = np.array(speedups_mean_u) - np.array(speedups_mean_m)
            faster_kernel = "Manually Modified" if gmean(speedups_mean_u) < gmean(speedups_mean_m) else "Automatically Modified"
            pvalue = wilcoxon(diff).pvalue
            faster_kernel = faster_kernel if pvalue < 10e-4 else "-"
            
            print(f"-- Overall, {o}")
            print(f"    {pvalue}, Faster Kernel: {faster_kernel}")
            test_res += [{"Kernel Name": "Overall, Harmonic Mean", "Optimization Level": o, "Wilcoxon Test p-value": pvalue, "Faster Kernel": faster_kernel}]           
    
    test_df = pd.DataFrame(test_res, columns=["Kernel Name", "Optimization Level", "Wilcoxon Test p-value", "Faster Kernel"])
    
    # Store the results;
    out_path = os.path.join(PLOT_FOLDER, "exec_time_wilcoxon.csv")
    test_df.to_csv(out_path, index=False, float_format="%.3g")