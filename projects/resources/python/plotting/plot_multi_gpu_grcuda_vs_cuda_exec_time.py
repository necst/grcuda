# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:02:05 2022

@author: albyr
"""

import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from segretini_matplottini.src.plot_utils import *
from segretini_matplottini.src.plot_utils import add_labels

from load_data import load_data_cuda_multigpu, load_data_grcuda_multigpu, PLOT_DIR

##############################
##############################

OUTPUT_DATE = "2022_01_20"

# V100;
V100 = "V100"
V100_RES_FOLDERS_CUDA = [
    "2022_01_16_18_09_04_cuda_1-2gpu_v100",
    "2022_01_16_18_17_05_cuda_4gpu_v100",
    ]
V100_RES_FOLDERS_GRCUDA = [
    "2022_01_18_16_10_41_grcuda_1-2gpu_v100",
    "2022_01_18_10_01_23_grcuda_4gpu_v100",
    ]


class ResultGPU:
    def __init__(self, gpu: str, results_grcuda: list, results_cuda: list):
        self.gpu = gpu
        self.results_grcuda_folders = results_grcuda
        self.results_cuda_folders = results_cuda
        self.results_grcuda = None
        self.results_cuda = None
        self.results_merged = None
    
    def load_cuda(self):
        self.results_cuda = load_data_cuda_multigpu([os.path.join(self.gpu, f) for f in self.results_cuda_folders], skip_iter=3)
        return self.results_cuda
    
    def load_grcuda(self):
        self.results_grcuda = load_data_grcuda_multigpu([os.path.join(self.gpu, f) for f in self.results_grcuda_folders], skip_iter=3)
        return self.results_grcuda
    
    def group_grcuda_results(self, group_sizes: bool=False, drop_sync: bool=False, drop_nan: bool=True):
        if self.results_grcuda is None:
            self.load_grcuda()
        group_fields = ["benchmark", "exec_policy", "gpus"] + \
            (["size"] if not group_sizes else []) + \
            ["parent_stream_policy", "device_selection_policy"]
        grouped = self.results_grcuda.groupby(group_fields).mean()[["computation_sec", "speedup"]].reset_index()
        if drop_nan:
            grouped = grouped.dropna().reset_index(drop=True)
        if drop_sync:
            grouped = grouped[grouped["exec_policy"] != "SYNC"]
        return grouped 
    
    def join_grcuda_and_cuda_results(self):
        if self.results_grcuda is None:
            self.load_grcuda()
        if self.results_cuda is None:
            self.load_cuda()    
        res_merged = self.results_grcuda.merge(self.results_cuda, how="left",
                                               on=["benchmark", "size", "gpus", "exec_policy",
                                                   "prefetch", "num_blocks", "block_size_1d", 
                                                   "block_size_2d", "num_iter",
                                                   "total_iterations", "block_size_str"],
                                               suffixes=["_grcuda", "_cuda"])
        # Keep only the GrCUDA speedup vs. GrCUDA sync, and the raw execution time of GrCUDA and CUDA;
        columns_to_keep = [c for c in res_merged.columns if "cuda" not in c] + \
            ["computation_sec_grcuda", "computation_sec_cuda", "baseline_time_grcuda"]
        res_merged.rename(columns={"speedup_grcuda": "speedup_grcuda_vs_grcuda_sync"}, inplace=True)
        res_merged = res_merged[columns_to_keep + ["speedup_grcuda_vs_grcuda_sync"]]
        # Compute speedup of GrCUDA vs CUDA;
        res_merged["speedup_grcuda_vs_cuda"] = res_merged["computation_sec_cuda"] / res_merged["computation_sec_grcuda"]
        self.res_merged = res_merged
        return self.res_merged
    
    def group_merged_results(self, group_sizes: bool=False, drop_sync: bool=False, drop_nan: bool=True):
        if self.res_merged is None:
            self.join_grcuda_and_cuda_results()
        group_fields = ["benchmark", "exec_policy", "gpus"] + \
            (["size"] if not group_sizes else []) + \
            ["parent_stream_policy", "device_selection_policy"]
        grouped = self.res_merged.groupby(group_fields).mean()[["computation_sec_grcuda", "computation_sec_cuda", 
                                                                "speedup_grcuda_vs_grcuda_sync", "speedup_grcuda_vs_cuda"]].reset_index()
        if drop_nan:
            grouped = grouped.dropna().reset_index(drop=True)
        if drop_sync:
            grouped = grouped[grouped["exec_policy"] != "SYNC"]
        return grouped 
    

V100_RESULTS = ResultGPU(
    gpu=V100,
    results_grcuda=V100_RES_FOLDERS_GRCUDA,
    results_cuda=V100_RES_FOLDERS_CUDA
    )

#%%###########################
##############################    

if __name__ == "__main__":
    g = V100_RESULTS
    res_cuda = g.load_cuda()
    res_grcuda = g.load_grcuda()
    res_grcuda_grouped = g.group_grcuda_results()
    res_grcuda_grouped_small = g.group_grcuda_results(group_sizes=True, drop_sync=True)
    res_merged = g.join_grcuda_and_cuda_results()
    res_merged_grouped = g.group_merged_results(group_sizes=True, drop_sync=True)