#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy.stats as st
from matplotlib.patches import Patch
import os
from scipy.stats.mstats import hmean, gmean

from plot_kernel_exec_time import load_data, remove_outliers, draw_plot, b1, b2, b3, r1, r2, r3, build_summary_dfs, update_width, get_upper_ci_size, add_labels

GPU = "GTX1660"
DATE = "2020_11_13_16_06_37"

RES_FOLDER = f"../../../../../data/oob/results/{GPU}/{DATE}"
PLOT_FOLDER = f"../../../../../data/oob/plots/{GPU}/{DATE}"

KERNELS =  ["axpy", "hotspot3d", 
            "backprop", "bfs", "nested", "mmul", "hotspot", "backprop2", "gaussian",
            "histogram"]
    
if __name__ == "__main__":
    
    # Plotting setup;
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    ##################################
    # Load data ######################
    ##################################
    
    # res_folder = "../../../data/results/unsafe_kernels/2019_10_23"
    # plot_dir = "../../../data/plots/unsafe_kernels/2019_10_23"
    
    # res_axpy = load_data(os.path.join(res_folder, "axpy_2019_10_23_19_01_58.csv"))
    # res_mmul = load_data(os.path.join(res_folder, "mmul_2019_10_23_17_42_09.csv"))
    # res_hotspot = load_data(os.path.join(res_folder, "hotspot_2019_10_23_17_42_09.csv"))
    # res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d_2019_10_23_19_01_58.csv"))
    # res_bb = load_data(os.path.join(res_folder, "backprop_2019_10_23_19_01_58.csv"))
    # res_bb2 = load_data(os.path.join(res_folder, "backprop2_2019_10_23_17_42_09.csv"))
    # res_bfs = load_data(os.path.join(res_folder, "bfs_2019_10_23_19_01_58.csv"))
    # res_gaussian = load_data(os.path.join(res_folder, "gaussian_2019_10_23_17_42_09.csv"))
    # res_histogram = load_data(os.path.join(res_folder, "histogram_2019_09_17_10_39_43.csv"))
    # res_nested = load_data(os.path.join(res_folder, "nested_2019_10_23_19_01_58.csv"))
  
    
    
    # res_list = [res_axpy, res_hotspot3d,
    #             res_bb, res_bfs, res_nested, res_mmul, res_hotspot, res_bb2, res_gaussian,
    #             res_histogram]
    
    # res_list = [remove_outliers(res) for res in res_list]
    
    # ##################################
    # # Plotting #######################
    # ##################################
    
    # #%%
    
    # num_plots = len(res_list)
    # num_col = 5
    # fig = plt.figure(figsize=(4.0 * num_col, num_plots * 5.2))
    # gs = gridspec.GridSpec(num_plots, 5)
    # plt.subplots_adjust(top=0.98,
    #                 bottom=0.03,
    #                 left=0.11,
    #                 right=0.95,
    #                 hspace=1.3,
    #                 wspace=0.7)
    
    # names = ["Axpy", "Hotspot 3D",
    #          "NN - Forward Pass", "BFS", "Matrix Multiplication",  "Hotspot", "NN - Backpropagation", "Gaussian Elimination",
    #          "Histogram"]
    # vlabel_offsets= [0.4, 0.15, 0.07, 0.2,
    #                  0.5, 0.5, 0.5, 0.5,
    #                  0.2]
    
    # for i, res in enumerate(res_list):
    #     print(f"Plot {names[i]}")
    #     draw_plot(res, fig, gs, i, names[i], vlabel_offsets[i])
        
    # ##################################
    # # Legend #########################
    
    # # Add custom legend;
    # custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Overall Time"),
    #                 Patch(facecolor=r3, edgecolor="#2f2f2f", label="Kernel Time"),
    #                 ]
    
    # ax = fig.get_axes()[0]
    # leg = ax.legend(custom_lines, ["Overall Time", "Kernel Time"],
    #                          bbox_to_anchor=(7.5, 0.9), fontsize=16)
    # leg.set_title("Exec. Time Group", prop={"size": 18})
    # leg._legend_box.align = "left"
    
    # plt.savefig(os.path.join(plot_dir, "exec_times.pdf"))
    # plt.savefig(os.path.join(plot_dir, "exec_times.png"))    
    
        #%% 
    
    ##################################
    # Other summary plot #############
    ##################################
    
    # res_axpy = load_data(os.path.join(res_folder, "axpy_2019_10_23_19_01_58.csv"))
    # res_mmul = load_data(os.path.join(res_folder, "mmul_2019_09_17_10_39_43.csv"))
    # res_hotspot = load_data(os.path.join(res_folder, "hotspot_2019_10_23_17_42_09.csv"))
    # res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d_2019_08_29_09_35_15.csv"))
    # res_bb = load_data(os.path.join(res_folder, "backprop_2019_10_23_19_01_58.csv"))
    # res_bb2 = load_data(os.path.join(res_folder, "backprop2_2019_10_23_17_42_09.csv"))
    # res_bfs = load_data(os.path.join(res_folder, "bfs_2019_10_23_19_01_58.csv"))
    # res_gaussian = load_data(os.path.join(res_folder, "gaussian_2019_09_17_10_39_43.csv"))
    # res_histogram = load_data(os.path.join(res_folder, "histogram_2019_10_23_17_42_09.csv"))
    # res_nested = load_data(os.path.join(res_folder, "nested_2019_10_23_16_15_00.csv"))
    
    # res_list = [res_axpy, res_hotspot3d,
    #             res_bb, res_bfs, res_nested, res_mmul, res_hotspot, res_bb2, res_gaussian,
    #             res_histogram]
    
    res_list = []
    for k in KERNELS:
        for f in os.listdir(RES_FOLDER):
            if k == "_".join(f.split("_")[:-6]):
                print(f, k)
                res_list += [load_data(os.path.join(RES_FOLDER, f))]
    
    res_list = [remove_outliers(res) for res in res_list]
    
    names = ["Summary,\nGeomean", "AXPY", "HP3D",
             "NN1", "BFS", "NEST", "MULT",  "HP", "NN2", "GE",
             "HIST"]
    
    plt.rcParams['axes.titlepad'] = 50 
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    fig = plt.figure(figsize=(7.5, 6))
    gs = gridspec.GridSpec(1, 1)
    
    plt.subplots_adjust(top=0.70, left=0.12, bottom=0.22, wspace=0.15, right=0.95, hspace=2) 
    
    for i, res in enumerate(res_list):
        res["kernel"] = names[i + 1]
        
    o = "O2"
        
    temp_dfs, summary_dfs = build_summary_dfs(res_list, names[0], o)
    
    temp_dfs_tot = pd.concat(temp_dfs)
                
    # Draw the main plot;   
    ax = fig.add_subplot(gs[0, 0])
    ax = sns.barplot(x="kernel", y="time_ms", data=temp_dfs_tot[temp_dfs_tot["type"] == "simplify_accesses"], palette=[b1, b2, b3], capsize=.1, edgecolor="#2f2f2f", estimator=hmean)
    ax = sns.barplot(x="kernel", y="time_k_ms", data=temp_dfs_tot[temp_dfs_tot["type"] == "simplify_accesses"], palette=[r3, r2, r1], ci=None, ax=ax, edgecolor="#2f2f2f", estimator=hmean)
    # Set labels;
    ax.set_ylabel("Relative Exec. Time", va="bottom", fontsize=18) 
        
    ax.set_yticklabels(labels=[])
    ax.set_xlabel(None)    
    
    v_offsets = [get_upper_ci_size(df.loc[df["type"] == "simplify_accesses", "time_ms"]) + 0.04 for df in temp_dfs]
    speedups = [hmean(df[df["type"] == "simplify_accesses"]["speedup_k"]) for df in temp_dfs]
    add_labels(ax, speedups, v_offsets, range(len(names)))
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    sns.despine(ax=ax)
    update_width(ax, 0.6)      
    # Turn off tick lines;
    ax.grid(False)
    
    # Add legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Overall Time"),
                    Patch(facecolor=r3, edgecolor="#2f2f2f", label="Kernel Time")]
    leg = ax.legend(custom_lines, ["Overall Time", "Kernel Time"],
                             bbox_to_anchor=(1, 1.6), fontsize=16)
    leg.set_title("Exec. Time Group", prop={"size": 18})
    leg._legend_box.align = "left"
        
    fig.suptitle("Kernel Relative Exec. Time\nw.r.t. Unmodified Kernels,\nO2 Opt. Level", ha="left", x=0.1, y=0.95)
        
    
    plt.savefig(os.path.join(PLOT_FOLDER, "exec_times_tot_unsafe.pdf"))
    plt.savefig(os.path.join(PLOT_FOLDER, "exec_times_tot_unsafe.png"))  
    