#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy.stats as st
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

import os


from plot_kernel_exec_time import build_plot, r1, r2, r3, b1, b2, b3, load_data, remove_outliers


def plot_violin(res, o="O0", s="no_simplification", vertical_plot_position=0, name="", vlabel_offset=0.15):
    ax0 = None
    # Build the required data view;
    curr_res = res[(res["opt_level"] == o) & (res["simplify"] == s) & (res["num_elements"] == max(res["num_elements"]))]            
    ax0, kernel_res, tot_res, speedup = build_plot(curr_res, fig, gs[vertical_plot_position, 0], ax0,
       f"{name},\n{o},\n{'with access merging' if s == 'simplify_accesses' else 'no access merging'}", vlabel_offset=vlabel_offset)
    
    # Plot Original Violin;     
    ax = fig.add_subplot(gs[vertical_plot_position, 1]) 
    ax = sns.violinplot(x="Type", y="time_ms", data=tot_res.loc[tot_res["Type"] == "Original", :], palette=[b3], ax=ax)
    ax = sns.violinplot(x="Type", y="time_ms", data=kernel_res.loc[kernel_res["Type"] == "Original", :], palette=[r1], ax=ax)
    ax.set_title("Original\nExec. Time Distribution", fontsize=18)
    ax.set_ylabel("Time [ms]", fontsize=16)     
    ax.set_xlabel("Type", fontsize=16) 
    sns.despine(ax=ax)    
    # Plot Modified Violin;      
    ax = fig.add_subplot(gs[vertical_plot_position, 2], sharey=ax) 
    ax = sns.violinplot(x="Type", y="time_ms", data=tot_res.loc[tot_res["Type"] == "Modified", :], palette=[b1], ax=ax)
    ax = sns.violinplot(x="Type", y="time_ms", data=kernel_res.loc[kernel_res["Type"] == "Modified", :], palette=[r3], ax=ax)
    ax.set_title("Modified\nExec. Time Distribution", fontsize=18)
    ax.set_ylabel("Time [ms]", fontsize=16)     
    ax.set_xlabel("Type", fontsize=16)  
    sns.despine(ax=ax)

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
    
    res_folder = "../../../data/results/ridge"
    plot_dir = "../../../data/plots/with_lower_bounds/2020_02_13"
    
    res_axpy = load_data(os.path.join(res_folder, "axpy_2019_09_17_10_39_43.csv"))
    res_dp = load_data(os.path.join(res_folder, "dot_product_2019_09_17_10_39_43.csv"))
    res_conv = load_data(os.path.join(res_folder, "convolution_2019_09_17_10_39_43.csv"))
    res_mmul = load_data(os.path.join(res_folder, "mmul_2019_09_17_10_39_43.csv"))
    res_autocov = load_data(os.path.join(res_folder, "autocov_2019_09_17_10_39_43.csv"))
    res_hotspot = load_data(os.path.join(res_folder, "hotspot_2019_09_17_10_39_43.csv"))
    res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d_2019_09_17_10_39_43.csv"))
    res_bb = load_data(os.path.join(res_folder, "backprop_2019_09_17_10_39_43.csv"))
    res_bb2 = load_data(os.path.join(res_folder, "backprop2_2019_09_17_10_39_43.csv"))
    res_bfs = load_data(os.path.join(res_folder, "bfs_2019_10_23_18_45_06.csv"))
    res_pr = load_data(os.path.join(res_folder, "pr_2019_09_17_10_39_43.csv"))  
    res_gaussian = load_data(os.path.join(res_folder, "gaussian_2019_09_17_10_39_43.csv"))
    res_histogram = load_data(os.path.join(res_folder, "histogram_2019_10_23_17_42_09.csv"))
    res_lud = load_data(os.path.join(res_folder, "lud_2019_10_23_17_42_09.csv"))
    res_needle = load_data(os.path.join(res_folder, "needle_2019_10_23_17_42_09.csv"))
    res_nested = load_data(os.path.join(res_folder, "nested_2019_10_22_14_54_43.csv"))
    
    boundary_merging_statistics = pd.read_csv("../../../data/results/access_merging_statistics.csv")
    boundary_merging_statistics["plot_merged"] = True # boundary_merging_statistics["unprotected_num_of_accesses"] > boundary_merging_statistics["merged_num_of_accesses"]
    
    res_list = [res_axpy, res_dp, res_conv, res_autocov, res_hotspot3d, res_bb, res_bfs, res_pr, res_nested, res_mmul, res_hotspot, 
                 res_bb2, res_gaussian,
                res_histogram, res_lud, res_needle]
    
    names = ["Axpy", "Dot Product", "Convolution 1D", "Auto-covariance", "Hotspot 3D",
             "NN - Forward Pass", "BFS", "PageRank", "Nested Loops", "Matrix Multiplication", "Hotspot",  "NN - Backpropagation",  "Gaussian Elimination",
             "Histogram", "LU Decomposition", "Needleman-Wunsch"]
    
    res_list = [remove_outliers(res) for res in res_list]
    
    ##################################
    # Plotting #######################
    ##################################
    
    num_plots = len(res_list)
    num_col = 4
    fig = plt.figure(figsize=(4.0 * num_col, num_plots * 4.8))
    gs = gridspec.GridSpec(num_plots, 4)
    plt.subplots_adjust(top=0.98,
                    bottom=0.02,
                    left=0.11,
                    right=0.95,
                    hspace=1.1,
                    wspace=0.7)
    
    vlabel_offsets= [0.4, 0.10, 0.10, 0.05, 0.09, 0.06, 0.2,
                     0.3, 0.5, 0.1, 0.1, 0.5,
                     0.1, 0.1, 0.2, 0.2]
    o_list = ["O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2", "O2"]
    s_list = ["no_simplification", "no_simplification", "no_simplification", "no_simplification", "no_simplification",
              "no_simplification", "no_simplification", "no_simplification", "no_simplification", "simplify_accesses",
              "no_simplification", "no_simplification", "no_simplification", "no_simplification", "no_simplification",
              "simplify_accesses", "simplify_accesses"]
    
    # for i, res in enumerate(res_list):
    #     plot_violin(res, o_list[i], s_list[i], i, names[i], vlabel_offsets[i])
 
    ##################################
    # Legend #########################
    
    # Add custom legend;
    # custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Overall Time"),
    #                 Patch(facecolor=r3, edgecolor="#2f2f2f", label="Kernel Time"),
    #                 ]
    
    # ax = fig.get_axes()[0]
    # leg = ax.legend(custom_lines, ["Overall Time", "Kernel Time"],
    #                          bbox_to_anchor=(5.8, 0.9), fontsize=16)
    # leg.set_title("Exec. Time Group", prop={"size": 18})
    # leg._legend_box.align = "left"
    
    # plt.savefig(os.path.join(plot_dir, "exec_times_violins.pdf"))
    # plt.savefig(os.path.join(plot_dir, "exec_times_violins.png"))
    
    
    #%
    
    ##################################
    # Densities ######################
    ##################################
    
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]

    num_col = 2
    for i in range(num_col):
        s = i * len(res_list) // num_col
        e = (i + 1) * len(res_list) // num_col
        for j in range(s, e):
            res_list[j]["group"] = i
            res_list[j]["kernel_group"] = j % (len(res_list) // num_col)
        
    # Create a unique table;
    res_list_filtered = []
    for i, res in enumerate(res_list):
        temp_res = res[(res["simplify"] == "simplify_accesses") & (res["opt_level"] == "O2") & (res["num_elements"] == max(res["num_elements"]))].reset_index()  
        temp_res["kernel"] = names[i]
         # Normalize using the unmodified kernel time median;
        temp_res["time_m_k_ms"] /= np.median(temp_res["time_u_k_ms"])
        temp_res["time_u_k_ms"] /= np.median(temp_res["time_u_k_ms"])
        res_list_filtered += [temp_res]
    res_tot = pd.concat(res_list_filtered)
#    res_summary = res_tot.copy()
#    res_summary["kernel"] = "Summary, Geomean"
#    res_tot = pd.concat([res_summary, res_tot])
    g = sns.FacetGrid(res_tot, row="kernel_group", hue="kernel", aspect=7, height=1, palette=["#2f2f2f"], sharey=False,
                      col="group")
    g.map(plt.axvline, x=1, lw=0.75, clip_on=True, zorder=0, linestyle="--", ymax=0.5)                                                                                          
    g.map(sns.kdeplot, "time_u_k_ms", clip_on=False, shade=True, alpha=0.6, lw=1, bw=0.02, color=b1, zorder=2, cut=10)  
    g.map(sns.kdeplot, "time_m_k_ms", clip_on=False, shade=True, alpha=0.6, lw=1, bw=0.02, color=r1, zorder=3, cut=10)
    g.map(sns.kdeplot, "time_u_k_ms", clip_on=False, color="w", lw=1.5, bw=0.02, zorder=2, cut=10)
    g.map(sns.kdeplot, "time_m_k_ms", clip_on=False, color="w", lw=1.5, bw=0.02, zorder=3, cut=10)
    
    g.map(plt.axhline, y=0, lw=1, clip_on=False, zorder=4)
#    g.fig.get_axes()[-1].axvline(x=1, ymax=0.5)
    
    def set_x_width(label="", color="#2f2f2f"):
        ax = plt.gca()
        ax.set_xlim(left=0.5, right=1.25)
    g.map(set_x_width)
    
    def label(x, label, color="#2f2f2f"):
        ax = plt.gca()
        ax.text(0, 0.15, label, color=color, ha="left", va="center", transform=ax.transAxes, fontsize=14)      
    g.map(label, "kernel")
    
    g.fig.subplots_adjust(top=0.83,
                          bottom=0.15,
                          right=0.95,
                          left=0.05,
                          hspace=-0.20,
                          wspace=0.1)
    
    g.set_titles("")
    g.set(xlabel=None)
    g.fig.get_axes()[-1].set_xlabel("Relative Kernel Execution Time", fontsize=15)
    g.fig.get_axes()[-2].set_xlabel("Relative Kernel Execution Time", fontsize=15)
    
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(100 * x)}%"
    g.fig.get_axes()[-1].xaxis.set_major_formatter(major_formatter)
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # Add custom legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Manually Modified"),
                    Patch(facecolor=r3, edgecolor="#2f2f2f", label="Automatically Modified"),
                    ]
    leg = g.fig.legend(custom_lines, ["Baseline", "Automatically Modified"],
                             bbox_to_anchor=(0.97, 0.98), fontsize=15)
    leg.set_title("Kernel Type", prop={"size": 15})
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    g.fig.suptitle("Kernel Relative Exec. Time Distribution,\nO2 Opt. Level", ha="left", x=0.05, y=0.95, fontsize=18)
    
    # plt.savefig(os.path.join(plot_dir, "exec_times_ridge.pdf"))
    # plt.savefig(os.path.join(plot_dir, "exec_times_ridge.png"))
    
   
