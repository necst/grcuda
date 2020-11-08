#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy.stats as st
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import os

from plot_kernel_exec_time import build_plot, r1, r2, r3, b1, b2, b3, b4, load_data, get_upper_ci_size, remove_outliers


def get_exp_label(val):
    
    # Get the power of 10
    exp_val = 0
    remaining_val = val
    while (remaining_val % 10 == 0):
        exp_val += 1
        remaining_val = remaining_val // 10
    return r"$\mathdefault{" + str(remaining_val) + r"·{10}^" + str(exp_val) + r"}$"


def build_exec_time_plot(data, position, opt=""):
    kernel_times = data.melt(id_vars="num_elements", value_vars=['time_u_k_ms', 'time_m_k_ms'], value_name="time_ms")
    tot_times = data.melt(id_vars="num_elements", value_vars=['time_u_ms', 'time_m_ms'], value_name="time_ms")
    
    labels = sorted(set(tot_times["num_elements"]))
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(position)
    ax = sns.lineplot(x="num_elements", y="time_ms", hue="variable", data=kernel_times, palette=[b2, r3], ax=ax,
                      err_style="bars", linewidth=3, legend=False, zorder=2, ci=None)
    
    # Add rectangles
    for c_i, c in enumerate(['time_u_k_ms', 'time_m_k_ms']):
        rectangles = []
        segments = []
        for s_i, s in enumerate(labels):
            curr_data = kernel_times[(kernel_times["variable"] == c) & (kernel_times["num_elements"] == s)]
            y = np.mean(curr_data["time_ms"])
            width = (max(labels) - min(labels)) / 30
            height = get_upper_ci_size(curr_data["time_ms"]) * 2
            lower_left = [s - width / 2, y - height / 2]
            # Add an offset to the x position, to avoid overlapping;
            lower_left[0] += (2 * c_i - 1) * (width / 3.5)
            rectangles += [Rectangle(lower_left, width, height)]
            
            segments += [[
                    (lower_left[0] + width / 2, y - np.std(curr_data["time_ms"])),
                    (lower_left[0] + width / 2, y + np.std(curr_data["time_ms"]))
                    ]]
        
        pc = PatchCollection(rectangles, facecolor=[b2, r3][c_i], edgecolor="#2f2f2f", linewidth=0.5, zorder=3, clip_on=False)        
        lc = LineCollection(segments, linewidth=0.5, zorder=3, clip_on=True, color="#2f2f2f")
#        ax.add_collection(lc)    
        ax.add_collection(pc)                
                
    # Set the x ticks;
    ax.set_xticks(labels)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=15)
    ax.tick_params(labelcolor="black")
    
    ax.set_yticklabels(labels=["{:.1f}".format(l) for l in ax.get_yticks()], ha="right", fontsize=15)
    
#    ax.set_title(f"Scalability, {opt}", fontsize=18)
    ax.set_ylabel("Time [ms]", fontsize=18)     
    ax.set_xlabel("Input Size", fontsize=16) 
    
    sns.despine(ax=ax)
    # Turn off tick lines;
    ax.xaxis.grid(False)
    
    return ax
    

def build_overhead_plot(data, position, opt=""):
            
    # Compute the percentage overhead;
    data.loc[:, "overhead_m"] = 100 * (data["time_m_ms"] - data["time_m_k_ms"]) / data["time_m_ms"]
    data.loc[:, "overhead_u"] = 100 * (data["time_u_ms"] - data["time_u_k_ms"]) / data["time_u_ms"]
    
    overheads = data.melt(id_vars="num_elements", value_vars=['overhead_m', 'overhead_u'], value_name="overhead")
    
    labels = sorted(set(overheads["num_elements"]))
    
    # Add a lineplot with the exec times;
    ax = fig.add_subplot(position)
    ax = sns.lineplot(x="num_elements", y="overhead", hue="variable", data=overheads, palette=[b1, r1], ax=ax,
                      err_style="bars", linewidth=3, legend=False, ci=None)
    
    # Add rectangles
    for c_i, c in enumerate(['overhead_m', 'overhead_u']):
        rectangles = []
        segments = []
        for s_i, s in enumerate(labels):
            curr_data = overheads[(overheads["variable"] == c) & (overheads["num_elements"] == s)]
            y = np.mean(curr_data["overhead"])
            width = (max(labels) - min(labels)) / 30
            height = get_upper_ci_size(curr_data["overhead"]) * 2
            lower_left = [s - width / 2, y - height / 2]
            # Add an offset to the x position, to avoid overlapping;
            lower_left[0] += (2 * c_i - 1) * (width / 3.5)
            rectangles += [Rectangle(lower_left, width, height)]
            
            segments += [[
                    (lower_left[0] + width / 2, y - np.std(curr_data["overhead"])),
                    (lower_left[0] + width / 2, y + np.std(curr_data["overhead"]))
                    ]]
        
        pc = PatchCollection(rectangles, facecolor=[b2, r3][c_i], edgecolor="#2f2f2f", linewidth=0.5, zorder=3, clip_on=False)        
        lc = LineCollection(segments, linewidth=0.5, zorder=3, clip_on=True, color="#2f2f2f")
#        ax.add_collection(lc)    
        ax.add_collection(pc)  
                
    # Set the x ticks;
    ax.set_xticks(labels)
    ax.set_xticklabels(labels=[get_exp_label(l) for l in labels], rotation=45, ha="right", fontsize=15)
    ax.tick_params(labelcolor="black")
    
#    ax.set_title(f"Overheads, {opt}", fontsize=18)
    ax.set_ylabel("Overhead [%]", fontsize=18)     
    ax.set_xlabel("Input Size", fontsize=16) 
    ax.set_yticklabels(labels=[f"{int(l)}%" for l in ax.get_yticks()], ha="right", fontsize=15)
    
    sns.despine(ax=ax)
    # Turn off tick lines;
    ax.xaxis.grid(False)
    
    return ax
            
            
#%%

if __name__ == "__main__":
    
    # Plotting setup;
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 

    ##################################
    # Load data ######################
    ##################################
    
    res_folder = "../../../data/results/scalability/2019_09_03"
    plot_dir = "../../../data/plots/scalability/2020_02_13"
    
    
    res_mmul = load_data(os.path.join(res_folder, "mmul_2019_09_03_08_21_45.csv"))
    res_autocov = load_data(os.path.join(res_folder, "autocov_2019_09_01_11_50_00.csv"))
    res_bb = load_data(os.path.join(res_folder, "backprop_2019_09_01_11_50_00.csv"))
    res_needle = load_data(os.path.join(res_folder, "needle_2019_09_01_11_19_29.csv"))
    
    res_list = [res_mmul, res_autocov, res_bb, res_needle]
    
    res_list = [remove_outliers(res) for res in res_list]
    
    ##################################
    # Plotting #######################
    ##################################
    
    num_plots = len(res_list)
    fig = plt.figure(figsize=((num_plots + 1) * 4.7, 4.7 * num_plots))
    gs = gridspec.GridSpec(num_plots, num_plots)
    plt.subplots_adjust(top=0.93,
                    bottom=0.1,
                    left=0.18,
                    right=0.8,
                    hspace=1.3,
                    wspace=0.6)
    
    names = ["Matrix Multiplication", "Auto-covariance", "NN - Forward Phase", "Needleman-Wunsch"]
    
    exec_time_axes = []
    overhead_axes = []
    for res_i, res in enumerate(res_list):
        for o_i, o in enumerate(["O0", "O2"]):
            # Build the required data view;
            curr_res = res[(res["opt_level"] == o)]    
            
            exec_time_axes += [build_exec_time_plot(curr_res, gs[res_i, 2 * o_i], opt=o)]
            overhead_axes += [build_overhead_plot(curr_res, gs[res_i, 2 * o_i + 1], opt=o)]
    
    for i, n in enumerate(names):    
        fig.get_axes()[i * 4].annotate(n, xy=(0, 0.5), xycoords="axes fraction", fontsize=18, ha="right", va="center",
                     textcoords="offset points", xytext=(-80, 0))

    # Legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Manually Modified"),
        Patch(facecolor=r1, edgecolor="#2f2f2f", label="Automatically Modified"),
        ]
    
    ax = fig.get_axes()[0]
    leg = ax.legend(custom_lines, ["Manually Modified", "Automatically Modified"],
                             bbox_to_anchor=(7.3, 0.75), fontsize=16)
    leg.set_title(None)
    leg._legend_box.align = "left"
    
    plt.savefig(os.path.join(plot_dir, "scalability_o1o2.pdf"))
    plt.savefig(os.path.join(plot_dir, "scalability_o1o2.png"))
    
    
    #%% Only O2 optimization;

    # Plotting setup;
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 20 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 

    ##################################
    # Load data ######################
    ##################################
    
    res_folder = "../../../data/results/scalability/2019_09_03"
    plot_dir = "../../../data/plots/scalability/2020_02_13"
    
    res_autocov = load_data(os.path.join(res_folder, "autocov_2019_09_01_11_50_00.csv"))
    res_bb = load_data(os.path.join(res_folder, "backprop_2019_09_01_11_50_00.csv"))
    res_mmul = load_data(os.path.join(res_folder, "mmul_2019_09_03_08_21_45.csv"))
    res_needle = load_data(os.path.join(res_folder, "needle_2019_09_01_11_19_29.csv"))
    
    res_list = [res_autocov, res_bb, res_mmul, res_needle]
    
    res_list = [remove_outliers(res) for res in res_list]
    
    ##################################
    # Plotting #######################
    ##################################
    
    num_plots = len(res_list)
    fig = plt.figure(figsize=(3.2 * num_plots + 2, 2 * 4.6))
    gs = gridspec.GridSpec(2, num_plots)
    plt.subplots_adjust(top=0.8,
                    bottom=0.15,
                    left=0.1,
                    right=0.95,
                    hspace=0.7,
                    wspace=0.8)
    
    names = ["Auto-covariance", "NN - Forward Pass", "Matrix Multiplication", "Needleman-Wunsch"]
    
    exec_time_axes = []
    overhead_axes = []
    for res_i, res in enumerate(res_list):
        for o_i, o in enumerate(["O2"]):
            # Build the required data view;
            curr_res = res[(res["opt_level"] == o)]    
            exec_time_axes += [build_exec_time_plot(curr_res, gs[2 * o_i, res_i], opt=o)]
    for res_i, res in enumerate(res_list):
        for o_i, o in enumerate(["O2"]):
            # Build the required data view;
            curr_res = res[(res["opt_level"] == o)]   
            overhead_axes += [build_overhead_plot(curr_res, gs[2 * o_i + 1, res_i], opt=o)]
   
    for i, n in enumerate(names):    
        fig.get_axes()[i].annotate(n, xy=(0, 1), xycoords="axes fraction", fontsize=18, ha="left", va="center",
                     textcoords="offset points", xytext=(0, 30))

    # Legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Baseline"),
        Patch(facecolor=r1, edgecolor="#2f2f2f", label="Automatically Modified"),
        ]
    
    leg = fig.legend(custom_lines, ["Baseline", "Automatically Modified"],
                     bbox_to_anchor=(0.98, 0.99), fontsize=16)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    fig.suptitle("Kernels Scalability & Overheads,\nO2 Opt. Level", ha="left", x=0.1, y=0.98, fontsize=20)
    
    plt.savefig(os.path.join(plot_dir, "scalability_o2.pdf"))
    plt.savefig(os.path.join(plot_dir, "scalability_o2.png"))

    