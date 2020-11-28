#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from scipy.stats.mstats import gmean

from plot_kernel_exec_time import compute_speedup, r1, r2, r3, b1, b2, b3, add_labels, get_upper_ci_size, update_width
    

def load_data(path):
    res = pd.read_csv(path, sep=",") 
    
    res["speedup"] = 1
    compute_speedup(res, "exec_time_u_s", "exec_time_m_s", "speedup")
    
    return res  


def remove_outliers(data, threshold=3):
    types = ["exec_time_m_s", "exec_time_u_s"]
    
    data_list = []
    
    for sim in [False, True]:
        for o_i, o in enumerate(["O0", "O2"]):
                temp_data = data[(data["opt_level"] == o) & (data["simplify"] == sim)]
                for t in types:
                    temp_data = temp_data[st.zscore(temp_data[t]) < 3]
                data_list += [temp_data]
            
    return pd.concat(data_list)
  

def draw_plot(res, fig, gs, plot_vertical_pos, name, vlabel_offset=0.15):
    plot_num = 0
    ax0 = None
    for o in ["O0", "O2"]:
        for s in [False, True]:
            # Build the required data view;
            curr_res = res[(res["opt_level"] == o) & (res["simplify"] == s)]    
            ax, _, _ = build_plot(curr_res, fig, gs[plot_vertical_pos, plot_num], ax0,
                                     f"{name},\n{o},\n{'with access merging' if s else 'no access merging'}",
                                     vlabel_offset=vlabel_offset)
            if plot_num == 0:
                ax0 = ax
            plot_num += 1
            

def build_plot(data, fig, position, ax0=None, title="", vlabel_offset=0.4):
    tot_res = pd.DataFrame({"Type": ["Original"] * len(data) + ["Modified"] * len(data),
                               "time_ms": list(data["exec_time_u_s"]) + list(data["exec_time_m_s"])})
    speedup = 1 / np.median(data["speedup"])
    
    # Add a barplot;
    ax = fig.add_subplot(position, sharey=ax0)
    ax = sns.barplot(x="Type", y="time_ms", data=tot_res, palette=[b3, b1], capsize=.1, ax=ax, edgecolor="#2f2f2f")
    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Exec. Time [ms]", fontsize=16)  
    ax.set_xlabel("Type", fontsize=16) 
    # Compute vertical labels offsets, using the speedup variance;
    v_offset = get_upper_ci_size(tot_res.loc[tot_res["Type"] == "Modified", "time_ms"]) + vlabel_offset
    add_labels(ax, [speedup], [v_offset], [1])
    update_width(ax, 0.5)
    sns.despine(ax=ax)
    
    return ax, tot_res, speedup


def build_summary_dfs(res_list, name="", o="", boundary_merging_statistics=None):
    
    kernel_set = []
    
    temp_dfs = []
    for i, res in enumerate(res_list):
        kernel_set += [res["kernel"].iloc[0]]
        res_temp = res[res["opt_level"] == o].reset_index()  
        
        # Unmodified median time;
        median_time_u_ns = np.median(res_temp[res_temp["simplify"] == False]["exec_time_u_s"])
        median_time_u_s = np.median(res_temp[res_temp["simplify"] == True]["exec_time_u_s"])
        # Obtain the normalized unmodified time;
        unmodified_times_normalized = list(res_temp[res_temp["simplify"] == False]["exec_time_u_s"] / median_time_u_ns) \
            + list(res_temp[res_temp["simplify"] == True]["exec_time_u_s"] / median_time_u_s)
        speedups_normalized = list(res_temp["exec_time_m_s"] / res_temp["exec_time_u_s"])
                
        res_new = pd.DataFrame({"kernel": [res_temp["kernel"].iloc[0]] * (len(unmodified_times_normalized) + len(speedups_normalized)),
                                  "type": len(unmodified_times_normalized)  * ["Unmodified"] + list(res_temp["simplify"]),
                                  "speedup": unmodified_times_normalized + speedups_normalized})
       
        res_new = res_new[st.zscore(res_new["speedup"]) < 3]
        temp_dfs += [res_new]
               
    # Compute speedups and summary plot;
    speedups_ns = []
    speedups_s = []
    speedups_u = []
    for i, df in enumerate(temp_dfs):
        speedups_ns += [gmean(df[df["type"] == False]["speedup"])]
        speedups_s += [gmean(df[df["type"] == True]["speedup"])]
        speedups_u += [gmean(df[df["type"] == "Unmodified"]["speedup"])]
    summary_dfs = pd.DataFrame({"kernel": kernel_set * 3,
                                "type": ["Unmodified"] * len(temp_dfs) + ["no_simplification"] * len(temp_dfs) + ["simplify_accesses"] * len(temp_dfs),
                                "speedup": speedups_u + speedups_ns + speedups_s})
    temp_dfs = [summary_dfs] + temp_dfs
    
    # Remove speedups of datas that do not have any merging being done;
    for res_temp in temp_dfs:
        if boundary_merging_statistics is not None \
                and res_temp["kernel"].iloc[0] in set(boundary_merging_statistics["kernel"]) \
                and not boundary_merging_statistics[boundary_merging_statistics["kernel"] == res_temp["kernel"].iloc[0]]["plot_merged"].iloc[0]:
            res_temp.loc[res_temp["type"] == "simplify_accesses", "speedup"] = 0.000000000001
    
    return temp_dfs, summary_dfs
            

#%%

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
    
    res_folder = "../../../../../data/oob/results/GTX1660/compilation"
    plot_dir = "../../../../../data/oob/plots/GTX1660/compilation"
    
    # res = load_data(os.path.join(res_folder, "makefile_2019_09_09_15_46_42.csv"))
    res = load_data(os.path.join(res_folder, "makefile_2020_11_24_12_15_25.csv"))

    kernel_set = []
    for k in res["kernel"]:
        if k not in kernel_set:
            kernel_set += [k]
    
    res_list = []
    for k in kernel_set:
        res_list += [remove_outliers(res[res["kernel"] == k])]
        
    #%%
    ##################################
    # Plotting #######################
    ##################################
    
    num_plots = len(res_list)
    num_col = 5
    fig = plt.figure(figsize=(4.0 * num_col, num_plots * 5.2))
    gs = gridspec.GridSpec(num_plots, 5)
    plt.subplots_adjust(top=0.98,
                    bottom=0.03,
                    left=0.11,
                    right=0.95,
                    hspace=1.3,
                    wspace=0.7)
    
    names = ["Axpy", "Dot Product", "Convolution 1D", "Matrix Multiplication", "Auto-covariance", "Hotspot", "Hotspot - 3D",
             "NN - Forward Phase", "NN - Backpropagation", "BFS", "PageRank", "Gaussian Elimination",
             "Histogram", "LU Decomposition", "Needleman-Wunsch"]
    vlabel_offsets= [0.4, 0.10, 0.10, 0.15, 0.5, 0.07, 0.2,
                     0.5, 0.5, 0.5, 0.5, 0.5,
                     0.2, 0.1, 0.2]
    
    for i, res in enumerate(res_list):
        print(f"Plot {names[i]}")
        draw_plot(res, fig, gs, i, names[i], vlabel_offsets[i])
        
    ##################################
    # Legend #########################
    
    # Add custom legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Overall Time"),
                    Patch(facecolor=r3, edgecolor="#2f2f2f", label="Kernel Time"),
                    ]
    
    ax = fig.get_axes()[0]
    leg = ax.legend(custom_lines, ["Overall Time", "Kernel Time"],
                             bbox_to_anchor=(7.5, 0.9), fontsize=16)
    leg.set_title("Exec. Time Group", prop={"size": 18})
    leg._legend_box.align = "left"
    
    # plt.savefig(os.path.join(plot_dir, "compilation_times.pdf"))
    # plt.savefig(os.path.join(plot_dir, "compilation_times.png"))         
    
    #%%
    
    ##################################
    # Summary plot ###########
    ##################################
    plt.rcParams['axes.titlepad'] = 30
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    
    fig = plt.figure(figsize=(2 * 2.2, 5.5))
    gs = gridspec.GridSpec(1, 2)
    plt.subplots_adjust(top=0.48,
                        bottom=0.24,
                        left=0.15,
                        right=0.95,
                        hspace=1.0,
                        wspace=0.8)   
    
    for o_i, o in enumerate(["O0", "O2"]):
        
        _, summary_dfs = build_summary_dfs(res_list, names[0], o)
        summary_dfs = summary_dfs[summary_dfs["type"] != "no_simplification"]
                    
        # Draw the main plot;   
        ax = fig.add_subplot(gs[0, o_i])
        ax = sns.barplot(x="type", y="speedup", data=summary_dfs, palette=[b1, b2, b3], capsize=.1, edgecolor="#2f2f2f", estimator=gmean)
        # Set labels;
        ax.set_ylabel("Compilation Time", va="bottom", fontsize=18) 
        ax.set_yticklabels(labels=[])
        ax.set_title(f"{o} Opt. Level", fontsize=16, verticalalignment="center")
        ax.set_xlabel(None)    
        ax.set_xticklabels(labels=["Baseline", "Transformed"], rotation=45, ha="right", fontsize=18)
        
#        v_offset1 = get_upper_ci_size(summary_dfs.loc[summary_dfs["type"] == "no_simplification", "speedup"]) + 0.05
        v_offset2 = get_upper_ci_size(summary_dfs.loc[summary_dfs["type"] == "simplify_accesses", "speedup"]) + 0.04
#        speedup_ns = gmean(summary_dfs[summary_dfs["type"] == "no_simplification"]["speedup"])
        speedup_s = gmean(summary_dfs[summary_dfs["type"] == "simplify_accesses"]["speedup"])
        add_labels(ax, [speedup_s], [v_offset2], [1], fontsize=18)
        
        sns.despine(ax=ax)
        update_width(ax, 0.55)         
        # Turn off tick lines;
        ax.grid(False)
        
    fig.suptitle("Relative Compilation Time,\nGeomean", ha="left", x=0.04, y=0.89, fontsize=18)
    plt.subplots_adjust(top=0.64)
    
    # plt.savefig(os.path.join(plot_dir, "compilation_times_summary.pdf"))
    # plt.savefig(os.path.join(plot_dir, "compilation_times_summary.png"))  