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
import matplotlib.lines as lines


def remove_outliers(data, threshold=3):
    sizes = set(data["num_elements"])
    types_k = ["time_m_k_ms", "time_u_k_ms"]
    types = ["time_m_ms", "time_u_ms"]
    
    data_list = []
    
    for sim in ["no_simplification", "simplify_accesses"]:
        for o_i, o in enumerate(["O0", "O2"]):
            for s in sizes:
                temp_data = data[(data["opt_level"] == o) & (data["num_elements"] == s) & (data["simplify"] == sim)]
                for t in types_k:
                    temp_data = temp_data[st.zscore(temp_data[t]) < 3]
                for t in types:
                    temp_data = temp_data[st.zscore(temp_data[t]) < 3]
                data_list += [temp_data]
            
    return pd.concat(data_list)


# Compute speedup of v2 w.r.t v1;
def compute_speedup(X, col_slow, col_fast, col_speedup):
    X[col_speedup] = X[col_slow] / X[col_fast]

# Used to add numerical labels above barplots;
def add_labels(ax, labels=None, vertical_offsets=None, patch_num=None, fontsize=14, rotation=0):
    
    if not vertical_offsets:
        vertical_offsets = [0] * len(ax.patches)
    if not labels:
        labels = ["{:.2f}".format(p.get_height()) for p in ax.patches]
    patches = []
    if not patch_num:
        patches = ax.patches
    else:
        patches = [p for i, p in enumerate(ax.patches) if i in patch_num]
    
    # Iterate through the list of axes' patches
    lab_num = 0
    for p in patches:
        if labels[lab_num]:
            ax.text(p.get_x() + p.get_width()/2., vertical_offsets[lab_num] + p.get_height(), "{:.2f}x".format(labels[lab_num]), 
                    fontsize=fontsize, color="#2f2f2f", ha='center', va='bottom', rotation=rotation)
        lab_num += 1

# Compute the size of the upper 0.95 interval, i.e. the size between the top of the bar and the top of the error bar;
def get_upper_ci_size(x, ci=0.95):
    ci_upper = st.t.interval(ci, len(x)-1, loc=np.mean(x), scale=st.sem(x))[1]
    return ci_upper - np.mean(x)

# Update width of bars;
def update_width(ax, width=1):
    for i, patch in enumerate(ax.patches):
        current_width = patch.get_width()
        diff = current_width - width
        # Change the bar width
        patch.set_width(width)
        # Recenter the bar
        patch.set_x(patch.get_x() + 0.5 * diff)
        
def build_plot(data, fig, position, ax0=None, title="", vlabel_offset=0.4):
    kernel_res = pd.DataFrame({"Type": ["Original"] * len(data) + ["Modified"] * len(data),
                               "time_ms": list(data["time_u_k_ms"]) + list(data["time_m_k_ms"])})
    tot_res = pd.DataFrame({"Type": ["Original"] * len(data) + ["Modified"] * len(data),
                               "time_ms": list(data["time_u_ms"]) + list(data["time_m_ms"])})
    speedup = 1 / np.median(data["speedup"])
    
    # Add a barplot;
    ax = fig.add_subplot(position, sharey=ax0)
    ax = sns.barplot(x="Type", y="time_ms", data=tot_res, palette=[b3, b1], capsize=.1, ax=ax, edgecolor="#2f2f2f")
    ax = sns.barplot(x="Type", y="time_ms", data=kernel_res, palette=[r1, r3], capsize=.1, ax=ax, edgecolor="#2f2f2f", ci=None)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Exec. Time [ms]", fontsize=16)  
    ax.set_xlabel("Type", fontsize=16) 
    # Compute vertical labels offsets, using the speedup variance;
    v_offset = get_upper_ci_size(tot_res.loc[tot_res["Type"] == "Modified", "time_ms"]) + vlabel_offset
    add_labels(ax, [speedup], [v_offset], [1])
    update_width(ax, 0.5)
    sns.despine(ax=ax)
    
    return ax, kernel_res, tot_res, speedup


def build_plot_simple(data,
                      fig,
                      position,
                      gs,
                      ax0=None,
                      o="O0",
                      s="no_simplification",
                      title="",
                      vlabel_offset=0.4,
                      time_cols="time_m_k_ms",
                      speedup_label=True):
    
    curr_res = data[(data["opt_level"] == o) & (data["simplify"] == s) & (data["num_elements"] == max(data["num_elements"]))]     

    res = pd.DataFrame({"Type": ["Original"] * len(curr_res) + ["Modified"] * len(curr_res),
                               "time_ms": list(curr_res[time_cols[0]]) + list(curr_res[time_cols[1]])})
    speedup = np.median(curr_res["speedup"])
    
    # Add a barplot;
    ax = fig.add_subplot(gs[position // 3, position % 3], sharey=ax0)
    ax = sns.barplot(x="Type", y="time_ms", data=res, palette=[r1, r3], capsize=.1, ax=ax, edgecolor="#2f2f2f")
    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Exec. Time [ms]", fontsize=16)  
    ax.set_xlabel("Type", fontsize=16) 
    
    ax.set_xticklabels(["Manually\nmodified", "Automatically\nmodified"])
    # Compute vertical labels offsets, using the speedup variance;
    if speedup_label:
        v_offset = get_upper_ci_size(res.loc[res["Type"] == "Modified", "time_ms"]) + vlabel_offset
        add_labels(ax, [speedup], [v_offset], [1])
        
    update_width(ax, 0.5)
    sns.despine(ax=ax)
    
    return ax, res, speedup    

def load_data(path):
    res = pd.read_csv(path, sep=", ") 
    
    res["speedup"] = 1
    res["speedup_k"] = 1
    compute_speedup(res, "exec_time_u_k_us", "exec_time_m_k_us", "speedup_k")
    compute_speedup(res, "exec_time_u_us", "exec_time_m_us", "speedup")
    
    res["time_u_k_ms"] = res["exec_time_u_k_us"] / 1000 
    res["time_m_k_ms"] = res["exec_time_m_k_us"] / 1000
    res["time_u_ms"] = res["exec_time_u_us"] / 1000
    res["time_m_ms"] = res["exec_time_m_us"] / 1000
    
    return res

def draw_plot(res, fig, gs, plot_vertical_pos, name, vlabel_offset=0.15):
    plot_num = 0
    ax0 = None
    for o in ["O0", "O2"]:
        for s in ["no_simplification", "simplify_accesses"]:
            # Build the required data view;
            curr_res = res[(res["opt_level"] == o) & (res["simplify"] == s) & (res["num_elements"] == max(res["num_elements"]))]              
            ax, _, _, _ = build_plot(curr_res, fig, gs[plot_vertical_pos, plot_num], ax0,
                                     f"{name},\n{o},\n{'with access merging' if s == 'simplify_accesses' else 'no access merging'}",
                                     vlabel_offset=vlabel_offset)
            if plot_num == 0:
                ax0 = ax
            plot_num += 1
            

def build_summary_dfs(res_list, name="", o="", boundary_merging_statistics=None):
    
    temp_dfs = []
    for i, res in enumerate(res_list):
        print(f"Printing {name}, {o}")
        res_temp = res[(res["opt_level"] == o) & (res["num_elements"] == max(res["num_elements"]))].reset_index()  
        
        # Unmodified median time;
        median_time_u_ns = np.median(res_temp[res_temp["simplify"] == "no_simplification"]["time_u_ms"])
        median_time_u_s = np.median(res_temp[res_temp["simplify"] == "simplify_accesses"]["time_u_ms"])
        # Obtain the normalized unmodified time;
        unmodified_times_normalized_k = list(res_temp[res_temp["simplify"] == "no_simplification"]["time_u_k_ms"] / median_time_u_ns) \
            + list(res_temp[res_temp["simplify"] == "simplify_accesses"]["time_u_k_ms"] / median_time_u_s)
        unmodified_times_normalized = list(res_temp[res_temp["simplify"] == "no_simplification"]["time_u_ms"] / median_time_u_ns) \
            + list(res_temp[res_temp["simplify"] == "simplify_accesses"]["time_u_ms"] / median_time_u_s)
        speedups_k_normalized = list(res_temp["time_m_k_ms"] / res_temp["time_u_k_ms"])
        times_k_normalized = list(res_temp["time_m_k_ms"] / res_temp["time_u_ms"])
        times_normalized = list(res_temp["time_m_ms"] / res_temp["time_u_ms"])
                
        res_new = pd.DataFrame({"kernel": [res_temp["kernel"].iloc[0]] * (len(unmodified_times_normalized) + len(times_k_normalized)),
                                  "type": len(unmodified_times_normalized)  * ["Unmodified"] + list(res_temp["simplify"]),
                                  "time_k_ms": unmodified_times_normalized_k + times_k_normalized,
                                  "time_ms": unmodified_times_normalized + times_normalized,
                                  "speedup_k": unmodified_times_normalized_k + speedups_k_normalized})
       
        res_new = res_new[st.zscore(res_new["time_k_ms"]) < 3]
        res_new = res_new[st.zscore(res_new["time_ms"]) < 3] 
        temp_dfs += [res_new]
               
    # Compute speedups and summary plot;
    speedups_ns = []
    speedups_s = []
    speedups_k_ns = []
    speedups_k_s = []
    speedups_u = []
    speedups_k_u = []
    real_speedups_k_ns = []
    real_speedups_k_s = []
    for i, df in enumerate(temp_dfs):
        speedups_ns += [gmean(df[df["type"] == "no_simplification"]["time_ms"])]
        speedups_s += [gmean(df[df["type"] == "simplify_accesses"]["time_ms"])]
        speedups_k_ns += [gmean(df[df["type"] == "no_simplification"]["time_k_ms"])]
        speedups_k_s += [gmean(df[df["type"] == "simplify_accesses"]["time_k_ms"])]
        real_speedups_k_ns += [gmean(df[df["type"] == "no_simplification"]["speedup_k"])]
        real_speedups_k_s += [gmean(df[df["type"] == "simplify_accesses"]["speedup_k"])]
        speedups_u += [gmean(df[df["type"] == "Unmodified"]["time_ms"])]
        speedups_k_u += [gmean(df[df["type"] == "Unmodified"]["speedup_k"])]
    summary_dfs = pd.DataFrame({"kernel": [name] * len(temp_dfs) * 3,
                                "type": ["Unmodified"] * len(temp_dfs) + ["no_simplification"] * len(temp_dfs) + ["simplify_accesses"] * len(temp_dfs),
                                "time_k_ms": speedups_k_u + speedups_k_ns + speedups_k_s,
                                "time_ms": speedups_u + speedups_ns + speedups_s,
                                "speedup_k": [1] * len(temp_dfs) + real_speedups_k_ns + real_speedups_k_s})
    temp_dfs = [summary_dfs] + temp_dfs
    
    # Remove speedups of datas that do not have any merging being done;
    for res_temp in temp_dfs:
        if boundary_merging_statistics is not None \
                and res_temp["kernel"].iloc[0] in set(boundary_merging_statistics["kernel"]) \
                and not boundary_merging_statistics[boundary_merging_statistics["kernel"] == res_temp["kernel"].iloc[0]]["plot_merged"].iloc[0]:
            res_temp.loc[res_temp["type"] == "simplify_accesses", "time_k_ms"] = 0.000000000001
            res_temp.loc[res_temp["type"] == "simplify_accesses", "time_ms"] = 0.000000000001
            res_temp.loc[res_temp["type"] == "simplify_accesses", "speedup_k"] = 0.000000000001
    
    return temp_dfs, summary_dfs
        

# Define some colors;
c1 = "#b1494a"
c2 = "#256482"
c3 = "#2f9c5a"
c4 = "#28464f"

r4 = "#CE1922"
r3 = "#F41922"
r2 = "#FA3A51"
r1 = "#FA4D4A"

b1 = "#97E6DB"
b2 = "#C6E6DB"
b3 = "#CEF0E4"
b4 = "#8FA69E"

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
    
    res_folder = "../../../data/results/with_lower_bounds/2019_10_23"
    plot_dir = "../../../data/plots/with_lower_bounds/2020_02_13"
    
    res_axpy = load_data(os.path.join(res_folder, "axpy_2019_10_23_16_55_54.csv"))
    res_dp = load_data(os.path.join(res_folder, "dot_product_2019_10_23_16_55_54.csv"))
    res_conv = load_data(os.path.join(res_folder, "convolution_2019_10_23_16_55_54.csv"))
    res_mmul = load_data(os.path.join(res_folder, "mmul_2019_10_23_16_55_54.csv"))
    res_autocov = load_data(os.path.join(res_folder, "autocov_2019_10_23_16_55_54.csv"))
    res_hotspot = load_data(os.path.join(res_folder, "hotspot_2019_10_23_16_55_54.csv"))
    res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d_2019_10_23_16_55_54.csv"))
    res_bb = load_data(os.path.join(res_folder, "backprop_2019_10_23_16_55_54.csv"))
    res_bb2 = load_data(os.path.join(res_folder, "backprop2_2019_10_23_16_55_54.csv"))
    res_bfs = load_data(os.path.join(res_folder, "bfs_2019_10_23_16_55_54.csv"))
    res_pr = load_data(os.path.join(res_folder, "pr_2019_10_23_16_55_54.csv"))  
    res_gaussian = load_data(os.path.join(res_folder, "gaussian_2019_10_23_16_55_54.csv"))
    res_histogram = load_data(os.path.join(res_folder, "histogram_2019_10_23_16_55_54.csv"))
    res_lud = load_data(os.path.join(res_folder, "lud_2019_10_23_16_55_54.csv"))
    res_needle = load_data(os.path.join(res_folder, "needle_2019_10_23_16_55_54.csv"))
    res_nested = load_data(os.path.join(res_folder, "nested_2019_10_23_16_55_54.csv"))

    res_list = [res_axpy, res_dp, res_conv, res_mmul, res_autocov, res_hotspot, res_hotspot3d,
                res_bb, res_bb2, res_bfs, res_pr, res_nested, res_gaussian,
                res_histogram, res_lud, res_needle]
    
    res_list = [remove_outliers(res) for res in res_list]
    
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
             "NN - Forward Phase", "NN - Backpropagation", "BFS", "PageRank", "Nested Loops", "Gaussian Elimination",
             "Histogram", "LU Decomposition", "Needleman-Wunsch"]
    vlabel_offsets= [0.4, 0.10, 0.10, 0.15, 0.5, 0.07, 0.2,
                     0.5, 0.5, 0.5, 0.5, 0.5,
                     0.2, 0.1, 0.2, 0.2]
    
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
    
    # plt.savefig(os.path.join(plot_dir, "exec_times.pdf"))
    # plt.savefig(os.path.join(plot_dir, "exec_times.png"))         
    
    
    #%
    
    ##################################
    # Small summary plot #############
    ##################################
    
#    num_col = 3
#    num_row = 3
#    fig = plt.figure(figsize=(num_col * 4.0, num_row * 5))
#    gs = gridspec.GridSpec(num_row, num_col)
#    plt.subplots_adjust(top=0.95,
#                        bottom=0.1,
#                        left=0.11,
#                        right=0.95,
#                        hspace=1.0,
#                        wspace=0.6)    
#
#    o_list = ["O1", "O0", "O0", "O0", "O0", "O1", "O0", "O0", "O0"]
#    s_list = ["simplify_accesses", "no_simplification", "simplify_accesses", "simplify_accesses", "simplify_accesses", "no_simplification", "simplify_accesses", "simplify_accesses", "simplify_accesses"]
#    
#    for i, res in enumerate(res_list):
#        build_plot_simple(res, fig, i, gs, None, o_list[i], s_list[i], names[i], vlabel_offsets[i], time_cols=["time_u_k_ms", "time_m_k_ms"], speedup_label=True)                                       
#
#    plt.savefig(os.path.join(plot_dir, "exec_times_small.pdf"))
#    plt.savefig(os.path.join(plot_dir, "exec_times_small.png"))   
    
    #%
    
    ##################################
    # Other summary plot #############
    ##################################
    
    res_axpy = load_data(os.path.join(res_folder, "axpy_2019_10_23_19_41_23.csv"))
    res_dp = load_data(os.path.join(res_folder, "dot_product_2019_10_23_16_55_54.csv"))
    res_conv = load_data(os.path.join(res_folder, "convolution_2019_10_23_17_42_09.csv"))
    res_mmul = load_data(os.path.join(res_folder, "mmul_2019_10_23_17_42_09.csv"))
    res_autocov = load_data(os.path.join(res_folder, "autocov_2019_10_23_16_55_54.csv"))
    res_hotspot = load_data(os.path.join(res_folder, "hotspot_2019_10_23_17_42_09.csv"))
    res_hotspot3d = load_data(os.path.join(res_folder, "hotspot3d_2019_10_23_19_41_23.csv"))
    res_bb = load_data(os.path.join(res_folder, "backprop_2019_10_23_19_41_23.csv"))
    res_bb2 = load_data(os.path.join(res_folder, "backprop2_2019_10_23_16_55_54.csv"))
    res_bfs = load_data(os.path.join(res_folder, "bfs_2019_10_23_19_41_23.csv"))
    res_pr = load_data(os.path.join(res_folder, "pr_2019_10_23_16_55_54.csv"))  
    res_gaussian = load_data(os.path.join(res_folder, "gaussian_2019_10_23_16_55_54.csv"))
    res_histogram = load_data(os.path.join(res_folder, "histogram_2019_10_23_17_42_09.csv"))
    res_lud = load_data(os.path.join(res_folder, "lud_2019_10_23_17_42_09.csv"))
    res_needle = load_data(os.path.join(res_folder, "needle_2019_10_23_16_55_54.csv"))
    res_nested = load_data(os.path.join(res_folder, "nested_2019_10_23_19_41_23.csv"))
    
    boundary_merging_statistics = pd.read_csv("../../../data/results/access_merging_statistics.csv")
    boundary_merging_statistics["plot_merged"] = True # boundary_merging_statistics["unprotected_num_of_accesses"] > boundary_merging_statistics["merged_num_of_accesses"]
    
    res_list = [res_axpy, res_dp, res_conv, res_autocov, res_hotspot3d, res_bb, res_bfs, res_pr, res_nested, res_mmul, res_hotspot, 
                 res_bb2, res_gaussian,
                res_histogram, res_lud, res_needle, ]
    
#    res_list = [remove_outliers(res) for res in res_list]
    
    names = ["Summary,\nGeomean", "AXPY", "DP", "CONV", "ACV", "HP3D",
             "NN1", "BFS", "PR", "NEST", "MULT", "HP",  "NN2",  "GE",
             "HIST", "LU", "NW"]
    
    boundary_merging_statistics["kernel"] = names[1:] 
    plt.rcParams['axes.titlepad'] = 90 
    
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    num_plots = len(res_list) + 3
    fig = plt.figure(figsize=(num_plots * 0.45 * 5, 5.5 * 2 + 14))
    gs = gridspec.GridSpec(2, num_plots)
    
    plt.subplots_adjust(top=0.78, left=0.035, bottom=0.11, wspace=0.15, right=0.985, hspace=1.0) 
    
    for i, res in enumerate(res_list):
        res["kernel"] = names[i + 1]
        
    def plot_merge(i):
        return i == 0 or boundary_merging_statistics[boundary_merging_statistics["kernel"] == names[i]]["plot_merged"].iloc[0]
        
    for o_i, o in enumerate(["O0", "O2"]):
        
        temp_dfs, summary_dfs = build_summary_dfs(res_list, names[0], o, boundary_merging_statistics)
        temp_dfs = [df[df["type"] != "no_simplification"] for df in temp_dfs]
                    
        # Draw the main plot;   
        ax0 = None
        for i, df in enumerate(temp_dfs):
            plot_position = i if i == 0 else i + 1
            plot_position = plot_position if plot_position < 11 else plot_position + 1 
            ax = fig.add_subplot(gs[o_i, plot_position], sharey=ax0)
            ax = sns.barplot(x="type", y="time_ms", data=df[df["kernel"] == names[i]], palette=[b1, b2, b3], capsize=.1, edgecolor="#2f2f2f", estimator=gmean)
            ax = sns.barplot(x="type", y="time_k_ms", data=df[df["kernel"] == names[i]], palette=[r3, r2, r1], ci=None, ax=ax, edgecolor="#2f2f2f", estimator=gmean)
            # Set labels;
            if i in [0, 1, 10]:
                ax.set_ylabel("Relative Exec. Time", va="bottom", fontsize=38) 
                ax0 = ax
            else:
                ax.set_ylabel(None) 
            ax.set_yticklabels(labels=[])
            ax.set_title(names[i], fontsize=36, verticalalignment="center")
            ax.set_xlabel(None)    
#            ax.set_xticklabels(labels=["Original", "Modified, no simp.", "Modified, simp."], rotation=45, ha="right", fontsize=24)
            ax.set_xticklabels(labels=["B", "T"], rotation=0, ha="center", fontsize=36)
            
#            v_offset1 = get_upper_ci_size(df.loc[df["type"] == "no_simplification", "time_ms"]) + 0.05
            v_offset2 = get_upper_ci_size(df.loc[df["type"] == "simplify_accesses", "time_ms"]) + 0.05
            speedup_ns = gmean(df[df["type"] == "no_simplification"]["speedup_k"])
            if plot_merge(i):
                speedup_s = gmean(df[df["type"] == "simplify_accesses"]["speedup_k"])
#                ax.set_xticklabels(labels=["Original", "Modified, no simp.", "Modified, simp."], rotation=45, ha="right", fontsize=25)
                ax.set_xticklabels(labels=["B", "T"], rotation=0, ha="center", fontsize=36)
            else:
                speedup_s = None
#                ax.set_xticklabels(labels=["Original", "Modified, no simp.", ""], rotation=45, ha="right", fontsize=25)
                ax.set_xticklabels(labels=["B", "T"], rotation=0, ha="center", fontsize=36)
            add_labels(ax, [speedup_s], [v_offset2], [1], fontsize=37, rotation=0)
            
            sns.despine(ax=ax)
            update_width(ax, 0.5)      
            # Turn off tick lines;
            ax.grid(False)
     
    # Add separation lines;
    lh = lines.Line2D([0.01, 0.99], [0.48, 0.48], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--")
    lv = lines.Line2D([0.605, 0.605], [0.05, 0.95], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--")
    lv2 = lines.Line2D([0.102, 0.102], [0.05, 0.95], transform=fig.transFigure, figure=fig, color="#2f2f2f", linestyle="--")
    fig.get_axes()[0].lines.extend([lh, lv, lv2])
        
    fig.get_axes()[0].annotate("O0 Optimization Level", xy=(0, 1), xycoords="axes fraction", fontsize=48, ha="left", 
                     textcoords="offset points", xytext=(-80, 180), backgroundcolor="white")
    fig.get_axes()[num_plots - 2].annotate("O2 Optimization Level", xy=(0, 1), xycoords="axes fraction", fontsize=48, ha="left", 
                     textcoords="offset points", xytext=(-80, 180), backgroundcolor="white")
    
    fig.get_axes()[0].annotate("vs. Manually Modified Kernels", xy=(.32, .04), xycoords="figure fraction", fontsize=48, ha="center")
    fig.get_axes()[0].annotate("vs. Original Kernels", xy=(.77, .04), xycoords="figure fraction", fontsize=48, ha="center")
    
        
    # Add legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Overall Time"),
                    Patch(facecolor=r3, edgecolor="#2f2f2f", label="Kernel Time")]
    ax = fig.get_axes()[0]
    leg = fig.legend(custom_lines, ["Overall Time", "Kernel Time"],
                             bbox_to_anchor=(0.99, 1), fontsize=44)
    leg.set_title("Exec. Time Group", prop={"size": 48})
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
    
    plt.savefig(os.path.join(plot_dir, "exec_times_tot_2.pdf"))
    plt.savefig(os.path.join(plot_dir, "exec_times_tot_2.png"))  
    
    #%%
    
    ##################################
    # Another summary plot ###########
    ##################################
    plt.rcParams['axes.titlepad'] = 30 
    
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
        ax = sns.barplot(x="type", y="time_ms", data=summary_dfs, palette=[b1, b2, b3], capsize=.1, edgecolor="#2f2f2f", estimator=gmean)
        ax = sns.barplot(x="type", y="time_k_ms", data=summary_dfs, palette=[r3, r2, r1], ci=None, ax=ax, edgecolor="#2f2f2f", estimator=gmean)
        # Set labels;
        ax.set_ylabel("Relative Exec. Time", va="bottom", fontsize=18) 
        ax.set_yticklabels(labels=[])
        ax.set_title(f"{o} Opt. Level", fontsize=16, verticalalignment="center")
        ax.set_xlabel(None)    
        ax.set_xticklabels(labels=["Baseline", "Transformed"], rotation=45, ha="right", fontsize=18)
        
#        v_offset1 = get_upper_ci_size(summary_dfs.loc[summary_dfs["type"] == "no_simplification", "speedup_k"]) + 0.05
        v_offset2 = get_upper_ci_size(summary_dfs.loc[summary_dfs["type"] == "simplify_accesses", "speedup_k"]) + 0.02
#        speedup_ns = gmean(summary_dfs[summary_dfs["type"] == "no_simplification"]["speedup_k"])
        speedup_s = gmean(summary_dfs[summary_dfs["type"] == "simplify_accesses"]["speedup_k"])
        add_labels(ax, [speedup_s], [v_offset2], [1], fontsize=18)
        
        sns.despine(ax=ax)
        update_width(ax, 0.5)      
        # Turn off tick lines;
        ax.grid(False)
        
    # Add legend;
    custom_lines = [Patch(facecolor=b1, edgecolor="#2f2f2f", label="Overall Time"),
                    Patch(facecolor=r3, edgecolor="#2f2f2f", label="Kernel Time")]
    ax = fig.get_axes()[0]
    leg = fig.legend(custom_lines, ["Overall Time", "Kernel Time"],
                             bbox_to_anchor=(1, 1), fontsize=16)
    leg.set_title("Exec. Time Group", prop={"size": 16})
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
        
    fig.suptitle("Kernel Relative\nExec. Time,\nGeomean", ha="left", x=0.04, y=0.92, fontsize=18)
    plt.subplots_adjust(top=0.64)
    
    plt.savefig(os.path.join(plot_dir, "exec_times_summary.pdf"))
    plt.savefig(os.path.join(plot_dir, "exec_times_summary.png"))  