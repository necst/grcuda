# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:34:37 2021

@author: albyr
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from segretini_matplottini.src.plot_utils import remove_outliers_df_iqr_grouped, compute_speedup_df, save_plot, PALETTE_G3
from load_data import PLOT_DIR

##############################
##############################

INPUT_DATE = "2021_11_02"
OUTPUT_DATE = "2021_11_02"
RES_FOLDER = "../../../../grcuda-data/results/scheduling_multiGPU"

# V100;
GPU = "V100"

SIZE = 2048

##############################
##############################

def load_data() -> (pd.DataFrame, pd.DataFrame):
    res_folder = os.path.join(RES_FOLDER, f"{INPUT_DATE}_partition_scaling")
    data = []
    for res in os.listdir(res_folder):
        size, gpus, partitions = [int(x) for x in os.path.splitext(res)[0].split("_")]
        try:
            res_data = pd.read_csv(os.path.join(res_folder, res))
            res_data["size"] = size
            res_data["gpus"] = gpus
            res_data["partitions"] = partitions
            data += [res_data]
        except pd._libs.parsers.ParserError as e:
            print(f"error parsing {res}, error={e}")      
    data = pd.concat(data, ignore_index=True)
    # Filter first few iterations;
    data = data[data["num_iter"] > 2]
    # Remove outliers;
    remove_outliers_df_iqr_grouped(data, column="computation_sec", group=["size", "gpus", "partitions"],
                                    reset_index=True, quantile=0.9, drop_index=True, debug=True)
    # Compute speedups;
    compute_speedup_df(data, key=["size"],
                       baseline_filter_col=["gpus", "partitions"], baseline_filter_val=[1, 1],  
                       speedup_col_name="speedup", time_column="computation_sec",
                       baseline_col_name="baseline_sec")
    # Obtain mean of computation times, grouped;
    data_agg = data.groupby(["size", "gpus", "partitions"]).mean()["speedup"].reset_index()
    return data, data_agg


def plot_scaling(data):
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 3
    plt.rcParams['xtick.major.pad'] = 4.2
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['axes.linewidth'] = 1
    FONTSIZE = 8
    PALETTE = [PALETTE_G3[i] for i in [1, 2, 4]]
    PALETTE[0] = "#C5E8C5"
    fig = plt.figure(figsize=(3.5, 2.5), dpi=600)
    plt.subplots_adjust(top=0.95,
                        bottom=0.2,
                        left=0.15,
                        right=0.95)
    ax = sns.lineplot(data=data, x="partitions", y="speedup", hue="gpus", legend=False, palette=PALETTE)
    # Axes labels;
    plt.xlabel("Number of partitions")
    plt.ylabel("Speedup")
    # Grid and axis limits;
    ax.yaxis.grid(True, linewidth=0.5)
    ax.xaxis.grid(False)
    # Axes limits;
    ax.set_xlim((data["partitions"].min(), data["partitions"].max()))
    ax.set_ylim((0.0, 1.75))
    # Ticks;
    ax.tick_params(length=4, width=1)
    ax.yaxis.set_major_locator(plt.LinearLocator(8))
    ax.set_yticklabels(labels=[f"{l:.2f}x" for l in ax.get_yticks()], ha="right", fontsize=FONTSIZE)
    x_ticks = sorted(list(data["partitions"].unique()))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels=x_ticks, ha="center",
                        va="top", rotation=0, fontsize=FONTSIZE)
    
    # Legend;
    labels = ["1 GPU", "2 GPUs", "4 GPUs"]
    patches = [Patch(facecolor=PALETTE[i], edgecolor="#2f2f2f", label=l, linewidth=0.5) for i, l in enumerate(labels)]
    leg = fig.legend(patches, labels, bbox_to_anchor=(0.95, 0.2), fontsize=FONTSIZE,
                     ncol=1, loc="lower right", handlelength=1.2, title=None,
                     handletextpad=0.2, columnspacing=0.5)
    leg._legend_box.align = "left"
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor('white')
    
    return fig, ax
    

##############################
##############################

if __name__ == "__main__":
    data, data_agg = load_data()
    
    # Use only some data size;
    data_agg = data_agg[data_agg["size"] == SIZE]
    
    # Plot;
    fig, ax = plot_scaling(data)    
    save_plot(PLOT_DIR, "cuda_partition_scaling" + "_{}.{}", date=OUTPUT_DATE, dpi=600)

    