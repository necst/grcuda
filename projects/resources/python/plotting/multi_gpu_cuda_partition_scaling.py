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
from segretini_matplottini.src.plot_utils import remove_outliers_df_iqr_grouped, compute_speedup_df

##############################
##############################

INPUT_DATE = "2021_11_02"
OUTPUT_DATE = "2021_11_03"
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
    # Obtain mean of computation times, grouped;
    data_agg = data.groupby(["size", "gpus", "partitions"]).mean()["computation_sec"].reset_index()
    # Compute speedups;
    compute_speedup_df(data_agg, key=["size"],
                       baseline_filter_col=["gpus", "partitions"], baseline_filter_val=[1, 1],  
                       speedup_col_name="speedup", time_column="computation_sec",
                       baseline_col_name="baseline_sec")
    return data, data_agg


def plot_scaling(data):
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": False})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['xtick.major.pad'] = 4.2
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['axes.linewidth'] = 0.5
    
    fig = plt.figure(figsize=(2, 0.95), dpi=600)
    plt.subplots_adjust(top=0.78,
                        bottom=0.17,
                        left=0.12,
                        right=0.99,
                        hspace=0.15,
                        wspace=0.8)
    ax = plt.plot(data["partitions"], data["speedup"]) # sns.lineplot(data=data, x="partitions", y="speedup", hue="gpus")
    return fig, ax
    

##############################
##############################

if __name__ == "__main__":
    data, data_agg = load_data()
    
    # Use only some data size;
    data_agg = data_agg[data_agg["size"] == SIZE]
    
    # Plot;
    fig, ax = plot_scaling(data_agg)
    
    
    