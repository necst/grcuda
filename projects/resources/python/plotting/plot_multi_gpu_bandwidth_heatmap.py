# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:00:58 2022

@author: albyr
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from segretini_matplottini.src.plot_utils import COLORS, get_exp_label, get_ci_size, save_plot
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from load_data import PLOT_DIR

OUTPUT_DATE = "2022_01_19"

DEFAULT_RES_DIR = "../../connection_graph/datasets"
NUM_GPU = 8
DATASET = "connection_graph_{}_{}.csv"
V100 = "V100"
V100_DATA = DATASET.format(NUM_GPU, V100.lower())

##############################
##############################

def plot_heatmap(data: pd.DataFrame) -> (plt.Figure, plt.Axes):
    plt.rcdefaults()
    sns.set_style("white", {"ytick.left": True, "xtick.top": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams["hatch.linewidth"] = 0.3
    plt.rcParams["axes.linewidth"] = 1
    FONTSIZE = 8
    
    # 2 x 2 as we draw the CPU heatmap in the top row, https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-and-subplots-py;
    fig, axes = plt.subplots(2, 2, sharex="col", figsize=(6, 6), gridspec_kw={"width_ratios": [100, 8], "height_ratios": [15, 100]})
    gs = axes[0, 1].get_gridspec()
    # Remove the existing axes in the right column;
    for ax in axes[0:, -1]:
        ax.remove()
    plt.subplots_adjust(bottom=0.05,
                        left=0.05,
                        right=0.85)
    ax_gpu = axes[1, 0]
    ax_cpu = axes[0, 0]
    # Create a large axis;
    ax_cbar = fig.add_subplot(gs[0:, 1])

    # Do not plot CPU, we plot it separetely;
    data_gpu = data[data.index != "CPU"]
    # Mask the lower anti-triagonal matrix, excluding the main diagonal, so it's not shown;
    mask = np.zeros_like(data_gpu)
    mask[np.tril_indices_from(mask)] = True
    mask ^= np.eye(NUM_GPU).astype(bool)
    # Obtain maximum of the matrix, excluding the main diagonal;
    max_bandwidth = (data_gpu.to_numpy() - np.eye(NUM_GPU) * data_gpu.to_numpy().diagonal()).max()
    # Black outline for non-empty cells, else white;
    linecolors = ["#2f2f2f" if i <= j else (0, 0, 0, 0) for i in range(NUM_GPU) for j in range(NUM_GPU)]
    # Black and white colormap, from black to white (https://stackoverflow.com/questions/58597226/how-to-customize-the-colorbar-of-a-heatmap);
    num_colors = 200
    cm = LinearSegmentedColormap.from_list("gray-custom", ["0.1", "white"], N=num_colors)
    custom_colors = np.array([list(cm(i)) for i in np.linspace(0, 1, num_colors)])
    # Add discrete steps;
    sorted_steps_colorbar = sorted([c for c in set(data_gpu.to_numpy().reshape(-1)) if c <= max_bandwidth], reverse=True)
    for c in sorted_steps_colorbar:
        custom_colors[:int(num_colors * c / max_bandwidth) + 1, :] = cm(c / max_bandwidth)
    custom_cm = ListedColormap(custom_colors)
    # Main heatmap plot;
    ax_gpu = sns.heatmap(data_gpu, square=True, mask=mask, vmin=0, vmax=max_bandwidth, linewidth=1, 
                         linecolor=linecolors, cmap=custom_cm, ax=ax_gpu, cbar_ax=ax_cbar,
                         cbar_kws={"pad": 0.02, "ticks": [0] + sorted_steps_colorbar})
    # Add hatches to the main diagonal (https://stackoverflow.com/questions/55285013/adding-hatches-to-seaborn-heatmap-plot);
    x = np.arange(len(data_gpu.columns) + 1)
    y = np.arange(len(data_gpu.index) + 1)
    zm = np.ma.masked_less(data_gpu.values, 200)
    plt.pcolor(x, y, zm, hatch="//" * 2, alpha=0.0)
    # Add borders to the plot;
    sns.despine(ax=ax_gpu, top=False, right=False)
    # Hide axis labels;
    ax_gpu.set(xlabel=None, ylabel=None)
    # Put x-tick labels on top;
    ax_gpu.xaxis.tick_top()
    ax_gpu.xaxis.set_label_position("top")
    # Dotted lines from left to main diagonal;
    for i in range(1, NUM_GPU):
        ax_gpu.axhline(i + 0.5, xmin=0, xmax=i / NUM_GPU, color="#2f2f2f", linewidth=1, linestyle=":")
    # Add border around colorbar;
    cbar = ax_gpu.collections[0].colorbar
    for spine in cbar.ax.spines.values():
        spine.set(visible=True, linewidth=0.8, edgecolor="black")
    # Customize labels of colorbar
    cbar.ax.set_yticklabels([f"{x} GB/s" for x in [0] + sorted_steps_colorbar]) 
    
    # Draw the heatmap for the CPU;
    data_cpu = data[data.index == "CPU"]
    ax_cpu = sns.heatmap(data_cpu, square=True, vmin=0, vmax=max_bandwidth, linewidth=1, 
                        linecolor=linecolors, cmap=custom_cm, ax=ax_cpu, cbar=False)
    
    return fig

##############################
##############################

if __name__ == "__main__":
    # Read data;
    v100_data = pd.read_csv(os.path.join(DEFAULT_RES_DIR, V100_DATA), names=["from", "to", "bandwidth"], skiprows=1)
    data = v100_data
    # Round to integer;
    data["bandwidth"] = data["bandwidth"].astype(int)
    for c in ["from", "to"]:
        # Replace "-1" with CPU and other numbers with the GPU name;
        data[c].replace({-1: "CPU", **{i: f"GPU{i}" for i in range(NUM_GPU)}}, inplace=True)
        # Use categorical labels for devices;
        data[c] = pd.Categorical(data[c], categories=["CPU"] + [f"GPU{i}" for i in range(NUM_GPU)], ordered=True)
    # Sort values;
    data.sort_values(["from", "to"], inplace=True)
    # Turn the dataframe into a matrix;
    data_matrix = data.pivot(index="from", columns="to", values="bandwidth")
    # Plot heatmap;
    fig = plot_heatmap(data_matrix)
    save_plot(PLOT_DIR, "cuda_partition_scaling_minmax_2gpu" + "_{}.{}", date=OUTPUT_DATE, dpi=600)

