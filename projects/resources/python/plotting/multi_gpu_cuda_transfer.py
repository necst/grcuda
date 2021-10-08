# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:54:09 2021

@author: albyr
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from segretini_matplottini.src.plot_utils import *
from multi_gpu_parse_nvprof_log import INPUT_FOLDER, create_nondirectional_transfer_matrix
from load_data import PLOT_DIR, DEFAULT_RES_CUDA_DIR

##############################
##############################

OUTPUT_DATE = "2021_10_06"

BENCHMARKS = [
    # "b1",
    "b5",
    "b6",
    "b6_4",
    "b11"
    ]

EDGE = 1
ANGLE = 35
FORSHORTENING = 1 / 3  # Proportional scaling of the axonometry;
X_STEP = EDGE * FORSHORTENING
Y_STEP = EDGE * FORSHORTENING * np.tan(np.deg2rad(ANGLE))
CPU_VSTEP = EDGE / 3
 
POINTS = [
    # Front face;
    [0, 0],  # 0: Lower left;
    [0, EDGE],  # 1: Upper left;
    [EDGE, EDGE],  # 2: Upper right;
    [EDGE, 0],  # 3: Lower right;
    # Right face;
    [EDGE + X_STEP, Y_STEP],  # 4: Lower
    [EDGE + X_STEP, EDGE + Y_STEP],  # 5: Upper
    # 6: Lower left corner;
    [X_STEP, Y_STEP],
    # 7: Upper left corner;
    [X_STEP, EDGE + Y_STEP], 
    ]

# Associate GPUs to points;
GPU = {
    0: POINTS[2],   
    1: POINTS[1],   
    2: POINTS[7],   
    3: POINTS[5],   
    4: POINTS[6],   
    5: POINTS[4],   
    6: POINTS[3],   
    7: POINTS[0],   
    }

CPU_POINTS = [
    [EDGE / 2 + X_STEP / 2, Y_STEP * FORSHORTENING - CPU_VSTEP],
    [EDGE / 2 + X_STEP / 2, EDGE + Y_STEP * FORSHORTENING + CPU_VSTEP],
]

CPU = {
   0: CPU_POINTS[0],    
   1: CPU_POINTS[1], 
   }

##############################
##############################

def setup(fig=None, ax=None):
    if fig == None and ax == None:
        plt.rcdefaults()
        plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
        # Create figure and axes
        fig, ax = plt.subplots()
    
    # Have axes with the same scale;
    plt.xlim(-0.6, EDGE * 3/2 + 0.4)
    plt.ylim(-0.6, EDGE * 3/2 + 0.4)
    ax.set_aspect("equal")
    plt.axis("off")
    return fig, ax


# Draw corners of the cube;
def draw_points(ax, alpha_scale=1):
    # Obtain list of coordinates;
    x = [x[0] for x in POINTS]
    y = [y[1] for y in POINTS]
    ax.scatter(x, y, color="#2f2f2f", alpha=alpha_scale, zorder=10)
    return ax

# Draw names of GPUs;
def draw_gpu_names(ax):
    x_offset = {
        2: -EDGE / 5,
        3: -EDGE / 10,
        6: -EDGE / 8,
        7: -EDGE / 5,
        }
    y_offset = {
        0: -EDGE / 10,
        1: -EDGE / 10,
        2: EDGE / 25,
        3: EDGE / 25,
        6: -EDGE / 8,
        7: -EDGE / 8,
        }
    for i, (g, p) in enumerate(GPU.items()):
        x = p[0] + EDGE * 0.02 + (x_offset[g] if g in x_offset else 0)
        y = p[1] + EDGE * 0.01 + (y_offset[g] if g in y_offset else EDGE / 70)
        ax.annotate(f"GPU{g}", xy=(x, y), color="#2f2f2f", fontsize=10, ha="left")
    return ax


# Draw a single line between GPUs;
def draw_line_gpu(ax, x, y, style):
        x = GPU[x]
        y = GPU[y]
        ax.plot((x[0], y[0]), (x[1], y[1]), **style)
        
        
# Join corners;
def draw_edges(ax, alpha_scale=1):    
    # Double NVLink;
    style_nv2 = dict(
        linewidth=2,
        linestyle="-",
        color="#2f2f2f",
        alpha=0.9 * alpha_scale,
        solid_capstyle="round",
    )
    # Single NVLink;
    style_nv1 = dict(
        linewidth=0.8,
        linestyle="--",
        color="#2f2f2f",
        alpha=0.7 * alpha_scale,
        solid_capstyle="round",
    )
    # Missing edge is PCIe;  
    
    # Connect GPUs;
    draw_line_gpu(ax, 0, 1, style_nv1)
    draw_line_gpu(ax, 0, 2, style_nv1)
    draw_line_gpu(ax, 0, 3, style_nv2)
    draw_line_gpu(ax, 0, 6, style_nv2)
    draw_line_gpu(ax, 1, 2, style_nv2)
    draw_line_gpu(ax, 1, 3, style_nv1)
    draw_line_gpu(ax, 1, 7, style_nv2)
    draw_line_gpu(ax, 2, 3, style_nv2)
    draw_line_gpu(ax, 2, 4, style_nv1)
    draw_line_gpu(ax, 3, 5, style_nv1)
    draw_line_gpu(ax, 4, 5, style_nv2)
    draw_line_gpu(ax, 4, 6, style_nv1)
    draw_line_gpu(ax, 4, 7, style_nv2)
    draw_line_gpu(ax, 5, 6, style_nv2)
    draw_line_gpu(ax, 5, 7, style_nv1)
    draw_line_gpu(ax, 6, 7, style_nv1)
    
    return ax


# Draw faces of the cube;
def draw_faces(ax):
    style = dict(
        linewidth=1,
        linestyle="--",
        edgecolor="#2f2f2f",
        facecolor="#2f2f2f",
        alpha=0.1,
    )
    patches_list = [
        patches.Polygon(xy=[POINTS[0], POINTS[1], POINTS[2], POINTS[3]], **style),
        patches.Polygon(xy=[POINTS[2], POINTS[5], POINTS[4], POINTS[3]], **style),
        patches.Polygon(xy=[POINTS[2], POINTS[5], POINTS[7], POINTS[1]], **style),
        patches.Polygon(xy=[POINTS[0], POINTS[3], POINTS[4], POINTS[6]], **style),
        patches.Polygon(xy=[POINTS[0], POINTS[1], POINTS[7], POINTS[6]], **style),
        patches.Polygon(xy=[POINTS[6], POINTS[4], POINTS[5], POINTS[7]], **style),
        ]
    for p in patches_list:
        ax.add_patch(p)
    return ax


def draw_cpu_points(ax, alpha_scale=1): 
    # Obtain list of coordinates;
    x = [x[0] for x in CPU_POINTS]
    y = [y[1] for y in CPU_POINTS]
    ax.scatter(x, y, color="#888888", alpha=alpha_scale, zorder=10)
    return ax


def draw_pci(ax, gpu0, gpu1, vertical_start, upper=True, style=None):
    medium_point = [(gpu0[0] + gpu1[0]) / 2, gpu0[1]]
    x_step = X_STEP / 2 - gpu0[0]
    y_step = Y_STEP * FORSHORTENING + CPU_VSTEP * (1 if upper else -1)
    cpu_point = [medium_point[0] + x_step, vertical_start + y_step]    
    
    t = np.sqrt(y_step**2 + x_step**2) / 2
    alpha = np.arctan(y_step / np.abs(x_step))
    y_offset = np.sin(alpha) * t
    
    x_offset = (cpu_point[0] - medium_point[0]) / 2
    split_point = [medium_point[0] + x_offset, vertical_start + y_offset + (gpu0[1] - vertical_start) / 2]
    
    ax.plot((cpu_point[0], split_point[0]), (cpu_point[1], split_point[1]), **style)
    ax.plot((split_point[0], gpu0[0]), (split_point[1], gpu0[1]), **style)
    ax.plot((split_point[0], gpu1[0]), (split_point[1], gpu1[1]), **style)  
        

def draw_cpu_lines(ax, alpha_scale=1):
    style = dict(
        color="#888888",
        alpha=0.8 * alpha_scale,
        linestyle="-",
        linewidth=1,
        solid_capstyle="round",
    )
                
    draw_pci(ax, GPU[1], GPU[0], EDGE, style=style)
    draw_pci(ax, GPU[2], GPU[3], EDGE, style=style)
    draw_pci(ax, GPU[7], GPU[6], 0, False, style=style)
    draw_pci(ax, GPU[4], GPU[5], 0, False, style=style)
    return ax


# Draw names of CPUs;
def draw_cpu_names(ax):
    y_offset = {
        0: -EDGE / 10,
        }
    for c in [0, 1][::-1]:
        p = CPU_POINTS[c]
        x = p[0] + EDGE * 0.02
        y = p[1] + EDGE * 0.01 + (y_offset[c] if c in y_offset else EDGE / 70)
        ax.annotate(f"CPU{c}", xy=(x, y), color="#888888", fontsize=10, ha="left")
    return ax


# Draw the GPU topology;
def draw_topology(fig=None, ax=None, **kwargs):
    fig, ax = setup(fig, ax)
    ax = draw_cpu_lines(ax, **kwargs)
    ax = draw_edges(ax, **kwargs)
    # ax = draw_faces(ax)
    ax = draw_points(ax, **kwargs)
    ax = draw_cpu_points(ax, **kwargs)
    ax = draw_gpu_names(ax)
    ax = draw_cpu_names(ax)
    return fig, ax


def draw_pci_transfer(ax, cpu, gpu, other_gpu, vertical_start, upper=True, style=None):
    medium_point = [(gpu[0] + other_gpu[0]) / 2, gpu[1]]
    x_step = X_STEP / 2 - min(gpu[0], other_gpu[0])
    y_step = Y_STEP * FORSHORTENING + CPU_VSTEP * (1 if upper else -1)
    cpu_point = [medium_point[0] + x_step, vertical_start + y_step]    
    
    t = np.sqrt(y_step**2 + x_step**2) / 2
    alpha = np.arctan(y_step / np.abs(x_step))
    y_offset = np.sin(alpha) * t
    
    x_offset = (cpu_point[0] - medium_point[0]) / 2
    split_point = [medium_point[0] + x_offset, vertical_start + y_offset + (gpu[1] - vertical_start) / 2]
    ax.plot((split_point[0], gpu[0]), (split_point[1], gpu[1]), **style)


def draw_pci_transfer_cpu(ax, cpu, gpu, other_gpu, vertical_start, upper=True, style=None, zorder=None):
    medium_point = [(gpu[0] + other_gpu[0]) / 2, gpu[1]]
    x_step = X_STEP / 2 - min(gpu[0], other_gpu[0])
    y_step = Y_STEP * FORSHORTENING + CPU_VSTEP * (1 if upper else -1)
    cpu_point = [medium_point[0] + x_step, vertical_start + y_step]    
    
    t = np.sqrt(y_step**2 + x_step**2) / 2
    alpha = np.arctan(y_step / np.abs(x_step))
    y_offset = np.sin(alpha) * t
    
    x_offset = (cpu_point[0] - medium_point[0]) / 2
    split_point = [medium_point[0] + x_offset, vertical_start + y_offset + (gpu[1] - vertical_start) / 2]
    ax.plot((cpu_point[0], split_point[0]), (cpu_point[1], split_point[1]), zorder=zorder, **style)


# Draw the transfer between devices;
def draw_transfer(ax, transfer_matrix_nondirectional, max_transfer: float=None, min_transfer: float=None, 
                  redraw_points: bool=True, **kwargs):
   
    PALETTE = sns.color_palette("YlOrBr", as_cmap=True)
    MIN_PAL = 0.3
    MAX_PAL = 0.7
    MAX_WIDTH = 4
    MIN_WIDTH = 0.5
    if max_transfer is None:
        max_transfer = transfer_matrix_nondirectional.max().max()
    if min_transfer is None:
        min_transfer = transfer_matrix_nondirectional.min().min()
        
    def style_gpu(transfer):
        return dict(
            linewidth=transfer * (MAX_WIDTH - MIN_WIDTH) + MIN_WIDTH,
            linestyle="-",
            color=PALETTE(transfer * (MAX_PAL - MIN_PAL) + MIN_PAL),
            alpha=0.7,
            solid_capstyle="round",
        )
    
    # Shared PCI express channels;
    total_pci_01 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["0", "1"])]["CPU"].sum()
    total_pci_23 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["2", "3"])]["CPU"].sum()   
    total_pci_45 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["4", "5"])]["CPU"].sum()
    total_pci_67 = transfer_matrix_nondirectional.loc[transfer_matrix_nondirectional.index.isin(["6", "7"])]["CPU"].sum()
    draw_pci_transfer_cpu(ax, CPU[0], GPU[1], GPU[0], EDGE, style=style_gpu(total_pci_01), zorder=9)
    draw_pci_transfer_cpu(ax, CPU[0], GPU[3], GPU[2], EDGE, style=style_gpu(total_pci_23), zorder=9)
    draw_pci_transfer_cpu(ax, CPU[1], GPU[4], GPU[5], 0, False, style=style_gpu(total_pci_45))
    draw_pci_transfer_cpu(ax, CPU[1], GPU[7], GPU[6], 0, False, style=style_gpu(total_pci_67))
    
    # All the other channels;  
    for ii, i in enumerate(transfer_matrix_nondirectional.index):
        for jj, j in enumerate(transfer_matrix_nondirectional.columns):
            # Symmetric matrix, the lower triangular part is skipped;
            if ii > jj:
                continue
            transfer = transfer_matrix_nondirectional.loc[i, j]
            if transfer > 0:
                # Draw GPU-GPU transfer;
                if i != "CPU" and j != "CPU":
                    draw_line_gpu(ax, int(i), int(j), style_gpu(transfer))
                # Draw CPU-GPU transfer;
                else:
                    if j == "CPU":
                        gpu = int(i)
                        if gpu < 4:
                            draw_pci_transfer(ax, CPU[0], GPU[gpu], GPU[(gpu + 1) % 2 + (2 if gpu > 1 else 0)], EDGE, style=style_gpu(transfer))
                        elif gpu >= 4:
                            draw_pci_transfer(ax, CPU[1], GPU[gpu], GPU[(gpu + 1) % 2 + (6 if gpu > 5 else 4)], 0, False, style=style_gpu(transfer))
    return ax

##############################
##############################


if __name__ == "__main__":
    
    fig, ax = draw_topology(alpha_scale=0.5)
    save_plot(PLOT_DIR, f"v100_topology" + "_{}.{}", date=OUTPUT_DATE, dpi=600)    
    
    #%% Draw transfer of GPUs;
    for b in BENCHMARKS:
        fig, ax = draw_topology(alpha_scale=0.5)
        transfer_matrix = pd.read_csv(os.path.join(DEFAULT_RES_CUDA_DIR, INPUT_FOLDER, b + "_transfer_matrix.csv"), index_col=0)
        # Create non-directional matrix;
        transfer_matrix_nondirectional = create_nondirectional_transfer_matrix(transfer_matrix)
        # Normalize matrix;
        transfer_matrix_nondirectional /= transfer_matrix_nondirectional.max().max()
        # Draw colored edges;
        ax = draw_transfer(ax, transfer_matrix_nondirectional)
        # Add benchmark name;
        ax.annotate(b.upper(), xy=(0.78, 0.85), xycoords="axes fraction", ha="left", color="#2f2f2f", fontsize=14, alpha=1)   
        save_plot(PLOT_DIR, f"v100_topology_{b}" + "_{}.{}", date=OUTPUT_DATE, dpi=600)    