# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:54:09 2021

@author: albyr
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from segretini_matplottini.src.plot_utils import *
from load_data import PLOT_DIR

##############################
##############################

OUTPUT_DATE = "2021_10_06"

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

##############################
##############################

def setup():
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
def draw_points(ax):
    # Obtain list of coordinates;
    x = [x[0] for x in POINTS]
    y = [y[1] for y in POINTS]
    ax.scatter(x, y, color="#2f2f2f")
    return ax

# Draw names of GPUs;
def draw_gpu_names(ax):
    x_offset = {
        2: -EDGE / 5,
        3: -EDGE / 10,
        7: -EDGE / 5,
        }
    y_offset = {
        0: -EDGE / 12,
        1: -EDGE / 12,
        2: EDGE / 25,
        3: EDGE / 25,
        6: -EDGE / 12,
        7: -EDGE / 8,
        }
    for i, (g, p) in enumerate(GPU.items()):
        x = p[0] + EDGE * 0.02 + (x_offset[g] if g in x_offset else 0)
        y = p[1] + EDGE * 0.01 + (y_offset[g] if g in y_offset else EDGE / 70)
        ax.annotate(f"GPU{g}", xy=(x, y), color="#2f2f2f", fontsize=10, ha="left")
    return ax

# Join corners;
def draw_edges(ax):
    def draw_line(x, y, style):
        x = GPU[x]
        y = GPU[y]
        ax.plot((x[0], y[0]), (x[1], y[1]), **style)
     
    # Double NVLink;
    style_nv2 = dict(
        linewidth=2,
        linestyle="-",
        color="#2f2f2f",
        alpha=0.9,
    )
    # Single NVLink;
    style_nv1 = dict(
        linewidth=0.8,
        linestyle="--",
        color="#2f2f2f",
        alpha=0.7,
    )
    # Missing edge is PCIe;  
    
    # Connect GPUs;
    draw_line(0, 1, style_nv1)
    draw_line(0, 2, style_nv1)
    draw_line(0, 3, style_nv2)
    draw_line(0, 6, style_nv2)
    draw_line(1, 2, style_nv2)
    draw_line(1, 3, style_nv1)
    draw_line(1, 7, style_nv2)
    draw_line(2, 3, style_nv2)
    draw_line(2, 4, style_nv1)
    draw_line(3, 5, style_nv1)
    draw_line(4, 5, style_nv2)
    draw_line(4, 6, style_nv1)
    draw_line(4, 7, style_nv2)
    draw_line(5, 6, style_nv2)
    draw_line(5, 7, style_nv1)
    draw_line(6, 7, style_nv1)
    
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


def draw_cpu_points(ax): 
   
    # Obtain list of coordinates;
    x = [x[0] for x in CPU_POINTS]
    y = [y[1] for y in CPU_POINTS]
    ax.scatter(x, y, color="#888888")
    return ax


def draw_cpu_lines(ax):
    
    style = dict(
        color="#888888",
        alpha=0.8,
        linestyle="-",
        linewidth=1,
    )
    
    def draw_pci(gpu0, gpu1, vertical_start, upper=True):
    
        # Upper left PCIe;
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
                
    draw_pci(GPU[1], GPU[0], EDGE)
    draw_pci(GPU[2], GPU[3], EDGE)
    draw_pci(GPU[7], GPU[6], 0, False)
    draw_pci(GPU[4], GPU[5], 0, False)
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

##############################
##############################


if __name__ == "__main__":
    
    fig, ax = setup()
    ax = draw_cpu_lines(ax)
    ax = draw_edges(ax)
    # ax = draw_faces(ax)
    ax = draw_points(ax)
    ax = draw_cpu_points(ax)
    ax = draw_gpu_names(ax)
    ax = draw_cpu_names(ax)
    save_plot(PLOT_DIR, f"v100_topology" + "_{}.{}", date=OUTPUT_DATE, dpi=600)    
