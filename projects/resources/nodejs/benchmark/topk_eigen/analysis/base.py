import numpy as np
from matplotlib.patches import Patch, Rectangle, Shadow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.stats.mstats import gmean
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.lines as lines
import math
from scipy.interpolate import interp1d


import os
from plot_utils_v2 import COLORS, get_exp_label, get_ci_size, save_plot, update_width, add_labels,\
    remove_outliers_df_iqr_grouped, get_upper_ci_size,remove_outliers_df_grouped

import matplotlib.ticker as ticker

sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
plt.rcParams["font.family"] = ["Latin Modern Roman Demi 10 Regular"]
plt.rcParams['font.family'] = 'Arial'

plt.rcParams['hatch.linewidth'] = 0.3
plt.rcParams['axes.labelpad'] = 5 
PALETTE = [COLORS["peach2"], COLORS["g1"]]
PALETTE_B = ["#CEF0D2", "#C8FCB6", "#96DE9B", "#66B784", "#469E7B"]
HATCHES = ["/" * 3, "\\" * 3, "/" * 3, "\\" * 3, "/" * 3]

styles=["o" , "v" , "^"]
MY_PALETTE = [c.lower() for c in ["#C2801B","#C73625","#AD5CB3","#4CD6A0","#1E99DF"]]
COLOR_PALETTE = ['#7bd490', '#C6E6DB', '#B3366C']

GRAPH_NAMES = {
    "web-NotreDame": "WB-ND",
    "wiki-Talk": "WB-TA",
    "web-Google": "WB-GO",
    "web-BerkStan": "WB-BE",
    "flickr": "FL",
    "italy_osm": "IT",
    "patents": "PA",
    "venturiLevel3" :"VL3",
    "germany_osm": "DE",
    "asia_osm": "ASIA",
    "road_central": "RC",
    "wikipedia-20070206": "WK",
    "hugetrace-00020": "HT",
    "wb-edu": "WB",
    "hugebubbles-00010": "HB",
    "soc-LiveJournal1": "LJ",
    "GAP-kron": "KRON",
    "GAP-urand": "URAND", 
    "MOLIERE-2016": "MOLIERE"
}
big_graphs = ["MOLIERE", "KRON", "URAND"]

df_grcuda = pd.read_csv("./results_grcuda_no_reorth.csv")
df_grcuda["dataset_name"] = df_grcuda["dataset_name"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_grcuda["dataset_name"] = df_grcuda["dataset_name"].replace(GRAPH_NAMES)
df_grcuda["execution_time(ms)"] = df_grcuda["execution_time"].apply(lambda elem: elem / 1000)
df_grcuda["kind"] = "GrCUDA"
df_grcuda = df_grcuda[(df_grcuda["dataset_name"] != "LJ")]
to_use = [d for d in df_grcuda["dataset_name"].unique() if "wikipedia" not in d]
df_grcuda = df_grcuda[df_grcuda.dataset_name.isin(to_use)]
df_grcuda.head()


aggregated_exec_times = []
aggregated_speedups = []

outliers = ["FL", "WB-BE"]

FONTSIZE = 8
plt.rcdefaults()
# plt.style.use('classic')
fig = plt.figure(figsize=(2.8, 3.3))
graph_selection = []
for (d_name, gpus), dataset in df_grcuda.groupby(["dataset_name", "gpus"]): 
    aggregated_exec_times.append([d_name, gpus, dataset.execution_time.mean()])

exclude_list = ["WK", "WB-TA"]
    
df_exec_times = pd.DataFrame(aggregated_exec_times, columns=["name", "gpus", "exec_time"])
df_exec_times = df_exec_times[~df_exec_times.name.isin(big_graphs + exclude_list)]
for d_name in df_exec_times.name.unique(): 
    cur_df = df_exec_times[df_exec_times.name == d_name]
    baseline = cur_df[cur_df.gpus == 1].exec_time.iloc[0]
    aggregated_speedups.append([d_name, "1", 1.0])
    
    for gpu_count in [2, 4, 8]: 
        aggregated_speedups.append([d_name, f"{gpu_count}", cur_df[cur_df.gpus == gpu_count].exec_time.iloc[0] / baseline])

        
mean_aggregated_speedups = []
for e in aggregated_speedups.copy():
    if e[0] not in outliers: 
        mean_aggregated_speedups.append(["Mean relative reduction in execution time", e[1], e[2]])
    else:
        mean_aggregated_speedups.append(e)

df_aggregated_speedups = pd.DataFrame(mean_aggregated_speedups, columns=["name", "gpus", "perc_reduction_exec_time"])
 
ax = sns.lineplot(data=df_aggregated_speedups[df_aggregated_speedups.name == "Mean relative reduction in execution time"],
                  x="gpus", y="perc_reduction_exec_time", hue="name", legend=False, palette=[COLOR_PALETTE[2]],err_style="bars",err_kws={"capsize": 10.0}, linewidth=2, color="black")
sns.lineplot(data=df_aggregated_speedups[df_aggregated_speedups.name != "Mean relative reduction in execution time"],
             x="gpus", y="perc_reduction_exec_time", hue="name", legend=False, palette=COLOR_PALETTE[:2],err_style="bars",err_kws={"capsize": 10.0}, ax=ax, linewidth=1, ls="--")

#ax.legend(title="Graph")
ax.set_ylabel("Relative reduction in execution times", fontsize=FONTSIZE, fontweight="bold"), 
ax.set_xlabel("Number of GPUs", fontsize=FONTSIZE, fontweight="bold")
#ax.set_xticklabels([1, 2, "",4 , "",  "","",  8])
plt.xticks(["1", "2", "4", "8"], weight="normal")
plt.yticks(weight="normal")
# plt.legend(title="")
#plt.legend(["Mean relative reduction in execution time", *outliers])

ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))

#ax = sns.scatterplot(x="gpus", y="perc_reduction_exec_time", data=df_aggregated_speedups.groupby(["name", "gpus"]).mean().reset_index(), style="name", palette=[COLOR_PALETTE[2], *COLOR_PALETTE[:2]], legend=False, s=100, zorder=1000)
cmap = dict(zip(df_aggregated_speedups.name.unique(), COLOR_PALETTE))
CP = [COLOR_PALETTE[0],COLOR_PALETTE[2], COLOR_PALETTE[1],]
for i, (name, df) in enumerate(df_aggregated_speedups.groupby(["name", "gpus"]).mean().reset_index().groupby("name")):  
    sns.scatterplot(x="gpus", y="perc_reduction_exec_time", data=df, style="name",
                    legend=False, zorder=4, s=100, edgecolor="#2f2f2f", color=CP[i], markers=styles[i], ax=ax)

# ax.legend(bbox_to_anchor=(1.1, 1.11), ncol=3, prop={"weight":"bold", "size": FONTSIZE})

legend_labels = ["FL", "Mean relative exec. time", "WB-BE"]
custom_lines = [
    lines.Line2D([], [], color="white", marker=styles[i], markersize=10, label=legend_labels[i],
                 markerfacecolor=CP[i], markeredgecolor="#2f2f2f") for i in range(len(legend_labels))
    ]

legend_labels = [legend_labels[1]] + legend_labels[::2]
custom_lines = [custom_lines[1]] + custom_lines[::2]
leg = fig.legend(custom_lines, legend_labels, shadow=True, borderpad=0.5, fancybox=False, edgecolor="#2f2f2f",
                         bbox_to_anchor=(0.915, 1), fontsize=FONTSIZE, ncol=len(legend_labels), columnspacing=0.3, handletextpad=0.01)

leg.set_title(None)
leg.get_frame().set_linewidth(0.8)
leg._legend_box.align = "left"

ax.tick_params(axis='y', which='major', labelsize=FONTSIZE)
ax.tick_params(axis='x', which='major', labelsize=FONTSIZE)
ax.grid(True, ls=":", color="#2f2f2f")

plt.savefig("gpu_scaling.pdf",  bbox_inches='tight')

#%%

df_acc = pd.read_csv("results_grcuda_orth.csv")
df_acc["dataset_name"] = df_acc["dataset_name"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_acc["type"] = "ORTH"
df_acc = df_acc[df_acc["dataset_name"] != "1138_bus"]
df_acc.head()

df_acc_nr = pd.read_csv("./results_grcuda_acc_no_reorth.csv")
df_acc_nr["dataset_name"] = df_acc_nr["dataset_name"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_acc_nr["type"] = "NO_REORTH"
df_acc_nr.head()

df_acc = df_acc.append(df_acc_nr)
accuracies = []
for ec, df in df_acc.groupby(["eigen_count"]): 
    accuracies.append([f"{ec}", df.lanczos_orth.to_numpy(), df.overall_reconstr.to_numpy(), df.type.to_numpy()])

unraveled = []
for a, l, re, k in accuracies: 
    for i in range(len(l)): 
        unraveled.append([f"{a}", l[i], re[i], k[i]])
df_u = pd.DataFrame(unraveled, columns=["eigen", "lanczos_orth", "rec_err", "kind"])
df_u["lanczos_orth"] = 90 -  df_u["lanczos_orth"]

#COLOR_PALETTE = ['#7bd490', '#C6E6DB', '#FA4D4A']
fig, axs = plt.subplots(2, figsize=(2.8, 3.3))

#HERE
sns.lineplot(data=df_u, x="eigen", y="lanczos_orth", hue="kind", palette=[COLOR_PALETTE[0], COLOR_PALETTE[-1]],  estimator="mean",
                      err_style="bars",err_kws={"capsize": 5.0, "elinewidth":1}, linewidth=1, legend=None, sort=False, ci=99, ax=axs[0])
axs[0].set_ylim((78,90))

axs[0].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°", ))


axs[0].grid(True, ls=":", color="#2f2f2f")

kind_form = {
    "ORTH": "Reorthogonalization every two iterations", 
    "NO_REORTH": "No reorthogonalization"
}

df_u["kind_formatted"] = df_u.kind.replace(kind_form)
sns.lineplot(data=df_u, x="eigen", y="rec_err", hue="kind_formatted",legend=None, palette=[COLOR_PALETTE[0], COLOR_PALETTE[-1]],  estimator="mean",
                      err_style="bars",err_kws={"capsize": 5.0, "elinewidth":1}, linewidth=1, sort=False, ci=99, ax=axs[1])
axs[1].set_yscale("log")
#axs[0].set_xlabel( "", fontsize=57)
axs[0].set_ylabel("Orthogonality", fontsize=FONTSIZE,  fontweight="bold")
axs[1].set_ylabel("Reconstruction Error", fontsize=FONTSIZE, weight="bold")
axs[1].set_xlabel("Eigencomponents", fontsize=FONTSIZE, weight="bold")
axs[1].grid(True, ls=":", color="#2f2f2f")
axs[1].yaxis.set_tick_params(labelsize=FONTSIZE, width=1)
axs[1].xaxis.set_tick_params(labelsize=FONTSIZE, width=1)
axs[0].yaxis.set_tick_params(labelsize=FONTSIZE, width=1)
axs[0].xaxis.set_tick_params(labelsize=FONTSIZE, width=1)
c_ticks = []
c_ticks_labels = []


for ii in range(7, 4, -1): 
    c_ticks += [1 * 10**-ii, 3 * 10**-ii]
    c_ticks_labels += [
        "$10^{" + str(-ii) +"}$",
        "3•$10^{" + str(-ii) +"}$",
        #"5•$10^{" + str(-ii) +"}$"
    ]
#axs[1].set_yscale("log")


axs[1].set_yticks(c_ticks[2:])
axs[1].set_yticklabels(c_ticks_labels[2:])
ii = 2
for tick in axs[1].yaxis.get_major_ticks():
    if c_ticks_labels[ii][0] == "2" or c_ticks_labels[ii][0] == "5": 
        tick.label.set_fontsize(FONTSIZE)
        tick.label.set_color("#2d2d2d")
    ii += 1

axs[0].yaxis.set_label_coords(-0.233, 0.6)
axs[1].yaxis.set_label_coords(-0.233, 0.47)

legend_labels = ["Reorth. every 2 iterations", "No reorth."]
custom_lines = [
    lines.Line2D([], [], color="white", marker=styles[0], markersize=10, label=legend_labels[i],
                 markerfacecolor=CP[i], markeredgecolor="#2f2f2f") for i in range(len(legend_labels))
    ]

leg = fig.legend(custom_lines, legend_labels, shadow=True, borderpad=0.5, fancybox=False, edgecolor="#2f2f2f", 
                         bbox_to_anchor=(0.92, 0.99), fontsize=FONTSIZE, ncol=len(legend_labels), columnspacing=0.1, handletextpad=0.01)


plt.savefig("orthogonality_and_reconstruction_error.pdf",  bbox_inches='tight')

#%%

df_fpga = pd.read_csv("./results_fpga.csv")
df_fpga["graph_path"] = df_fpga["graph_path"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_fpga["graph_path"] = df_fpga["graph_path"].replace(GRAPH_NAMES)

df_cpu = pd.read_csv("./results_cpu_xeon_platinum.csv")
df_cpu["graph"] = df_cpu["graph"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_cpu["graph"] = df_cpu["graph"].replace(GRAPH_NAMES)



df_fpga["execution_time(ms)"] = df_fpga["lanczos_time"] / 1000
df_cpu["execution_time(ms)"] = df_cpu["execution_time"] * 1000
df_cpu.head()

execution_times = []
for graph in df_grcuda.dataset_name.unique(): 
    df_graph_fpga = df_fpga[df_fpga.graph_path == graph]
    df_graph_gpu = df_grcuda[df_grcuda.dataset_name == graph]
    df_graph_cpu = df_cpu[df_cpu.graph == graph]


    for i, r in df_graph_fpga.iterrows(): 
        execution_times.append([graph, r.edges,r["execution_time(ms)"], "U280", r["num_eigenvalues"]])
    for i, r in df_graph_gpu.iterrows(): 
        execution_times.append([graph, r.e, r["execution_time(ms)"], "V100", r["eig_count"]])
    for i, r in df_graph_cpu.iterrows(): 
        execution_times.append([graph, r.edges, r["execution_time(ms)"], "XeonPlatinum", r["k"]])

df_execution_times = pd.DataFrame(execution_times, columns=["graph_name", "nnz", "execution_time", "arch", "eig_count"])
#df_execution_times = df_execution_times[(df_execution_times.graph_name != "WB-TA") & (df_execution_times.graph_name != "WK")]

df_execution_times[df_execution_times.graph_name == "URAND"]


df_execution_times["speedup"] = 0

exec_times2 = []

for graph in df_execution_times.graph_name.unique(): 
    for eig_count in range(8, 28, 4):
          
        if graph in big_graphs and eig_count != 8: 
            continue
        cpu_selector = (df_execution_times.graph_name == graph) & (df_execution_times.arch == "XeonPlatinum") & (df_execution_times.eig_count == eig_count)
        gpu_selector = (df_execution_times.graph_name == graph) & (df_execution_times.arch == "V100") & (df_execution_times.eig_count == eig_count)
        fpga_selector = (df_execution_times.graph_name == graph) & (df_execution_times.arch == "U280") & (df_execution_times.eig_count == eig_count)
      
        baseline_cpu_ms = df_execution_times[cpu_selector]["execution_time"].mean()
        
        
        if(baseline_cpu_ms == np.nan): 
            continue
        
        exec_times2.append([graph, df_execution_times[gpu_selector]["execution_time"].mean() / 1000, df_execution_times[fpga_selector]["execution_time"].mean() / 1000, df_execution_times[cpu_selector]["execution_time"].mean() / 1000])
        
        df_execution_times.loc[gpu_selector, "speedup"] = baseline_cpu_ms / df_execution_times[gpu_selector]["execution_time"] 
        df_execution_times.loc[fpga_selector, "speedup"] = baseline_cpu_ms / df_execution_times[fpga_selector]["execution_time"] 
        df_execution_times.loc[cpu_selector, "speedup"] = baseline_cpu_ms / df_execution_times[cpu_selector]["execution_time"] 

df_execution_times[(df_execution_times.arch == "XeonPlatinum") & (df_execution_times.graph_name == "HT")]

exec_times_filtered = [a for a in exec_times2 if not np.isnan(a[-1])]
proper_times = pd.DataFrame(exec_times_filtered, columns=["graph", "gpu_time", "fpga_time", "cpu_time"]).groupby("graph").mean().reset_index()

geomean = []


mean_cpu = df_execution_times[(df_execution_times.arch=="XeonPlatinum") ]["execution_time"].mean()
mean_gpu = df_execution_times[(df_execution_times.arch=="V100") ]["execution_time"].mean()
mean_fpga = df_execution_times[(df_execution_times.arch=="U280") ]["execution_time"].mean()

print(mean_cpu, mean_gpu,mean_fpga)

geomean = [
    [
        "GMEAN", 
        0, 
        0,
        "XeonPlatinum", 
        0,
        1.0
    ],
    [
        "GMEAN", 
        0, 
        0,
        "V100", 
        0,
        mean_cpu / mean_gpu
    ], 
    [
        "GMEAN", 
        0, 
        0,
        "U280", 
        0,
        mean_cpu / mean_fpga
    ],
    [
        "GMEAN", 
        0, 
        0,
        "XeonPlatinum", 
        0,
        1.0
    ],
    [
        "GMEAN", 
        0, 
        0,
        "V100", 
        0,
        mean_cpu / mean_gpu
    ], 
    [
        "GMEAN", 
        0, 
        0,
        "U280", 
        0,
        mean_cpu / mean_fpga
    ]
]

df_execution_times = pd.DataFrame(geomean, columns=["graph_name", "nnz", "execution_time", "arch", "eig_count", "speedup"]).append(df_execution_times)
df_execution_times = remove_outliers_df_iqr_grouped(df_execution_times, "speedup", ["arch", "graph_name", "eig_count"], debug=False)

proper_times = proper_times.append(pd.DataFrame([[
        "GMEAN", 
        mean_gpu / 1000, 
        mean_fpga / 1000, 
        mean_cpu / 1000
    ]], columns=["graph", "gpu_time", "fpga_time", "cpu_time"]))

df_execution_times.sort_values("nnz", inplace=True)
df_execution_times["execution_time(s)"] = df_execution_times["execution_time"] / 1000
graphs = df_execution_times["graph_name"].unique()

df_execution_times.loc[(df_execution_times.graph_name == "GMEAN") & (df_execution_times.arch == "XeonPlatinum"), "execution_time(s)"] = mean_cpu / 1000
df_execution_times.loc[(df_execution_times.graph_name == "GMEAN") & (df_execution_times.arch == "V100"), "execution_time(s)"] =  mean_gpu / 1000
df_execution_times.loc[(df_execution_times.graph_name == "GMEAN") & (df_execution_times.arch == "U280"), "execution_time(s)"] =  mean_fpga / 1000

exec_times_agg_s = []



for (graph, arch), df in df_execution_times.groupby(["graph_name", "arch"]): 
    if graph != "GMEAN": 
        cpu_exec = df_execution_times[(df_execution_times.graph_name == graph) & (df_execution_times.arch == "XeonPlatinum")]["execution_time"].mean()
        
        mean_speedup_time = df["speedup"].mean()
        mean_exec_time = df["execution_time"].mean()
        #TODO: here
        exec_times_agg_s.append([graph, arch, mean_exec_time])

gmean_graph = df_execution_times[df_execution_times.graph_name == "GMEAN"]
cpu = gmean_graph[gmean_graph.arch == "XeonPlatinum"]
gpu = gmean_graph[gmean_graph.arch == "V100"]
fpga = gmean_graph[gmean_graph.arch == "U280"]


    
exec_times_agg_s = [
    ["GMEAN", "XeonPlatinum", cpu["execution_time(s)"].mean()],
    ["GMEAN", "U280", fpga["execution_time(s)"].mean()],
    ["GMEAN", "V100", gpu["execution_time(s)"].mean()],
    *exec_times_agg_s
]

#df_exec_times_agg_s = pd.DataFrame(exec_times_agg_s, columns=["graph", "arch", "time"])
graphs = df_execution_times["graph_name"].unique()
graphs
df_execution_times[(df_execution_times.graph_name == "IT") & (df_execution_times.arch == "U280")]

TEXT_PALETTE = ["#4d4d4d", "#262626", "#000000"]

fig = plt.figure(figsize=(20,6))
gs = gridspec.GridSpec(1, 1)
plt.subplots_adjust(top=0.97,
                    bottom=0.39,
                    left=0.08,
                    right=0.98,
                    hspace=1.0,
                    wspace=0.8)  
ax = fig.add_subplot(gs[0, 0])

arch_rename_map = dict(zip(["XeonPlatinum", "U280", "V100"], ["Xeon Platinum 8167M", "Alveo U280", "NVIDIA Tesla V100"]))

df_execution_times["arch_ext"] = df_execution_times["arch"].replace(arch_rename_map)

ax = sns.barplot(data=df_execution_times, x="graph_name", y="speedup", ci=99, hue="arch_ext", palette=PALETTE_B, ax=ax,capsize=.05, errwidth=0.8, edgecolor="#2f2f2f",hue_order=["Xeon Platinum 8167M", "Alveo U280", "NVIDIA Tesla V100"])
#ax = sns.barplot(data=df_execution_times, x="graph_name", y="execution_time", ci=99, hue="arch",palette=PALETTE_B, ax=ax,capsize=.05, errwidth=0.8, edgecolor="#2f2f2f",)


plt.legend(title="Platform")
ax.set_xlabel("", fontsize=15)
_ = ax.set_ylabel("Speedup (log-scale)", fontsize=17)
num_benchmarks = len(df_grcuda.dataset_name.unique())
for j, bar in enumerate([p for p in ax.patches if not pd.isna(p)]):
    bar.set_hatch(HATCHES[j // num_benchmarks])



ax.set_yscale('log')
plt.yticks(list(plt.yticks()[0]) + [1.0])
ax.axhline(1.0, ls="--", color="red")
plt.axvline(x=0.5, ls="--", color="teal")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.rcParams['legend.fontsize'] = 17
ax.set_ylim((0.7, 100000)) 


ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000])
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}x"))

ax.tick_params(axis='y', which='major', labelsize=14)
ax.grid(True, axis="y")

offsets = []
for k, g in df_execution_times.groupby(["graph_name", "arch_ext"], sort=False):
    co = get_upper_ci_size(g["speedup"], ci=0.99)
    if np.isnan(co): 
        co = g["speedup"].mean()
    offsets += [co]
    #print(offsets[-1])

#offsets = [max(o, 0.3) for o in offsets]


#add_labels(ax, vertical_offsets=offsets,rotation=90, format_str="{:.2f}x", fontsize=13, skip_zero=False, skip_value=1, skip_threshold=0.04)
labels = [p.get_height() for p in ax.patches]
for i, p in enumerate(ax.patches): 
    if 0.98 < p.get_height() < 1.04: 
        continue
        
    scale = p.get_height() / 1.7
    ax.text(p.get_x() + p.get_width()/2., scale + p.get_height(), "{:.2f}x".format(labels[i]), fontsize=14, color="black", ha='center', va='bottom', rotation=90, weight="bold")

plt.margins(x=0.001, y=0.001) 


FORBIDDEN_SET = []
group_length = 3

#print mean execution times for each graph below
for i in range(len(ax.patches) // group_length):
    if i not in FORBIDDEN_SET:
        y_min = 0.6
        y_max = 0.45
        x_middle = (ax.patches[i].get_x() + ax.patches[i].get_x() + group_length * ax.patches[i].get_width()) / 2
        fpga_time = proper_times[(proper_times.graph == graphs[i])]['fpga_time'].mean()
        if np.isnan(fpga_time): 
            fpga_time = "N/A"
        else: 
            fpga_time = f"{(fpga_time):.2f}"
        #ax.plot([ax.patches[i].get_x(), ax.patches[i].get_x() + group_length * ax.patches[i].get_width()], [y_min, y_min], clip_on=False, color="#2f2f2f", linewidth=1)
        #ax.plot([x_middle, x_middle], [y_min, y_max], clip_on=False, color="#2f2f2f", linewidth=1)
        # Also add median execution time;
        ax.text(x_middle, 0.15, f"{(proper_times[(proper_times.graph == graphs[i])]['cpu_time'].mean()):.2f}", fontsize=15, color=TEXT_PALETTE[0], ha="center")
        ax.text(x_middle, 0.07, fpga_time, fontsize=15, color=TEXT_PALETTE[1], ha="center")
        ax.text(x_middle, 0.035, f"{(proper_times[(proper_times.graph == graphs[i])]['gpu_time'].mean()):.2f}", fontsize=15, color=TEXT_PALETTE[2], ha="center")


        j += 1
    ax.text(-1.2, 0.09, "Median\nexec. time [s]", fontsize=15, color="#666666", ha="right", va="center")
    ax.text(-1, 0.15, "CPU", fontsize=15, color=TEXT_PALETTE[0], ha="left")
    ax.text(-1, 0.07, "FPGA", fontsize=15, color=TEXT_PALETTE[1], ha="left")
    ax.text(-1, 0.035, "GPU", fontsize=15, color=TEXT_PALETTE[2], ha="left")


ax.legend(bbox_to_anchor=(0.50, -0.45), ncol=3)



plt.savefig("speedup.pdf", bbox_inches='tight')

#%% bubble plots of accuracy vs execution time
graph_properties = {}

for graph, df in df_grcuda.groupby("dataset_name"): 
    graph_properties[graph] = [
        *df.iloc[1, 1:4]
    ]

def scale(eigen_count, graph_name): 
    gp = graph_properties[graph_name]
    return gp[1] + 2 * gp[0] + (eigen_count - 1) * (4 * gp[0] + gp[1]) + 2 * eigen_count**2 * gp[1]

dataset_to_use = [ 'WB-BE', 'WB-GO','PA', 'HT', 'VL3', 'IT']
df_mp_ddd = pd.read_csv("./results_grcuda_all_double_double_double.csv")
df_mp_ddd["dataset_name"] = df_mp_ddd["dataset_name"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_mp_ddd["dataset_name"] = df_mp_ddd["dataset_name"].replace(GRAPH_NAMES)
df_mp_ddd["normalized_execution_time"] = df_mp_ddd.apply(lambda row: row["execution_time"] / scale(row["eigen_count"], row["dataset_name"]), axis=1)
df_mp_ddd = df_mp_ddd[df_mp_ddd.dataset_name.isin(dataset_to_use)]
df_mp_ddd["overall_reconstr"] /= df_mp_ddd["eigen_count"]
#df_mp_ddd["dataset_name"] = df_mp_ddd["dataset_name"].apply(lambda x: x + " double double double")
df_mp_ddd["datatype"] = "double double double"
#scale(row["eigen_count"], row["dataset_name"])

df_mp_fdf = pd.read_csv("./results_grcuda_all_float_float_double.csv")
df_mp_fdf["dataset_name"] = df_mp_fdf["dataset_name"].apply(lambda elem: elem.split("/")[-1].replace(".mtx", ""))
df_mp_fdf["dataset_name"] = df_mp_fdf["dataset_name"].replace(GRAPH_NAMES)
df_mp_fdf["normalized_execution_time"] = df_mp_fdf.apply(lambda row: row["execution_time"] / scale(row["eigen_count"], row["dataset_name"]), axis=1)
df_mp_fdf = df_mp_fdf[df_mp_fdf.dataset_name.isin(dataset_to_use)]
df_mp_fdf["overall_reconstr"] /= df_mp_fdf["eigen_count"]
#df_mp_fdf["dataset_name"] = df_mp_fdf["dataset_name"].apply(lambda x: x + " float double float")
df_mp_fdf["datatype"] = "float double float"


df_mp = df_mp_fdf.append(df_mp_ddd)
#df_mp = remove_outliers_df_iqr_grouped(df_mp, "overall_reconstr", ["kind"])
#df_mp = remove_outliers_df_iqr_grouped(df_mp, "normalized_execution_time", ["kind"])

df_grcuda["normalized_execution_time"] = df_grcuda.apply(lambda row: row["execution_time"] * 1000 / scale(row["eig_count"], row["dataset_name"]), axis=1)
df_grcuda["datatype"] = "float float float"

df_acc_nr["dataset_name"] = df_acc_nr["dataset_name"].replace(GRAPH_NAMES)
df_acc["dataset_name"] = df_acc["dataset_name"].replace(GRAPH_NAMES)

#plt.figure(figsize=(20, 12))
plt.figure(figsize=(4, 3.0))
agg = []
for(name, dtype), df in df_mp.groupby(["dataset_name", "datatype"]): 
    #print(name, df["normalized_execution_time"].mean(), df["overall_reconstr"].mean())
    agg.append([name, df["overall_reconstr"].mean(), df["normalized_execution_time"].mean(), dtype])
    
for name, df in df_mp.groupby(["dataset_name"]): 
    gc = df_grcuda[df_grcuda.dataset_name == name]
    nr = df_acc_nr[df_acc_nr.dataset_name == name]
    exec_time = gc["normalized_execution_time"].mean()
    reconstr_accuracy = nr["overall_reconstr"].mean()
    agg.append([name, reconstr_accuracy, exec_time, "float float float"])
    #print(agg[-1])


df_mp_agg = pd.DataFrame(agg, columns=["dataset_name", "overall_reconstr", "normalized_execution_time", "datatype"])
#df_mp_agg["overall_reconstr"] = np.log10(df_mp_agg["overall_reconstr"])

rename_map = {
    "float float float": "FFF", 
    "float double float": "FDF", 
    "double double double": "DDD"
}

legend_items = []
for graph, df in df_mp_agg.groupby("dataset_name"): 
    sns.lineplot(x=df["overall_reconstr"], y=df["normalized_execution_time"], linewidth=0.8)


df_mp_agg["Graph"] = df_mp_agg["dataset_name"]
df_mp_agg["Datatype"] = df_mp_agg["datatype"].replace(rename_map)

ax = sns.scatterplot(data=df_mp_agg, x="overall_reconstr", y="normalized_execution_time", hue="Graph", style="Datatype", s=70)
sns.regplot(data=df_mp_agg[["overall_reconstr", "normalized_execution_time"]], 
            x="overall_reconstr",
            y="normalized_execution_time", 
            ax=ax, 
            truncate=True, 
            n_boot=1000,
            scatter=False, 
            ci=95,
            logx=True,
            #order=2,
            #robust=True, 
            #logistic=True, NO
            #lowess=True,
            color=COLOR_PALETTE[2],
            line_kws={"linewidth": 0.8, "linestyle": "--", "zorder": -1},
            #x_jitter=10**-7,
            #y_jitter=10
           )



fontdict={
    "fontsize": "x-large"
}
xytext=(-10, -10)
for name, df in df_mp_agg.groupby("dataset_name"): 
    for i, el in df.iterrows(): 
        ax.annotate(rename_map[el["datatype"]], xy=(el['overall_reconstr'], el['normalized_execution_time']), textcoords="offset points", ha="center",xytext=xytext, fontsize=FONTSIZE-2)

ax.grid(True, ls=":")


ticks = []
labels = []
for i in range(4, 9): 
    tmp = []
    for j in [5, 2, 1]: 
        if i == 4 and j != 1: 
            continue
        if j == 1: 
            tmp.append(j * 10**-i)
            labels.append("$" + "10^{-" + str(i) + "}" + "$")
        
    ticks += [*tmp]
plt.xticks(ticks=ticks[::-1], labels=labels[::-1], weight="normal")
plt.yticks(weight="normal")

ax.set_xlim((2 * 10**-8, 10**-4))
ax.set_ylim((0, 7))

ax.tick_params(axis='x', which='both', labelsize=FONTSIZE)
ax.tick_params(axis='y', which='both', labelsize=FONTSIZE)

ax.set_xlabel("Normalized Reconstruction Error", fontsize=FONTSIZE, weight="bold")
ax.set_ylabel("Normalized Execution Time", fontsize=FONTSIZE, weight="bold")
ax.set_xscale("log")
plt.legend(prop={"weight":"normal", "size": FONTSIZE}, shadow=True, borderpad=0.5, fancybox=False, edgecolor="#2f2f2f")

plt.savefig("pareto.pdf",  bbox_inches='tight')


