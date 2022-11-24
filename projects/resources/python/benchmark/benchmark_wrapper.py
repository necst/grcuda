# Copyright (c) 2020, 2021, 2022, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import subprocess
import time
import os
from datetime import datetime
from benchmark_result import BenchmarkResult
from pathlib import Path

##############################
##############################

V100 = "V100"
A100 = "A100"
gtx960 = "GTX960"
GPU = gtx960

BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph.csv"

DEFAULT_NUM_BLOCKS = 32  # GTX 960, 8 SM
# DEFAULT_NUM_BLOCKS = 176  # GTX 1660 Super, 22 SM
# DEFAULT_NUM_BLOCKS = 448  # P100, 56 SM
if GPU == V100:
    DEFAULT_NUM_BLOCKS = 640  # V100, 80 SM
elif GPU == A100:
    DEFAULT_NUM_BLOCKS = 640

HEAP_SIZE = 26 
# HEAP_SIZE = 140 # P100
if GPU == V100:
    HEAP_SIZE = 470 # 2 x V100
elif GPU == A100:
    HEAP_SIZE = 470

# Benchmark settings;
benchmarks = [
    # Single GPU;
    "b1",
    "b5",
    "b6",
    "b7",
    "b8",
    "b10",
    # Multi GPU;
    # "b1m",
    # "b5m",
    # "b6m",
    # "b9m",
    # "b11m",
]

# GTX 960
num_elem = {
    "b1": [ 1353501,  51116203,  99592562, 107224034, 103446588,  79787132,
            39121728,  62085243,  35811029, 118574325,   7034598,  82292242,
            53552780,  87990537,  97334680,  13672474,  84208971,  64222994,
            92104396,  69495104,  32305654,  70336935,   2297978, 111088873,
            33393121,    851271,  51370939,  86914730,  25959085,  35192241,
            36990874,  84880236, 115913797,   5753345,  19617050,  63123900,
            22675468,  89853021,  45911910,  62465307,  23833209,   9496540,
            70221310,  72902774, 107455950,  59336597,  52166532,  32300678,
            10470469,   3772507,  59082620,   3006611,  90041118,  16184849,
            117199466,   5329819,  39058642,  76191920,  26203356, 115764294,
            74522292,  46263208,   7363045,   5732617, 101002510,  71087980,
            6515100,  18680756,  42190777, 116959113,  97326517,  25605978,
            70087602,  31722848,  75099384,  17929396, 103191460,  55682231,
            50943588, 112399462,   7674129,  43420977, 108644306, 101616196,
            68259037,  58715698,  17940056,  80118686,  64416895,  64820637,
            107213511,  21932775,  10120936,  85970355,  71895100,  93771429,
            72680532, 106132861, 116128851,  30616185],
    "b5": [ 1877067,  687298,  761509, 1827083, 1059021, 1930572, 1165158,
            1930978, 1456041,  703600, 1518298,  441729,  290361, 1124788,
            1112261,  992817,  251882, 1666557,  203818, 1353099,  787721,
            1872330,  202643, 1057054, 1725049,  208773, 1540845, 1014416,
            297651,  833593,  910570, 1981314, 1073114, 1195571, 1353675,
            1774238, 1708039, 1794481, 1555064, 1735997,  419656,  608529,
            446315,  218659, 1037244, 1170941,  683221, 1903379, 1008804,
            1375753,  759298,  290190,  310064,  494587,  684473,  999328,
            1530831,  411378,  139430, 1176571, 1874707,  824981, 1540940,
            1370870, 1659241, 1490664,  697967, 1571053, 1574549, 1424753,
            345585,  132252,  834383, 1242446, 1454237,  980564, 1586654,
            709014, 1404836, 1792277,  202943,  947622, 1879869, 1139411,
            1523548,  675601,  495633,  521253, 1494221, 1313785,   41891,
            1407676, 1575546,  624461, 1963356,  193533,  151055,  168133,
            1946558, 1037288],
    "b6": [ 245354, 1043434,  945569,  285921,  214970,  267558,  649723,
            426980,  318388,  366964,  324893,  567623,  839007, 1143490,
            1166253,  327027,  174801,  735724,  587446,  822246,  623052,
            831510,  597103,   17765, 1018702,  748404,  918229,   87850,
            932055,  527520,  904175, 1103024,  104310,  347990,  653447,
            59874,  601191,  403271,  701781,  956725, 1171487,  805240,
            840709,  652459,  509623, 1157326,  702564,  777099, 1051263,
            433795,  893700,  319286,  481626,  758919,   88308,  326871,
            763683,  754398,  864083,  544980, 1172895, 1189587, 1155656,
            976410,  672249, 1038937,   98241,   35696,  230085,  934440,
            738546,  950485,  775178, 1112906, 1160216,   61894, 1026477,
            252794,   10962,    4333, 1163549, 1188014, 1058880,  254167,
            424691,  769972,  557982, 1081066,  221779,  174524,  166086,
            317922, 1046580,  757946,  626648,   35804, 1136085, 1054884,
            193607, 1082471],
    "b7": [ 18741678,  1972299,  2614931,  2698617, 18404450, 14793130,
            18149264, 13670848,  2266514,  2812907, 14843910,  5481175,
            15919280,  5669713,  8090897,  3803532, 18172767,  5638891,
            1374879, 10745982, 11147624,  8467049,  9374533, 15600741,
            8127641, 12690257,  5759805,  1341171,  7577937,  2431218,
            2178302,  4248609,  9800534,  7413192, 13226228, 18590749,
            18818767, 12999602,    64803,  7009697, 13019287, 10665511,
            12203490,  5479663, 18572262, 17462123,  1779736, 17767210,
            1681173, 16318445,  3949048,  9897544, 15052453,  2459827,
            2987240, 16010768,   298563,  8721289,  4749634, 11286264,
            4607974,  5810839,  3591282, 12899288,  2021717,  4122720,
            10828127, 12923355,  5676136, 18100197, 12182200, 10836856,
            19371598,  2189685,  1198345, 11261515, 13922705, 17293335,
            9543180, 12239071,  5009260,  6610197,  6714703, 17328934,
            13603594,  7922831, 15401426,  9808473, 14263396,  5094413,
            12305162, 18811726,  4697422, 17164221,  1601953,  8528874,
            18201466,  5843464,  2468330,  9670264],
    "b8": [ 4073, 2287, 4533, 3466, 1192, 2929, 1464, 3584, 3961, 2785, 3052,
            4701, 1991, 3358, 3745, 2658, 1763, 4494, 2123, 3903, 2584, 2614,
            3135, 4389, 1665, 1392, 1057, 4389, 3954, 1916, 2828, 1405, 3564,
            1550, 3741, 1356, 2173, 4118, 1280, 3971, 1463, 4622, 3228, 2217,
            3368, 4688, 4515, 1587, 2100, 1144, 3167, 1447, 3083, 2878, 3332,
            3356, 1857, 1271, 3520, 3710, 1536, 2584, 2414, 1265, 1373, 2517,
            3058, 3266, 4783, 3197, 2234, 2962, 3395, 1666, 1471, 2822, 1082,
            4056, 4728, 2965, 2257, 2335, 1538, 3866, 4371, 1243, 4022, 1125,
            2293, 4170, 1605, 3279, 2478, 1629, 3129, 3687, 1022, 1735, 1695,
            3132],
    "b10": [1604, 2602, 4684, 3641, 5249, 1803, 4802, 4774, 2111, 5447, 6886,
            4484, 5015, 4300, 6103, 5455, 6545, 4120, 5143, 1554, 3998, 3088,
            5582, 6958, 3561, 5237, 4234, 1941, 1880, 4040, 4099, 3591, 1160,
            6702, 4119, 3138, 5526, 1808, 5018, 2366, 6272, 6624, 3384, 6075,
            3736, 4172, 6307, 1923, 6521, 3964, 6964, 2725, 2658, 5751, 3509,
            6182, 2198, 4163, 4652, 6968, 2044, 6120, 2776, 5206, 2661, 6281,
            6999, 4766, 4977, 5644, 3312, 3328, 2356, 1667, 5137, 4840, 3181,
            4597, 3446, 1649, 1542, 5329, 4028, 2115, 4662, 4228, 2479, 6910,
            1121, 3122, 3946, 1396, 5770, 2879, 1529, 4064, 5236, 4275, 4068,
            4825],
}

# GTX 1660 Super
# num_elem = {
#     "b1": [60_000_000, 80_000_000, 100_000_000, 120_000_000, 200_000_000],
#     "b5": [6_000_000, 8_000_000, 10_000_000, 12_000_000, 20_000_000],
#     "b6": [500_000, 800_000, 1_000_000, 1_200_000, 2_000_000],
#     "b7": [7_000_000, 10_000_000, 15_000_000, 20_000_000, 40_000_000],
#     "b8": [3200, 4000, 4800, 8000, 10000],
#     "b10": [6000, 7000, 10000, 12000, 14000],
# }

# P100
# num_elem = {
#     "b1": [120_000_000, 200_000_000, 500_000_000, 600_000_000, 700_000_000],
#     "b5": [12_000_000, 20_000_000, 50_000_000, 60_000_000, 70_000_000],
#     "b6": [1_200_000, 2_000_000, 4_000_000, 5_000_000, 6_000_000],
#     "b7": [20_000_000, 40_000_000, 60_000_000, 100_000_000, 140_000_000],
#     "b8": [4800, 8000, 10000, 12000, 16000],
#     "b10": [7000, 10000, 12000, 14000, 16000],
#}

# V100
if GPU == V100 or GPU == A100:
    num_elem = {
        # Single GPU;
        "b1": [160_000_000, 250_000_000, 500_000_000, 800_000_000, 950_000_000],
        "b5": [10_000_000, 16_000_000, 21_000_000, 28_000_000, 35_000_000], # out of core 50_000_000, 80_000_000, 95_000_000],
        "b6": [1_600_000, 2_500_000, 5_000_000, 6_500_000, 8_000_000],
        "b7": [25_000_000, 50_000_000, 80_000_000, 130_000_000, 180_000_000], 
        "b8": [6400, 10000, 13000, 16000, 20000],
        "b10": [12000, 16000, 18000, 20000, 22000], 
        # Multi GPU;
        "b1m": [160_000_000, 250_000_000, 500_000_000, 800_000_000, 950_000_000],
        "b5m": [10_000_000, 16_000_000, 21_000_000, 28_000_000, 35_000_000],  # out of core 50_000_000, 80_000_000, 95_000_000]
        "b6m": [1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000],
        "b9m": [20000, 30000, 40000, 50000, 60000],
        "b11m": [20000, 30000, 40000, 50000, 60000],
    }
# num_elem = {k: [int(v[0] / 100)] for (k, v) in num_elem.items()}  # Use this for small sizes, for debugging;

# 960
block_dim_dict = {
    # Single GPU;
    "b1": DEFAULT_NUM_BLOCKS,
    "b5": DEFAULT_NUM_BLOCKS,
    "b6": 32,
    "b7": DEFAULT_NUM_BLOCKS,
    "b8": 12,
    "b10": 16,
    "b11": DEFAULT_NUM_BLOCKS,
    # Multi GPU;
    "b1m": 64,
    "b5m": 64,
    "b6m": 64,
    "b9m": 64,
    "b11m": 64,
}

# P100
# block_dim_dict = {
#     "b1": DEFAULT_NUM_BLOCKS,
#     "b5": DEFAULT_NUM_BLOCKS,
#     "b6": 64,
#     "b7": DEFAULT_NUM_BLOCKS,
#     "b8": 32,
#     "b10": DEFAULT_NUM_BLOCKS,
#     "b11": DEFAULT_NUM_BLOCKS,
# }

# V100
if GPU == V100 or GPU == A100:
    block_dim_dict = {
        # Single GPU;
        "b1": DEFAULT_NUM_BLOCKS,
        "b5": DEFAULT_NUM_BLOCKS,
        "b6": 64,
        "b7": DEFAULT_NUM_BLOCKS,
        "b8": 32,
        "b10": DEFAULT_NUM_BLOCKS,
        "b11": DEFAULT_NUM_BLOCKS,
        # Multi GPU;
        "b1m": 64,
        "b5m": 64,
        "b6m": 64,
        "b9m": 64,
        "b11m": 64,
    }

# 1660
# block_dim_dict = {
#     "b1": DEFAULT_NUM_BLOCKS,
#     "b5": DEFAULT_NUM_BLOCKS,
#     "b6": 32,
#     "b7": DEFAULT_NUM_BLOCKS,
#     "b8": 16,
#     "b10": DEFAULT_NUM_BLOCKS,
#     "b11": DEFAULT_NUM_BLOCKS,
# }

cuda_exec_policies = ["async"]  # ["sync", "async", "cudagraph", "cudagraphmanual", "cudagraphsingle"]

exec_policies = ["async"]

dependency_policies = ["with-const"]  #, "no-const"]

new_stream_policies = ["always-new"]  #, "reuse"]

parent_stream_policies = ["disjoint"]  # ["same-as-parent", "disjoint", "multigpu-early-disjoint", "multigpu-disjoint"]

choose_device_policies = ["minmax-transfer-time"]  # ["single-gpu", "round-robin", "stream-aware", "min-transfer-size", "minmin-transfer-time", "minmax-transfer-time"]

memory_advise = ["none"]

prefetch = ["false"]

stream_attach =  [False]

time_computation = [False]

train_computation = [True]

num_gpus = [1]

block_sizes1d_dict = {
    "b1": 32,
    "b5": 1024,
    "b6": 32,
    "b7": 32,
    "b8": 32,
    "b10": 32,
    "b11": 256, 
    # Multi GPU;
    "b1m": 32,
    "b5m": 1024,
    "b6m": 32,
    "b9m": 32,
    "b11m": 256,
}

block_sizes2d_dict = {
    "b1": 8,
    "b5": 8,
    "b6": 8,
    "b7": 8,
    "b8": 16,
    "b10": 8,
    "b11": 16,
    # Multi GPU;
    "b1m": 8,
    "b5m": 8,
    "b6m": 8,
    "b9m": 8,
    "b11m": 8,
}

##############################
##############################

CUDA_CMD = "./b -k {} -p {} -n {} -b {} -c {} -t {} -m {} -g {} {} {} | tee {}"

def execute_cuda_benchmark(benchmark, size, block_size, exec_policy, num_iter, debug, prefetch=False, stream_attach=False, num_blocks=DEFAULT_NUM_BLOCKS, num_gpus=1, output_date=None, mock=False):
    if debug:
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={b}, size={n},"
                                    f" block size={block_size}, "
                                    f" prefetch={prefetch}, "
                                    f" num blocks={num_blocks}, "
                                    f" num GPUs={num_gpus}, "
                                    f" exec policy={exec_policy}")
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")

    do_prefetch = prefetch is not None and prefetch and prefetch != "none" and prefetch != "false"

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"cuda_{output_date}_{benchmark}_{exec_policy}_{size}_gpu{num_gpus}_{block_size['block_size_1d']}_{block_size['block_size_2d']}_{prefetch}_{num_iter}_{num_blocks}.csv"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_cuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder_path, file_name)

    benchmark_cmd = CUDA_CMD.format(benchmark, exec_policy, size, block_size["block_size_1d"],
                                    block_size["block_size_2d"], num_iter, num_gpus, num_blocks, "-r" if do_prefetch else "", "-a" if stream_attach else "", output_path)
    if not mock:
        start = time.time()
        result = subprocess.run(benchmark_cmd,
                                shell=True,
                                stdout=None,
                                cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/cuda/bin")
        result.check_returncode()
        end = time.time()
        if debug:
            BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start):.2f} seconds")
    else:
        # Just print the command, for debugging;
        if debug:
            BenchmarkResult.log_message(benchmark_cmd)


##############################
##############################

GRAALPYTHON_CMD = "graalpython --vm.XX:MaxHeapSize={}G --jvm --polyglot --experimental-options " \
                  "--grcuda.ExecutionPolicy={} --grcuda.DependencyPolicy={} --grcuda.RetrieveNewStreamPolicy={} " \
                  "--grcuda.NumberOfGPUs={} --grcuda.RetrieveParentStreamPolicy={} " \
                  "--grcuda.DeviceSelectionPolicy={} --grcuda.MemAdvisePolicy={} --grcuda.InputPrefetch={} --grcuda.BandwidthMatrix={} {} {} {} " \
                  "benchmark_main.py -i {} -n {} -g {} --number_of_gpus {} --reinit false --realloc false " \
                  "-b {} --block_size_1d {} --block_size_2d {} --execution_policy {} --dependency_policy {} --new_stream {} "\
                  "--parent_stream {} --device_selection {} --memory_advise_policy {} --prefetch {} --no_cpu_validation {} {} {} {} -o {}"

def execute_grcuda_benchmark(benchmark, size, num_gpus, block_sizes, exec_policy, dependency_policy, new_stream_policy,
                      parent_stream_policy, choose_device_policy, memory_advise, prefetch, num_iter, bandwidth_matrix, time_phases, debug, stream_attach=False,
                      time_computation=False, train_computation=False, num_blocks=DEFAULT_NUM_BLOCKS, output_date=None, mock=False):
    if debug:
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message(f"Benchmark {i + 1}/{tot_benchmarks}")
        BenchmarkResult.log_message(f"benchmark={benchmark}, size={n},"
                                    f"gpus={num_gpus}, "
                                    f"block-sizes={block_sizes}, "
                                    f"num-blocks={num_blocks}, "
                                    f"exec-policy={exec_policy}, "
                                    f"dependency-policy={dependency_policy}, "
                                    f"new-stream-policy={new_stream_policy}, "
                                    f"parent-stream-policy={parent_stream_policy}, "
                                    f"choose-device-policy={choose_device_policy}, "
                                    f"mem-advise={memory_advise}, "
                                    f"prefetch={prefetch}, "
                                    f"stream-attachment={stream_attach}, "
                                    f"time-computation={time_computation}, "
                                    f"train-computation={train_computation}, "
                                    f"bandwidth-matrix={bandwidth_matrix}, "
                                    f"time-phases={time_phases}")
        BenchmarkResult.log_message("")

    if not output_date:
        output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{output_date}_{benchmark}_{size}_{num_gpus}_{num_blocks}_{exec_policy}_{dependency_policy}_" \
                f"{new_stream_policy}_{parent_stream_policy}_{choose_device_policy}_" \
                f"{memory_advise}_{prefetch}_{stream_attach}.json"
    # Create a folder if it doesn't exist;
    output_folder_path = os.path.join(BenchmarkResult.DEFAULT_RES_FOLDER, output_date + "_grcuda")
    if not os.path.exists(output_folder_path):
        if debug:
            BenchmarkResult.log_message(f"creating result folder: {output_folder_path}")
        if not mock:
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_folder_path, file_name)
    b1d_size = " ".join([str(b['block_size_1d']) for b in block_sizes])
    b2d_size = " ".join([str(b['block_size_2d']) for b in block_sizes])

    benchmark_cmd = GRAALPYTHON_CMD.format(HEAP_SIZE, exec_policy, dependency_policy, new_stream_policy,
                                           num_gpus, parent_stream_policy, choose_device_policy, memory_advise, prefetch, bandwidth_matrix,
                                           "--grcuda.ForceStreamAttach" if stream_attach else "", 
                                           "--grcuda.EnableComputationTimers" if time_computation else "",
                                           "--grcuda.EnableTrainingComputation" if train_computation else "",
                                           num_iter, size, num_blocks, num_gpus, benchmark, b1d_size, b2d_size, exec_policy, dependency_policy,
                                           new_stream_policy, parent_stream_policy, choose_device_policy, memory_advise, prefetch,
                                           "-d" if debug else "",
                                           "-p" if time_phases else "", 
                                           "--force_stream_attach" if stream_attach else "", 
                                           "--timing" if time_computation else "",
                                           output_path)    
    if debug:
        BenchmarkResult.log_message(benchmark_cmd)
        BenchmarkResult.log_message("#" * 30)
        BenchmarkResult.log_message("")
        BenchmarkResult.log_message("")
    if not mock:
        start = time.time()
        result = subprocess.run(benchmark_cmd,
                                shell=True,
                                stdout=None, #subprocess.STDOUT,
                                cwd=f"{os.getenv('GRCUDA_HOME')}/projects/resources/python/benchmark")
        result.check_returncode()
        end = time.time()
        if debug:
            BenchmarkResult.log_message(f"Benchmark total execution time: {(end - start):.2f} seconds")

##############################
##############################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wrap the GrCUDA benchmark to specify additional settings")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-c", "--cuda_test", action="store_true",
                        help="If present, run performance tests using CUDA")
    parser.add_argument("-i", "--num_iter", metavar="N", type=int, default=BenchmarkResult.DEFAULT_NUM_ITER,
                        help="Number of times each benchmark is executed")
    parser.add_argument("-g", "--num_blocks", metavar="N", type=int,
                        help="Number of blocks in each kernel, when applicable")
    parser.add_argument("-p", "--time_phases", action="store_true",
                        help="Measure the execution time of each phase of the benchmark;"
                             " note that this introduces overheads, and might influence the total execution time")
    parser.add_argument("-m", "--mock", action="store_true",
                        help="If present, simply print the benchmark CMD without executing it")
    parser.add_argument("--gpus", metavar="N", type=int, nargs="*",
                        help="Specify the maximum number of GPUs to use in the computation")

    # Parse the input arguments;
    args = parser.parse_args()

    debug = args.debug if args.debug else BenchmarkResult.DEFAULT_DEBUG
    num_iter = args.num_iter if args.num_iter else BenchmarkResult.DEFAULT_NUM_ITER
    use_cuda = args.cuda_test
    time_phases = args.time_phases
    num_blocks = args.num_blocks
    mock = args.mock
    gpus = args.gpus

    if gpus is not None:
        num_gpus = gpus

    if debug:
        BenchmarkResult.log_message(f"using block sizes: {block_sizes1d_dict} {block_sizes2d_dict}; using low-level CUDA benchmarks: {use_cuda}")

    def tot_benchmark_count():
        tot = 0
        if use_cuda:
            for b in benchmarks:
                for e in cuda_exec_policies:
                    if e == "sync":
                        tot += len(num_elem[b]) * len(prefetch) * len(stream_attach) 
                    else:
                        tot += len(num_elem[b]) * len(prefetch) * len(num_gpus) * len(stream_attach) 
        else:
            for b in benchmarks:
                for e in exec_policies:
                    if e == "sync":
                        tot += len(num_elem[b]) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation) * len(train_computation)
                    else:
                        for n in num_gpus:
                            if n == 1:
                                tot += len(num_elem[b]) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation) * len(train_computation)
                            else:
                                tot += len(num_elem[b]) * len(dependency_policies) * len(new_stream_policies) * len(parent_stream_policies) * len(choose_device_policies) * len(memory_advise) * len(prefetch) * len(stream_attach) * len(time_computation) * len(train_computation)
        return tot

    output_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Execute each test;
    i = 0
    tot_benchmarks = tot_benchmark_count()
    for b in benchmarks:
        for n in num_elem[b]:
            if use_cuda:
                # CUDA Benchmarks;
                for e in cuda_exec_policies:
                    if e == "sync":
                        ng = [1]
                    else:
                        ng = num_gpus
                    block_sizes = BenchmarkResult.create_block_size_list([block_sizes1d_dict[b]], [block_sizes2d_dict[b]])
                    for block_size in block_sizes:
                        for p in prefetch:
                            for a in stream_attach:
                                for num_gpu in ng:
                                    nb = num_blocks if num_blocks else block_dim_dict[b]
                                    execute_cuda_benchmark(b, n, block_size, e, num_iter, debug, num_gpus=num_gpu, num_blocks=nb, prefetch=p, stream_attach=a, mock=mock, output_date=output_date)
                                    i += 1
            # GrCUDA Benchmarks;
            else:
                for exec_policy in exec_policies:
                    if exec_policy == "sync":
                        dp = [dependency_policies[0]]
                        nsp = [new_stream_policies[0]]
                        psp = [parent_stream_policies[0]]
                        cdp = [choose_device_policies[0]]
                        ng = [1]
                    else:
                        dp = dependency_policies
                        nsp = new_stream_policies
                        psp = parent_stream_policies
                        cdp = choose_device_policies
                        ng = num_gpus
                    for num_gpu in ng:
                        if exec_policy == "async" and num_gpu == 1:
                            dp = [dependency_policies[0]]
                            nsp = [new_stream_policies[0]]
                            psp = [parent_stream_policies[0]]
                            cdp = [choose_device_policies[0]]
                        else:
                            dp = dependency_policies
                            nsp = new_stream_policies
                            psp = parent_stream_policies
                            cdp = choose_device_policies
                        for m in memory_advise:
                            for p in prefetch:
                                for s in stream_attach:
                                    for tr in train_computation:
                                        for t in time_computation:
                                            # Select the correct connection graph;
                                            # if GPU == V100:
                                            #     BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph_{num_gpu}_v100.csv"
                                            # elif GPU == A100:
                                            #      BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph_8_a100.csv"
                                            BANDWIDTH_MATRIX = f"{os.getenv('GRCUDA_HOME')}/projects/resources/connection_graph/datasets/connection_graph.csv"

                                            for dependency_policy in dp:
                                                for new_stream_policy in nsp:
                                                    for parent_stream_policy in psp:
                                                        for choose_device_policy in cdp:
                                                            nb = num_blocks if num_blocks else block_dim_dict[b]
                                                            block_sizes = BenchmarkResult.create_block_size_list([block_sizes1d_dict[b]], [block_sizes2d_dict[b]])
                                                            execute_grcuda_benchmark(b, n, num_gpu, block_sizes,
                                                                exec_policy, dependency_policy, new_stream_policy, parent_stream_policy, choose_device_policy,
                                                                m, p, num_iter, BANDWIDTH_MATRIX, time_phases, debug, s, t, tr, nb, output_date=output_date, mock=mock)
                                                            i += 1
