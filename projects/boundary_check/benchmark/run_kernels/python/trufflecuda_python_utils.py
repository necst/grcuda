#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import polyglot
import random

NUM_THREADS = 128

def timeit(function):
    def timed(*args, **kw):
        ts = time.time()
        result = function(*args, **kw)
        te = time.time()
        print("{}, overall exec. time: {:.2f} ms".format(function.__name__, (te - ts) * 1000))
        return result
    return timed


def check_array_equality(x_gold, x, error_threshold=10e-6, debug=True):
    num_errors = 0
    if debug:
        print("checking errors...")
    for pos, i in enumerate(range(len(x_gold))):
        diff = abs(x_gold[i] - x[i])
        if diff > error_threshold:
            num_errors += 1
            if debug and num_errors < 20:
                print("\t{}) error - correct {}, found {} - diff. {}".format(pos, x_gold[i], x[i], diff))
    if debug:
        print("number of errors: {}".format(num_errors))
    
    return num_errors

def check_equality(x_gold, x, error_threshold=10e-6, debug=True):
    if debug:
        print("checking errors...")
    
    diff = abs(x_gold - x)
    if diff > error_threshold:
        if debug:
            print("\terror - correct {}, found {} - diff. {}".format(x_gold, x, diff))
    
        return False
    return True


def copy_array(x):
    return [i for i in x]

def copy_device_array(x, x_new):
    for i in range(min(len(x), len(x_new))):
        x_new[i] = x[i]


def mean(x):
    return sum(x) / len(x)


def std(x):
    m = 0
    m_sq = 0
    for i in x:
        m += i
        m_sq += i * i
    return math.sqrt((m_sq - m * m / len(x)) / len(x))

def generate_graph(N, max_degree=10, avoid_self_edges=True):
    ptr = [0]
    idx = []
    for v in range(N):
        num_edges = random.randint(0, min(max_degree, N))
        edge_set = set()
        for _ in range(num_edges):
            edge_set.add(random.randint(0, N - 1))
        if v in edge_set:
            edge_set.remove(v)
        idx += list(edge_set)
        ptr += [len(edge_set) + ptr[v - 1]]
    
    return ptr, idx


##############################
##############################

def run_kernel(debug: bool, num_blocks: int, kernel_name: str, kernel_path: str,
               kernel_params: str, params: list, num_threads=NUM_THREADS, deduct_sizes=False):

    start = time.perf_counter()
    
    code = """bindkernel("{}", "{}", "{}", "{}")""".format(kernel_path, kernel_name, kernel_params, deduct_sizes)
    kernel = polyglot.eval(language='grcuda', string=code)

    if debug:
        print("got '{}' kernel: {}".format(kernel_name, kernel))
        print("invoking kernel as {}<<<{}, {}>>>(...)".format(kernel_name, num_blocks, num_threads))

    # Call the kernel;
    start_k = time.perf_counter()
    kernel(num_blocks, num_threads)(*params)
    end = time.perf_counter()

    exec_time = (end - start) * 1000000
    exec_time_k = (end - start_k) * 1000000
    
    if debug:
        print("{} kernel exec.time: {:.2f} μs".format(kernel_name, exec_time_k))
        print("{} overall exec.time: {:.2f} μs".format(kernel_name, exec_time))

    return (exec_time, exec_time_k)
