import os
import time
import argparse
import polyglot
import random

from trufflecuda_python_utils import *

##############################
##############################

OPT_LEVEL = "O0"
SIMPLIFICATION= "no_simplification"
DEDUCT_SIZES = "track"

UNMODIFIED_CUBIN_NAME = "bfs_checked.cubin"
UNMODIFIED_KERNEL_NAME = "bfs_checked"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
UNMODIFIED_KERNEL_PARAMS = "pointer, pointer, pointer, uint32, uint32, uint32, pointer, pointer, pointer"

MODIFIED_CUBIN_NAME = "bfs.cubin"
MODIFIED_KERNEL_NAME = "bfs"
MODIFIED_KERNEL_FOLDER = "../../cubin/"
MODIFIED_KERNEL_PARAMS = "pointer, pointer, pointer, uint32, uint32, uint32, pointer, pointer, pointer"

##############################
##############################

def main(args):

    random.seed(42)

    num_elements = args.num_elements if args.num_elements else 1000
    num_blocks = (num_elements + 1 + NUM_THREADS - 1) // NUM_THREADS

    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    ptr, idx = generate_graph(num_elements)

    N = num_elements
    E = len(idx)  

    # Use separate data for the 2 kernels, to have the same memory transfers across tests;
    ptr_d = polyglot.eval(language="grcuda", string='int[{}]'.format(N + 1))
    idx_d = polyglot.eval(language="grcuda", string='int[{}]'.format(E))
    res = polyglot.eval(language="grcuda", string='int[{}]'.format(N))
    graph_mask = polyglot.eval(language="grcuda", string='int[{}]'.format(N))
    graph_visited = polyglot.eval(language="grcuda", string='int[{}]'.format(N))
    updating_graph_mask = polyglot.eval(language="grcuda", string='int[{}]'.format(N))

    ptr_d2 = polyglot.eval(language="grcuda", string='int[{}]'.format(N + 1))
    idx_d2 = polyglot.eval(language="grcuda", string='int[{}]'.format(E))
    res2 = polyglot.eval(language="grcuda", string='int[{}]'.format(N))
    graph_mask2 = polyglot.eval(language="grcuda", string='int[{}]'.format(N))
    graph_visited2 = polyglot.eval(language="grcuda", string='int[{}]'.format(N))
    updating_graph_mask2 = polyglot.eval(language="grcuda", string='int[{}]'.format(N))

    exec_time_unmodified = []
    exec_time_k_unmodified = []
    exec_time_modified = []
    exec_time_k_modified = []

    errors = []

    for i in range(len(ptr)):
        ptr_d[i] = ptr[i]
        ptr_d2[i] = ptr[i]
    for i in range(len(idx)):
        idx_d[i] = idx[i]
        idx_d2[i] = idx[i]

    for i in range(num_test):

        for i in range(N):
            res[i] = 0
            res2[i] = 0
            graph_mask[i] = random.randint(0, 1)
            graph_mask2[i] = graph_mask[i]
            graph_visited[i] = graph_mask[i]
            graph_visited2[i] = graph_mask[i]
            updating_graph_mask[i] = 0
            updating_graph_mask2[i] = 0
        
        # Run the kernel without boundary checks;
        unmodified_kernel_path = os.path.join(UNMODIFIED_KERNEL_FOLDER, opt_level, UNMODIFIED_CUBIN_NAME)
        if os.path.isfile(unmodified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, UNMODIFIED_KERNEL_PARAMS,
             [ptr_d, idx_d, res, 1, N, E, graph_mask, graph_visited, updating_graph_mask])
            
            exec_time_unmodified += [exec_time]
            exec_time_k_unmodified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of ptr: ', ptr_d[0:10])
                print('first 10 elements of graph_mask: ', graph_mask[0:10])
                print('first 10 elements of res:', res[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
        res_unmodified = copy_array(res)

        # Run the kernel with boundary checks;
        modified_kernel_path = os.path.join(MODIFIED_KERNEL_FOLDER, opt_level, simplify, MODIFIED_CUBIN_NAME)
        if os.path.isfile(modified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, MODIFIED_KERNEL_NAME, modified_kernel_path, MODIFIED_KERNEL_PARAMS,
             [ptr_d2, idx_d2, res2, 1, N, E, graph_mask2, graph_visited2, updating_graph_mask2], deduct_sizes=DEDUCT_SIZES)
            
            exec_time_modified += [exec_time]
            exec_time_k_modified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of ptr: ', ptr_d2[0:10])
                print('first 10 elements of graph_mask: ', graph_mask2[0:10])
                print('first 10 elements of res:', res2[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
        res_modified = copy_array(res2)

        errors += [check_array_equality(res_unmodified, res_modified, debug=debug)]

    # Print execution times;
    if debug:
        print("\nSUMMARY - UNMODIFIED KERNEL:")
        print("    overall exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_unmodified), std(exec_time_unmodified)))
        print("    kernel exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_k_unmodified), std(exec_time_k_unmodified)))
        print("\nSUMMARY - MODIFIED KERNEL:")
        print("    overall exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_modified), std(exec_time_modified)))
        print("    kernel exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_k_modified), std(exec_time_k_modified)))
        print("errors: {:.2f}±{:.2f}".format(mean(errors), std(errors)))
    else:
        for i in range(len(errors)):
            print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                i,
                num_elements,
                opt_level,
                simplify,
                exec_time_k_unmodified[i],
                exec_time_unmodified[i],
                exec_time_k_modified[i],
                exec_time_modified[i],
                errors[i]))

##############################
##############################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test axpy kernel")
    
    parser.add_argument("-n", "--num_elements", metavar="N", type=int, nargs="?", 
                        help="Size of the arrays")
    parser.add_argument("-t", "--num_test", metavar="N", type=int, nargs="?", 
                        help="Number of times each test is performed")
    parser.add_argument("-o", "--opt_level", metavar="[O0|O1]", nargs="?", 
                        help="Optimization level of the input code")
    parser.add_argument("-s", "--simplify", action='store_true',
                        help="If present, use code with simpified array accesses")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="If present, print debug messages")
    
    args = parser.parse_args()

    main(args)
