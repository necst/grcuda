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

UNMODIFIED_CUBIN_NAME = "backprop_checked.cubin"
UNMODIFIED_KERNEL_NAME = "backprop_checked"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
UNMODIFIED_KERNEL_PARAMS = "pointer, pointer, pointer, pointer, uint32, uint32"

MODIFIED_CUBIN_NAME = "backprop.cubin"
MODIFIED_KERNEL_NAME = "backprop"
MODIFIED_KERNEL_FOLDER = "../../cubin/"
MODIFIED_KERNEL_PARAMS = "pointer, pointer, pointer, pointer, uint32, uint32"

##############################
##############################

def main(args):
    num_elements = args.num_elements if args.num_elements else 1000
    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    in_size = num_elements
    hid = 16
    num_blocks = in_size // 16
    grid = (1, num_blocks)
    threads = (16, 16)

    WIDTH = 16

    # Use separate data for the 2 kernels, to have the same memory transfers across tests;
    input_units = polyglot.eval(language="grcuda", string='float[{}]'.format(in_size + 1))
    input_weights_one_dim = polyglot.eval(language="grcuda", string='float[{}]'.format((in_size + 1) * (hid + 1)))
    partial_sum = polyglot.eval(language="grcuda", string='float[{}]'.format(num_blocks * WIDTH))
    output_hidden = polyglot.eval(language="grcuda", string='float[{}]'.format(hid + 1))

    input_units2 = polyglot.eval(language="grcuda", string='float[{}]'.format(in_size + 1))
    input_weights_one_dim2 = polyglot.eval(language="grcuda", string='float[{}]'.format((in_size + 1) * (hid + 1)))
    partial_sum2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_blocks * WIDTH))
    output_hidden2 = polyglot.eval(language="grcuda", string='float[{}]'.format(hid + 1))

    exec_time_unmodified = []
    exec_time_k_unmodified = []
    exec_time_modified = []
    exec_time_k_modified = []

    errors = []

    unmodified_kernel_path = os.path.join(UNMODIFIED_KERNEL_FOLDER, opt_level, UNMODIFIED_CUBIN_NAME)
    modified_kernel_path = os.path.join(MODIFIED_KERNEL_FOLDER, opt_level, simplify, MODIFIED_CUBIN_NAME)

    if not os.path.isfile(unmodified_kernel_path) or not os.path.isfile(modified_kernel_path):
        if debug:
            print("kernel {} or {} not found!".format(unmodified_kernel_path, modified_kernel_path))
        return

    for _ in range(num_test):

        for i in range(len(input_units)):
            input_units[i] = random.uniform(0, 1)
            input_units2[i] = input_units[i]
        for i in range(len(input_weights_one_dim)):
            input_weights_one_dim[i] = random.uniform(0, 1)
            input_weights_one_dim2[i] = input_weights_one_dim[i]
        for i in range(len(partial_sum)):
            partial_sum[i] = 0
            partial_sum2[i] = 0
        for i in range(len(output_hidden)):
            output_hidden[i] = 0
            output_hidden2[i] = 0

        # Normalize array;
        input_units_mean = mean(input_units)
        input_weights_one_dim_mean = mean(input_weights_one_dim)
        for i in range(len(input_units)):
            input_units[i] -= input_units_mean
            input_units2[i] -= input_units_mean
        for i in range(len(input_weights_one_dim)):
            input_weights_one_dim[i] -= input_weights_one_dim_mean
            input_weights_one_dim2[i] -= input_weights_one_dim_mean
        
        # Run the kernel without boundary checks;
        if os.path.isfile(unmodified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, grid, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, UNMODIFIED_KERNEL_PARAMS,
             [input_units, output_hidden, input_weights_one_dim, partial_sum, in_size, hid], threads)
            
            exec_time_unmodified += [exec_time]
            exec_time_k_unmodified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of x: ', input_units[0:10])
                print('first 10 elements of res:', partial_sum[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
        res_unmodified = copy_array(partial_sum)

        # Run the kernel with boundary checks;
        if os.path.isfile(modified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, grid, MODIFIED_KERNEL_NAME, modified_kernel_path, MODIFIED_KERNEL_PARAMS,
             [input_units2, output_hidden2, input_weights_one_dim2, partial_sum2, in_size, hid], threads, deduct_sizes=True)
            
            exec_time_modified += [exec_time]
            exec_time_k_modified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of x: ', input_units2[0:10])
                print('first 10 elements of res:', partial_sum2[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
        res_modified = copy_array(partial_sum2)

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
