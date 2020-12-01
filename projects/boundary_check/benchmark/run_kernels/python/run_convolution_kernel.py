import polyglot
import os
import time
import argparse
import random

from trufflecuda_python_utils import *

##############################
##############################

OPT_LEVEL = "O0"
SIMPLIFICATION= "no_simplification"
DEDUCT_SIZES = "prevent"

UNMODIFIED_CUBIN_NAME = "convolution_checked.cubin"
UNMODIFIED_KERNEL_NAME = "convolution_checked"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
UNMODIFIED_KERNEL_PARAMS = "pointer, pointer, uint32, pointer"

MODIFIED_CUBIN_NAME = "convolution.cubin"
MODIFIED_KERNEL_NAME = "convolution"
MODIFIED_KERNEL_FOLDER = "../../cubin/"
MODIFIED_KERNEL_PARAMS = "pointer, pointer, uint32, pointer"

##############################
##############################

def main(args):
    num_elements = args.num_elements if args.num_elements else 1000
    num_blocks = (num_elements + NUM_THREADS - 1) // NUM_THREADS

    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    # Use separate data for the 2 kernels, to have the same memory transfers across tests;
    x = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    y = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    res = polyglot.eval(language="grcuda", string='float[{}]'.format(1))
    
    x2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    y2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    res2 = polyglot.eval(language="grcuda", string='float[{}]'.format(1))

    exec_time_unmodified = []
    exec_time_k_unmodified = []
    exec_time_modified = []
    exec_time_k_modified = []

    errors = []

    for i in range(num_test):

        for i in range(num_elements):
            x[i] = random.uniform(0, 1)
            y[i] = random.uniform(0, 1)
        res[0] = 0
        
        copy_device_array(x, x2)
        copy_device_array(y, y2)
        res2[0] = 0
        
        # Run the kernel without boundary checks;
        unmodified_kernel_path = os.path.join(UNMODIFIED_KERNEL_FOLDER, opt_level, UNMODIFIED_CUBIN_NAME)
        if os.path.isfile(unmodified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, UNMODIFIED_KERNEL_PARAMS, [x, y, len(x), res])
            
            exec_time_unmodified += [exec_time]
            exec_time_k_unmodified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of x: ', x[0:10])
                print('first 10 elements of y: ', y[0:10])
                print('res:', res[0])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
        res_unmodified = res[0]

        # Run the kernel with boundary checks;
        modified_kernel_path = os.path.join(MODIFIED_KERNEL_FOLDER, opt_level, simplify, MODIFIED_CUBIN_NAME)
        if os.path.isfile(modified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, MODIFIED_KERNEL_NAME, modified_kernel_path, MODIFIED_KERNEL_PARAMS,
             [x2, y2, len(x2), res2], deduct_sizes=DEDUCT_SIZES)
            
            exec_time_modified += [exec_time]
            exec_time_k_modified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of x: ', x2[0:10])
                print('first 10 elements of y: ', y2[0:10])
                print('res:', res2[0])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
        res_modified = res2[0]

        errors += [0 if check_equality(res_unmodified, res_modified, debug=debug) else 1]

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
