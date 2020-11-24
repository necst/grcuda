import os
import time
import argparse
import polyglot

from trufflecuda_python_utils import *

##############################
##############################

OPT_LEVEL = "O0"
SIMPLIFICATION= "no_simplification"

UNMODIFIED_CUBIN_NAME = "gaussian.cubin"
UNMODIFIED_KERNEL_NAME = "gaussian"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
KERNEL_PARAMS = "pointer, pointer, pointer, uint32, uint32, uint32"

MODIFIED_CUBIN_NAME = "gaussian.cubin"
MODIFIED_KERNEL_NAME = "gaussian"
MODIFIED_KERNEL_FOLDER = "../../cubin/"

##############################
##############################

def main(args):

    NUM_THREADS = 16

    matrix_side = args.num_elements if args.num_elements else 1000
    num_elements = matrix_side**2

    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    threads_per_block = (NUM_THREADS, NUM_THREADS)
    num_blocks_rows = (matrix_side + NUM_THREADS - 1) // NUM_THREADS
    num_blocks_cols = (matrix_side + NUM_THREADS - 1) // NUM_THREADS
    num_blocks = (num_blocks_rows, num_blocks_cols)

    # Use separate data for the 2 kernels, to have the same memory transfers across tests;
    x = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    y = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    z = polyglot.eval(language="grcuda", string='float[{}]'.format(matrix_side))
    
    x2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    y2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    z2 = polyglot.eval(language="grcuda", string='float[{}]'.format(matrix_side))

    exec_time_unmodified = []
    exec_time_k_unmodified = []
    exec_time_modified = []
    exec_time_k_modified = []

    errors = []

    for i in range(num_test):

        for i in range(num_elements):
            x[i] = random.uniform(0, 1)
            x2[i] = x[i]
        for i in range(num_elements):   
            y[i] = 0
            y2[i] = 0
        for i in range(matrix_side):   
            z[i] = random.uniform(0, 1)
            z2[i] = z[i]
        
        # Run the kernel without boundary checks;
        unmodified_kernel_path = os.path.join(UNMODIFIED_KERNEL_FOLDER, opt_level, UNMODIFIED_CUBIN_NAME)
        if os.path.isfile(unmodified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, KERNEL_PARAMS,
             [x, y, z, matrix_side, matrix_side, 0], threads_per_block)
            
            exec_time_unmodified += [exec_time]
            exec_time_k_unmodified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of x: ', x[0:10])
                print('first 10 elements of y: ', y[0:10])
                print('first 10 elements of res:', z[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
        res_unmodified = copy_array(z)

        # Run the kernel with boundary checks;
        modified_kernel_path = os.path.join(MODIFIED_KERNEL_FOLDER, opt_level, simplify, MODIFIED_CUBIN_NAME)
        if os.path.isfile(modified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, MODIFIED_KERNEL_NAME, modified_kernel_path, KERNEL_PARAMS,
             [x2, y2, z2, matrix_side, matrix_side, 0], threads_per_block, deduct_sizes=True)
            
            exec_time_modified += [exec_time]
            exec_time_k_modified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of x: ', x2[0:10])
                print('first 10 elements of y: ', y2[0:10])
                print('first 10 elements of res:', z2[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
        res_modified = copy_array(z2)

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
