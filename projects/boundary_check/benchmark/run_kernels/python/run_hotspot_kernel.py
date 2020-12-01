import polyglot
import os
import time
import argparse
import random
import math

from trufflecuda_python_utils import *

##############################
##############################

OPT_LEVEL = "O0"
SIMPLIFICATION= "no_simplification"
DEDUCT_SIZES = "prevent"

UNMODIFIED_CUBIN_NAME = "calculate_temp_checked.cubin"
UNMODIFIED_KERNEL_NAME = "calculate_temp_checked"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
UNMODIFIED_KERNEL_PARAMS = "uint32, pointer, pointer, pointer, uint32, uint32, uint32, uint32, float, float, float, float, float, float"

MODIFIED_CUBIN_NAME = "calculate_temp.cubin"
MODIFIED_KERNEL_NAME = "calculate_temp"
MODIFIED_KERNEL_FOLDER = "../../cubin/"
MODIFIED_KERNEL_PARAMS = "uint32, pointer, pointer, pointer, uint32, uint32, uint32, uint32, float, float, float, float, float, float"

##############################
##############################

FACTOR_CHIP = 0.5
SPEC_HEAT_SI = 1.75e6
K_SI = 100
MAX_PD = 3e6
PRECISION = 0.001
T_CHIP = 0.0005
CHIP_HEIGHT = 0.016
CHIP_WIDTH = 0.016
EXPAND_RATE = 2
PYRAMID_HEIGHT = 1
BLOCK_SIZE = 16

##############################
##############################

def main(args):
    
    # Assume that the input is a square matrix;
    num_elements = args.num_elements if args.num_elements else 1600
    
    # If the input size is not a perfect square, keep just part of the elements;
    grid_rows = int(math.sqrt(num_elements))
    grid_cols = num_elements // grid_rows
    num_elements = grid_rows * grid_cols
    
    # Define a 2-dimensional threads and block subdivision;
    borderCols = (PYRAMID_HEIGHT)*EXPAND_RATE // 2
    borderRows = (PYRAMID_HEIGHT)*EXPAND_RATE // 2
    smallBlockCol = BLOCK_SIZE - (PYRAMID_HEIGHT)*EXPAND_RATE
    smallBlockRow = BLOCK_SIZE - (PYRAMID_HEIGHT)*EXPAND_RATE
    blockCols = int(grid_cols / smallBlockCol + (0 if (grid_cols % smallBlockCol == 0) else 1))
    blockRows = int(grid_rows / smallBlockRow + (0 if (grid_rows % smallBlockRow == 0) else 1))
    
    num_blocks = (blockCols, blockRows)
    num_threads = (BLOCK_SIZE, BLOCK_SIZE)
    
    row = grid_rows
    col = grid_cols
    
    grid_height = CHIP_HEIGHT / grid_rows
    grid_width = CHIP_WIDTH / grid_cols
    Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height
    Rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height)
    Ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width)
    Rz = T_CHIP / (K_SI * grid_height * grid_width)
    max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI)
    step = PRECISION / max_slope   

    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    matrix_power = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    matrix_temp_src = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    matrix_temp_dest = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    matrix_power2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    matrix_temp_src2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))
    matrix_temp_dest2 = polyglot.eval(language="grcuda", string='float[{}]'.format(num_elements))

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

    for i in range(num_test):

        for i in range(num_elements):
            matrix_power[i] = random.uniform(0, 1)
            matrix_temp_src[i] = random.uniform(0, 1)
            matrix_temp_dest[i] = 0
            matrix_power2[i] = matrix_power[i]
            matrix_temp_src2[i] = matrix_temp_src[i]
            matrix_temp_dest2[i] = 0
        
        # Run the kernel without boundary checks;        
        if os.path.isfile(unmodified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, UNMODIFIED_KERNEL_PARAMS,
                [0, matrix_power, matrix_temp_src, matrix_temp_dest, col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, 0], num_threads)        
            exec_time_unmodified += [exec_time]
            exec_time_k_unmodified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of matrix_power: ', matrix_power[0:10])
                print('first 10 elements of matrix_temp_src: ', matrix_temp_src[0:10])
                print('first 10 elements of matrix_temp_dest: ', matrix_temp_dest[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
        res_unmodified = copy_array(matrix_temp_dest)

        # Run the kernel with boundary checks;
        if os.path.isfile(modified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, MODIFIED_KERNEL_NAME, modified_kernel_path, MODIFIED_KERNEL_PARAMS,
                [0, matrix_power2, matrix_temp_src2, matrix_temp_dest2, col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, 0], num_threads, deduct_sizes=DEDUCT_SIZES)
            
            exec_time_modified += [exec_time]
            exec_time_k_modified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of matrix_power: ', matrix_power2[0:10])
                print('first 10 elements of matrix_temp_src: ', matrix_temp_src2[0:10])
                print('first 10 elements of matrix_temp_dest: ', matrix_temp_dest2[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
        res_modified = copy_array(matrix_temp_dest2)

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
