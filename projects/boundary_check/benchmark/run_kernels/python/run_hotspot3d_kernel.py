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

UNMODIFIED_CUBIN_NAME = "hotspot3d_checked.cubin"
UNMODIFIED_KERNEL_NAME = "hotspot3d_checked"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
KERNEL_PARAMS = "pointer, pointer, pointer, float, uint32, uint32, uint32, float, float, float, float, float, float, float"

MODIFIED_CUBIN_NAME = "hotspot3d.cubin"
MODIFIED_KERNEL_NAME = "hotspot3d"
MODIFIED_KERNEL_FOLDER = "../../cubin/"

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
    
    nx = num_elements
    ny = num_elements
    nz = num_elements

    dx = CHIP_HEIGHT / nx
    dy = CHIP_WIDTH / ny
    dz = T_CHIP / nz
    
    Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * dx * dy
    Rx = dy / (2.0 * K_SI * T_CHIP * dx)
    Ry = dx / (2.0 * K_SI * T_CHIP * dy)
    Rz = dz / (K_SI * dx * dy)
    max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI)
    dt = PRECISION / max_slope  

    stepDivCap = dt / Cap
    ce = stepDivCap / Rx
    cw = stepDivCap / Rx
    cn = stepDivCap / Ry
    cs = stepDivCap / Ry
    ct = stepDivCap / Rz
    cb = stepDivCap / Rz
    cc = 1 - (2 * ce + 2 * cn + 3 * ct)
    s = nx * ny * nz

    num_blocks = ((nx + 64 - 1) // 64, (ny + 4 - 1) // 4, 1)
    num_threads = (64, 4, 1)

    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    p = polyglot.eval(language="grcuda", string='float[{}]'.format(s))
    t_in = polyglot.eval(language="grcuda", string='float[{}]'.format(s))
    t_out = polyglot.eval(language="grcuda", string='float[{}]'.format(s))
    p2 = polyglot.eval(language="grcuda", string='float[{}]'.format(s))
    t_in2 = polyglot.eval(language="grcuda", string='float[{}]'.format(s))
    t_out2 = polyglot.eval(language="grcuda", string='float[{}]'.format(s))

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

        for i in range(s):
            t_in[i] = random.uniform(0, 1)
            t_out[i] = 0
            p[i] = random.uniform(0, 1)
            t_in2[i] = t_in[i]
            t_out2[i] = t_out[i]
            p2[i] = p[i]
        
        # Run the kernel without boundary checks;        
        if os.path.isfile(unmodified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, KERNEL_PARAMS,
                [p, t_in, t_out, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc], num_threads)        
            exec_time_unmodified += [exec_time]
            exec_time_k_unmodified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of matrix_power: ', p[0:10])
                print('first 10 elements of matrix_temp_src: ', t_in[0:10])
                print('first 10 elements of matrix_temp_dest: ', t_out[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
        res_unmodified = copy_array(t_out)

        # Run the kernel with boundary checks;
        if os.path.isfile(modified_kernel_path):
            exec_time, exec_time_k = run_kernel(debug, num_blocks, MODIFIED_KERNEL_NAME, modified_kernel_path, KERNEL_PARAMS,
                [p2, t_in2, t_out2, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc], num_threads, deduct_sizes=DEDUCT_SIZES)
            
            exec_time_modified += [exec_time]
            exec_time_k_modified += [exec_time_k]
            # Print results;
            if debug:
                print('first 10 elements of matrix_power: ', p2[0:10])
                print('first 10 elements of matrix_temp_src: ', t_in2[0:10])
                print('first 10 elements of matrix_temp_dest: ', t_out2[0:10])
        else:
            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
        res_modified = copy_array(t_out2)

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
