import os
import time
import argparse
import polyglot
from java.lang import System

from trufflecuda_python_utils import *

##############################
##############################

OPT_LEVEL = "O2"
SIMPLIFICATION= "simplify_accesses"

UNMODIFIED_CUBIN_NAME = "axpy_checked.cubin"
UNMODIFIED_KERNEL_NAME = "axpy_checked"
UNMODIFIED_KERNEL_FOLDER = "../../unmodified_kernel_cubin"
UNMODIFIED_KERNEL_PARAMS = "pointer, pointer, float, uint32, pointer"

MODIFIED_CUBIN_NAME = "axpy.cubin"
MODIFIED_KERNEL_NAME = "axpy"
MODIFIED_KERNEL_FOLDER = "../../cubin/"
MODIFIED_KERNEL_PARAMS = "pointer, pointer, float, pointer"

AXPY_KERNEL = """
extern "C" __global__ void axpy(float* x, float* y, float a, int n, float *z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = a * x[i] + y[i];
    }
}
"""

##############################
##############################

def main(args):
    NUM_THREADS = 32

    num_elements = args.num_elements if args.num_elements else 1000
    num_blocks = (num_elements + NUM_THREADS - 1) // NUM_THREADS

    opt_level = args.opt_level if args.opt_level else OPT_LEVEL
    simplify = "simplify_accesses" if args.simplify else SIMPLIFICATION

    num_test = args.num_test if args.num_test else 1
    debug = args.debug

    # Use separate data for the 2 kernels, to have the same memory transfers across tests;
    x = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    y = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    res = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    
    x2 = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    y2 = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    res2 = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    
    x3 = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    y3 = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    res3 = polyglot.eval(language='grcuda', string='float[{}]'.format(num_elements))
    sizes = polyglot.eval(language='grcuda', string='int[{}]'.format(3))
    sizes[0] = num_elements
    sizes[1] = num_elements
    sizes[2] = num_elements
    a = 2.0

    exec_time_unmodified = []
    exec_time_modified = []
    
    # Load kernel from source;
    build_kernel = polyglot.eval(language="grcuda", string="buildkernel")
    kernel = build_kernel(AXPY_KERNEL, "axpy", "pointer, pointer, float, sint32, pointer")
    
    # Load unmodified CUBIN;
    kernel_path = os.path.join(UNMODIFIED_KERNEL_FOLDER, OPT_LEVEL, UNMODIFIED_CUBIN_NAME)
    code = """bindkernel("{}", "{}", "{}", "{}")""".format(kernel_path, UNMODIFIED_KERNEL_NAME, UNMODIFIED_KERNEL_PARAMS, False)
    kernel_2 = polyglot.eval(language='grcuda', string=code)
    
    # Load modified CUBIN;
    kernel_path = os.path.join(MODIFIED_KERNEL_FOLDER, OPT_LEVEL, SIMPLIFICATION, MODIFIED_CUBIN_NAME)
    code = """bindkernel("{}", "{}", "{}", "{}")""".format(kernel_path, MODIFIED_KERNEL_NAME, MODIFIED_KERNEL_PARAMS, True)
    kernel_3 = polyglot.eval(language='grcuda', string=code)
    
    exec_time_source = []

    for i in range(num_test):
        for i in range(num_elements):
            x[i] = i
            y[i] = i
            res[i] = 0
            x2[i] = i
            y2[i] = i
            res2[i] = 0
            x3[i] = i
            y3[i] = i
            res3[i] = 0
            
        # Run kernel from source;
        start = System.nanoTime()
        kernel(num_blocks, NUM_THREADS)(x, y, a, num_elements, res)
        exec_time = (System.nanoTime() - start) / 1_000_000_000
        print(f"exec time: {exec_time:.6f} sec")
        if debug:
            print('first 10 elements of x: ', x[0:10])
            print('first 10 elements of y: ', y[0:10])
            print('first 10 elements of res:', res[0:10])
        exec_time_source += [exec_time]
        
        # Run the unmodified CUBIN;
        start = System.nanoTime()
        kernel_2(num_blocks, NUM_THREADS)(x2, y2, a, num_elements, res2)
        exec_time = (System.nanoTime() - start) / 1_000_000_000
        print(f"exec time: {exec_time:.6f} sec")
        if debug:
            print('first 10 elements of x: ', x2[0:10])
            print('first 10 elements of y: ', y2[0:10])
            print('first 10 elements of res:', res2[0:10])
        exec_time_unmodified += [exec_time]
        
        # Run the unmodified CUBIN;
        start = System.nanoTime()
        kernel_3(num_blocks, NUM_THREADS)(x3, y3, a, res3)
        exec_time = (System.nanoTime() - start) / 1_000_000_000
        print(f"exec time: {exec_time:.6f} sec")
        if debug:
            print('first 10 elements of x: ', x3[0:10])
            print('first 10 elements of y: ', y3[0:10])
            print('first 10 elements of res:', res3[0:10])
        exec_time_modified += [exec_time]
        
#        # Run the kernel without boundary checks;
#        unmodified_kernel_path = os.path.join(UNMODIFIED_KERNEL_FOLDER, opt_level, UNMODIFIED_CUBIN_NAME)
#        if os.path.isfile(unmodified_kernel_path):
#            exec_time, exec_time_k = run_kernel(debug, num_blocks, UNMODIFIED_KERNEL_NAME, unmodified_kernel_path, UNMODIFIED_KERNEL_PARAMS, [x, y, a, len(x), res])
#            
#            exec_time_unmodified += [exec_time]
#            exec_time_k_unmodified += [exec_time_k]
#            # Print results;
#            if debug:
#                print('first 10 elements of x: ', x[0:10])
#                print('first 10 elements of y: ', y[0:10])
#                print('first 10 elements of res:', res[0:10])
#        else:
#            raise FileNotFoundError("ERROR: kernel {} not found!".format(unmodified_kernel_path))
#        res_unmodified = copy_array(res)
#
#        # Run the kernel with boundary checks;
#        modified_kernel_path = os.path.join(MODIFIED_KERNEL_FOLDER, opt_level, simplify, MODIFIED_CUBIN_NAME)
#        if os.path.isfile(modified_kernel_path):
#            exec_time, exec_time_k = run_kernel(debug, num_blocks, MODIFIED_KERNEL_NAME, modified_kernel_path, MODIFIED_KERNEL_PARAMS,
#             [x2, y2, a, res2], deduct_sizes=True)
#            
#            exec_time_modified += [exec_time]
#            exec_time_k_modified += [exec_time_k]
#            # Print results;
#            if debug:
#                print('first 10 elements of x: ', x2[0:10])
#                print('first 10 elements of y: ', y2[0:10])
#                print('first 10 elements of res:', res2[0:10])
#        else:
#            raise FileNotFoundError("ERROR: kernel {} not found!".format(modified_kernel_path))
#        res_modified = copy_array(res2)
#
#        errors += [check_array_equality(res_unmodified, res_modified, debug=debug)]

    # Print execution times;
#    if debug:
#        print("\nSUMMARY - UNMODIFIED KERNEL:")
#        print("    overall exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_unmodified), std(exec_time_unmodified)))
#        print("    kernel exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_k_unmodified), std(exec_time_k_unmodified)))
#        print("\nSUMMARY - MODIFIED KERNEL:")
#        print("    overall exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_modified), std(exec_time_modified)))
#        print("    kernel exec. time: {:.2f}±{:.2f} μs".format(mean(exec_time_k_modified), std(exec_time_k_modified)))
#        print("errors: {:.2f}±{:.2f}".format(mean(errors), std(errors)))
#    else:
#        for i in range(len(errors)):
#            print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(
#                i,
#                num_elements,
#                opt_level,
#                simplify,
#                exec_time_k_unmodified[i],
#                exec_time_unmodified[i],
#                exec_time_k_modified[i],
#                exec_time_modified[i],
#                errors[i]))

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
