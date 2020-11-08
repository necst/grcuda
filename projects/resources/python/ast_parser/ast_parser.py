#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import clang.cindex as ci
from resources.python.code_explorer import CodeExplorer


if __name__ == "__main__":

    input_file = "../../examples/axpy/axpy.cu"
    # input_file = "../../examples/dot_product/dot_product.cu"
    # input_file = "../../examples/hotspot/hotspot.cu"
    # input_file = "../../examples/montecarlo/MonteCarlo_kernel.cu"

    index = ci.Index.create()
    tu = index.parse(input_file, options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD, args=[
            "--cuda-gpu-arch=sm_60",
            "-L/usr/local/cuda/lib64",
            "-lcudart_static", "-ldl", "-lrt", "-pthread"])
    print("Translation unit:", tu.spelling)
    
    explorer = CodeExplorer(tu.cursor, tu.spelling)
    kernels = explorer.kernels
    trees = []
    
    # for k in kernels:
    #     print(f"\nExploring kernel {k}...\n")
    #     new_tree = explorer.create_kernel_tree(k)
    #     trees += [new_tree]
    #     print(new_tree)
    #     print("-" * 30)
    #     print(new_tree.get_arrays_repr())
    print(explorer.create_kernel_tree())
