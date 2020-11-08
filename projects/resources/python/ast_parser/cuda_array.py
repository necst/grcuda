#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import clang.cindex as ci
from resources.python.enriched_tree import CBLUE, CEND, CGREEN, CRED, EnrichedTree

ARRAY_TYPES = [
    ci.TypeKind.CONSTANTARRAY,
    ci.TypeKind.VECTOR,
    ci.TypeKind.INCOMPLETEARRAY,
    ci.TypeKind.VARIABLEARRAY,
    ci.TypeKind.DEPENDENTSIZEDARRAY,
    ci.TypeKind.POINTER
]


class CudaArray:

    def __init__(self, name="", size=None, size_expr=None, array_type=None, variable_type=None, is_kernel_parameter=False):
        self.name = name
        self.size = size if size else []
        self.size_expr = size_expr if size_expr else []
        self.array_type = array_type if array_type else []
        self.variable_type = variable_type if variable_type else []
        self.is_kernel_parameter = is_kernel_parameter
        self.accesses = {"marked": [], "unmarked": []}
        self.dimensions = len(self.size)

    @staticmethod
    def _join_array_info(info):
        return "[" + "][".join([str(x) for x in info]) + "]"

    def __repr__(self) -> str:
        repr = []
        repr += [f"┌ {CGREEN}array name:{CEND} {CBLUE}{self.name}{CEND}"]
        repr += [f"├──── array type: {CBLUE}{CudaArray._join_array_info(self.array_type)}{CEND}"]
        repr += [f"├──── variable type: {CBLUE}{CudaArray._join_array_info(self.variable_type)}{CEND}"]
        repr += [f"├──── is a kernel parameter? {CRED + 'yes' + CEND if self.is_kernel_parameter else CGREEN + 'no' + CEND}"]
        repr += [f"├──── known size: {CGREEN + CudaArray._join_array_info(self.size) + CEND if all(int(x) > 0 for x in self.size) else CRED + 'unknown' + CEND}"]
        repr += [f"├──── symbolic size: {CGREEN + CudaArray._join_array_info(self.size_expr) + CEND if len(self.size_expr) > 0 else CRED + 'unknown' + CEND}"]

        num_marked_accesses = len(self.accesses['marked'])
        repr += [f"├──── CUDA-dependent accesses, number={(CRED if num_marked_accesses > 0 else CGREEN) + str(num_marked_accesses) + CEND}"]
        for i, a in enumerate(self.accesses["marked"]):
            repr += [f"│        {i + 1}- variable: "
                     f"{CRED}{CudaArray._join_array_info(a['accessing_variable']) if a['accessing_variable'] else '_'}{CEND}, "
                     f"expression: {CudaArray._join_array_info([EnrichedTree.get_tokens(x) for x in a['accessing_node']])}"]

        num_unmarked_accesses = len(self.accesses['unmarked'])
        repr += [f"└──── Other accesses, number={num_unmarked_accesses}"]
        for i, a in enumerate(self.accesses["unmarked"]):
            repr += [f"         {i + 1}- variable: "
                     f"{CudaArray._join_array_info(a['accessing_variable']) if a['accessing_variable'] else '_'}, "
                     f"expression: {CudaArray._join_array_info([EnrichedTree.get_tokens(x) for x in a['accessing_node']])}"]

        return "\n".join(repr)

    def __str__(self) -> str:
        return self.__repr__()

