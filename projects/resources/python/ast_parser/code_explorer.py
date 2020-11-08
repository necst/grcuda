#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import deque, defaultdict
import clang.cindex as ci
from resources.python.enriched_tree import EnrichedTree
from resources.python.cuda_array import CudaArray, ARRAY_TYPES

KERNEL_IDENTIFIERS = ["__global__", "__device__", "__host__"]
DEVICE_KERNEL_IDENTIFIERS = ["__global__", "__device__"]
CUDA_REF_EXPR = ["threadIdx", "blockIdx", "blockDim", "gridDim"]


class CodeExplorer:
    
    def __init__(self, root, translation_unit_name=""):
        self.see_all = False
        self.root = root
        self.translation_unit_name = translation_unit_name
        self.translation_unit_basename = os.path.basename(translation_unit_name)
        self.trees = []
        self.kernels = self.get_kernels()
        self.macro_instantiations = defaultdict(list)
        self._macro_instantiation_locations = {}
        self.get_macro_instantiations()

    @staticmethod
    def check_kernel_name(node, kernel_name, check_displayname=False) -> bool:
        return node.spelling == kernel_name or (check_displayname and node.displayname.split("(")[0] == kernel_name)

    @staticmethod
    def find_array_name(node, arrays) -> str:
        """
        Find the name of the array which is accessed with a ci.CursorKind.ARRAY_SUBSCRIPT_EXPR.
        The name is identified by a hierarchy of nodes ARRAY_SUBSCRIPT_EXPR->UNEXPOSED_EXPR->DECL_REF_EXPR,
        and the variable name is the "displayname" of the last node.
        In case the array is accessed with a variable, pass the name of the variable to avoid obtaining
        its name instead.
        :param node: an array for which we want to obtain the name
        :param arrays: list of arrays present in the kernel
        :return: the variable name of the array
        """
        temp_queue = deque([node])
        while len(temp_queue) > 0:
            temp_node = temp_queue.pop()
            if temp_node.kind == ci.CursorKind.DECL_REF_EXPR and temp_node.displayname in arrays:
                return temp_node.displayname
            for child in temp_node.get_children():
                temp_queue.append(child)
        return ""

    def is_array(self, node, tree) -> bool:
        """
        Check if a node denotes an array or pointer declaration (or an array/pointer passed as function parameter).
        If so, store it in the kernel tree;
        :param node: the node to be inspected
        :param tree: the [[EnrichedTree]] where arrays are stored
        :return: if the node denotes an array
        """
        if node.kind in [ci.CursorKind.VAR_DECL, ci.CursorKind.PARM_DECL] and node.type.kind in ARRAY_TYPES:

            # Multidimensional arrays are arrays composed by other arrays.
            # Obtain the size of each dimension in an iterative way;
            sizes = []
            array_types = []
            variable_types = []
            curr_type = node.type
            while curr_type.kind in ARRAY_TYPES:
                sizes += [curr_type.get_array_size()]
                array_types += [curr_type.kind]
                variable_types += [curr_type.get_pointee().kind
                                   if curr_type.kind == ci.TypeKind.POINTER else curr_type.get_array_element_type().kind]
                curr_type = curr_type.get_pointee() \
                    if curr_type.kind == ci.TypeKind.POINTER else curr_type.get_array_element_type()

            # Retrieve the textual expression used to instantiate the array, if available.
            # A VAR_DECL has > 1 child, which can be an INTEGER_LITERAL or an EXPRESSION;
            children = [x for x in node.get_children()]
            access_expressions = []
            for c in children:
                access_location = (c.location.line, c.location.column)
                if access_location in self._macro_instantiation_locations:
                    access_expressions += [self._macro_instantiation_locations[access_location].displayname]

            tree.arrays[node.spelling] = CudaArray(
                name=node.spelling,
                size=sizes,
                size_expr=access_expressions,
                array_type=array_types,
                variable_type=variable_types,
                is_kernel_parameter=node.kind == ci.CursorKind.PARM_DECL
            )
            return True
        return False

    @staticmethod
    def is_variable_cuda_dependent(node, tree) -> bool:
        """
        If a node is a variable declaration, check if it contains a reference to a CUDA "index" (e.g. threadIdx),
        or if it uses another marked variable.
        If so, mark that node and variable
        :param node: the node that is analyzed
        :param tree: [[EnrichedTree]] used to keep track of variables that have been marked already
        :return: if the node contained a marked variable
        """
        if node.kind == ci.CursorKind.VAR_DECL:
            temp_queue = deque([node])
            while len(temp_queue) > 0:
                temp_node = temp_queue.pop()
                for child in temp_node.get_children():
                    if child.kind == ci.CursorKind.DECL_REF_EXPR and\
                            (child.displayname in CUDA_REF_EXPR or child.displayname in tree.marked_variables):
                        return True
                    else:
                        temp_queue.append(child)
        return False

    @staticmethod
    def process_array_access(node, tree):
        """
        If a node is an array access,
        check if it is accessed with an expression that contains a CUDA "index" (e.g. threadIdx),
        or with a variable that contains CUDA "indices".
        If so, mark that node and array access
        :param node: the node that is analyzed
        :param tree: [[EnrichedTree]] where information about array accessed are stored
        """

        if node.kind == ci.CursorKind.ARRAY_SUBSCRIPT_EXPR and node.hash not in tree.array_accesses_hashes:
            # Obtain the array name;
            array_name = CodeExplorer.find_array_name(node, tree.arrays)
            if not array_name:
                raise KeyError(f"Found access to non-existing array "
                               f"at line={node.location.line}, col={node.location.column}")

            # Obtain the accesses to the array, and their depth;
            queue = deque([node])
            accesses = []
            accessed_nodes = []
            while len(queue) > 0:
                # The array access subtree is processed with a Breadth-First visit to obtain the outermost access first;
                curr_node = queue.popleft()
                if curr_node.kind in [ci.CursorKind.DECL_REF_EXPR, ci.CursorKind.BINARY_OPERATOR] and curr_node.displayname != array_name:
                    accesses += [curr_node]
                # Handle accesses to multi-dimensional arrays, and keep track to inner accesses;
                elif curr_node.kind == ci.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                    accessed_nodes += [curr_node]
                # Handle accesses with an integer literal, e.g. x[0];
                elif curr_node.kind == ci.CursorKind.INTEGER_LITERAL:
                    accesses += [curr_node]
                # Process children only if the node isn't a binary operator, because we want to keep track
                # of the overall expression;
                if curr_node.kind != ci.CursorKind.BINARY_OPERATOR:
                    for child in curr_node.get_children():
                        queue.append(child)

            # If we found any access, store the result;
            if len(accesses) > 0:
                new_array_access = {
                    "accessed_node": accessed_nodes,
                    "accessing_node": accesses,
                    "accessed_variable": array_name,
                    "accessing_variable": [a.displayname for a in accesses]
                }
                # Check if any variable used to access the array is CUDA-dependent;
                marked = any((a in tree.marked_variables) or (a in CUDA_REF_EXPR)
                             for a in new_array_access["accessing_variable"])
                # If the array is accessed with a complex expression, we have to process
                # the remaining expression subtree to check each variable that is used;
                for a in accesses:
                    if a.kind == ci.CursorKind.BINARY_OPERATOR:
                        queue = deque([a])
                        while len(queue) > 0 and not marked:
                            temp_node = queue.pop()
                            if temp_node.displayname in tree.marked_variables or temp_node.displayname in CUDA_REF_EXPR:
                                marked = True
                            for c in temp_node.get_children():
                                queue.append(c)

                if marked:
                    tree.arrays[array_name].accesses["marked"] += [new_array_access]
                else:
                    tree.arrays[array_name].accesses["unmarked"] += [new_array_access]
                # Bookkeping of the new array accesses that have been found;
                for a in accessed_nodes:
                    tree.array_accesses_hashes.add(a.hash)
            else:
                raise ValueError(f"Found array access to {array_name} without any access expression, "
                                 f"at line={node.location.line}, col={node.location.column}")

    def get_kernels(self) -> list:
        """
        Parse the original source code to obtain the list of GPU kernels executed on device, 
        identified by __global__ or __device__ tags.
        These tags are removed by the clang parser, so this approach is necessary;
        """
        tokens = EnrichedTree.get_tokens(self.root, join=False)
        kernels = []
        for i, t1 in enumerate(tokens):
            if t1 in DEVICE_KERNEL_IDENTIFIERS:
                for j, t2 in enumerate(tokens[(i + 1):]):
                    if t2 == "(":
                        kernel_name = tokens[i + j]
                        if kernel_name not in kernels:
                            kernels += [kernel_name]
                        break
        return kernels

    def get_macro_instantiations(self) -> None:
        """
        Obtain a dictionary that associates to each macro instantiation its node and position;
        """
        for node in self.root.get_children():
            # Skip macros defined in other files;
            node_filename = os.path.basename(str(node.location.file))
            if node_filename == self.translation_unit_basename and node.kind == ci.CursorKind.MACRO_INSTANTIATION:
                self.macro_instantiations[node.spelling] += [{
                    "node": node,
                    "line": node.location.line,
                    "column": node.location.column
                }]
                self._macro_instantiation_locations[(node.location.line, node.location.column)] = node
        
    def create_kernel_tree(self, kernel_name="", check_displayname=False, skip_includes=True) -> EnrichedTree:
        
        queue = deque([{"node": self.root, "depth": 0}])
        new_tree = None
        
        # Find the specified kernel root;
        if kernel_name:      
            while len(queue) > 0:
                curr_elem = queue.popleft()
                node = curr_elem["node"]
                depth = curr_elem["depth"]

                # If we find the desired kernel root, re-initialize the queue and interrupt the exploration.
                # Also, initialize the enriched tree with the kernel root as first node.
                # The queue elements have an additional "children" field, used to assemble the tree;
                if node.kind == ci.CursorKind.FUNCTION_DECL \
                        and CodeExplorer.check_kernel_name(node, kernel_name, check_displayname):
                    new_tree = EnrichedTree(node, kernel_name)
                    new_tree.tree = {"node": node, "depth": 0, "children": []}
                    queue = deque([new_tree.tree])
                    break
                
                for child in node.get_children():
                    # Skip includes (perform this check only if kernel name is empty, as optimization);
                    if skip_includes or kernel_name:
                        child_filename = os.path.basename(str(child.location.file))
                        if child_filename != self.translation_unit_basename:
                            continue
                    queue.append({"node": child, "depth": depth + 1})
                        
        if kernel_name and not new_tree:
            raise ValueError(f"kernel {kernel_name} not found!")
        # Initialize the queue if we look at the entire translation unit instead of a single kernel;
        if not kernel_name:
            new_tree = EnrichedTree(self.root, self.translation_unit_basename)
            new_tree.tree = {"node": self.root, "depth": 0, "children": []}
            queue = deque([new_tree.tree])

        # Explore the desired kernel (or the whole tree if no kernel name was specified);
        if len(queue) != 1:
            print(f"ERROR: the queue has size {len(queue)}, which is != 1")
        while len(queue) > 0:
            curr_elem = queue.pop()
            node = curr_elem["node"]
            depth = curr_elem["depth"]

            ###################################
            # Perform checks for OoB accesses #
            ###################################

            # Check if a node is an array (or a pointer). If so, keep track of it;
            self.is_array(node, new_tree)

            # If a node is a variable declaration, check if it contains a reference to a CUDA "index" (e.g. threadIdx).
            # If so, mark that node and variable;
            if new_tree and CodeExplorer.is_variable_cuda_dependent(node, new_tree):
                new_tree.marked_variables[node.displayname] = node

            # Check if an array access is performed using a marked variable or a CUDA "index-dependent" expression.
            # If so, add it to the tree data structure;
            CodeExplorer.process_array_access(node, new_tree)

            ######################################
            # Add the node children to the queue #
            ######################################

            # Process children of the current node;
            for child in reversed([x for x in node.get_children()]):
                # Skip includes (perform this check only if kernel name is empty, as optimization);
                if skip_includes or kernel_name:
                    child_filename = os.path.basename(str(child.location.file))
                    if child_filename != self.translation_unit_basename:
                        continue
                # Create a new children, and store it into the enriched tree;
                new_node = {"node": child, "depth": depth + 1, "children": []}
                curr_elem["children"] += [new_node]
                queue.append(new_node)
        
        # Store the new tree;
        self.trees += [new_tree]
        return new_tree
