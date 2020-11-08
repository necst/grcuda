#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque

CGREEN = '\033[0;92m'
CRED = '\033[31m'
CBLUE = '\033[34m'
CEND = '\033[0m'


class EnrichedTree:
    
    def __init__(self, root, kernel_name=""):
        self.root = root
        self.kernel_name = kernel_name
        self.tree = None
        # Dictionary of variables whose value depends on CUDA indices or dimensions.
        # We assume that each variable has a unique name, and for now don't analyze the variable liveliness;
        self.marked_variables = {}
        self.arrays = {}
        self._repr = ""
        # Bookkeping of array accesses hashes, to avoid processing the same array access twice;
        self.array_accesses_hashes = set()

    @staticmethod
    def get_tokens(node, join=True):
        tokens = [x.spelling for x in node.get_tokens()]
        if join:
            return " ".join(tokens)
        else:
            return tokens

    def get_marked_accesses(self) -> list:
        """
        Return a list that contains all the array accesses performed using marked variables
        :return: a list of dictionaries, each containing a marked access
        """
        return [y for x in self.arrays.values() for y in x.accesses["marked"]]

    def get_arrays_repr(self) -> str:
        """
        Obtain a textual representation of the arrays stored in the tree
        :return: a textual representation of the arrays stored in the tree
        """
        title = f"- List of arrays in kernel {CBLUE}{self.kernel_name}{CEND}:"
        repr = ["-" * 30]
        repr += [title]
        repr += [str(x) for x in self.arrays.values()]
        repr += ["-" * 30]
        return "\n".join(repr)
                
    def __repr__(self) -> str:
        
        def get_node_repr(node, depth, print_tokens=True, marked=False):
            if print_tokens:
                return f"{' ' * (depth-1)}└{CRED if marked else CBLUE}{str(node.kind).replace('CursorKind.', '')}{CEND}, " \
                    f"{CRED if marked else CGREEN}{node.displayname if node.displayname else '_'}{CEND}, " \
                    f"{CRED if marked else CEND}{EnrichedTree.get_tokens(node, join=True)}{CEND} " \
                    f"{CGREEN}[line={node.location.line}, col={node.location.column}]{CEND}"
            else:
                return f"{' ' * (depth - 1)}└{CRED if marked else CBLUE}{str(node.kind).replace('CursorKind.', '')}{CEND}, " \
                    f"{CRED if marked else CGREEN}{node.displayname if node.displayname else '_'}{CEND} " \
                    f"{CGREEN}[line={node.location.line}, col={node.location.column}]{CEND}"
        
        self._repr = ""
        tree_lines = []
        queue = deque([self.tree])
        
        # Create a textual representation of each node, and put it in a list;
        while len(queue) > 0:
            curr_elem = queue.pop()
            # Check if the current element is marked;
            marked = curr_elem["node"].hash in [x.hash for x in self.marked_variables.values()]
            marked |= curr_elem["node"].hash in [y.hash for x in self.get_marked_accesses() for y in x["accessed_node"]]
            tree_lines += [{"repr": get_node_repr(curr_elem["node"], curr_elem["depth"], marked=marked), "depth": curr_elem["depth"]}]
            for child in curr_elem["children"]:
                queue.append(child)
        
        # Replace leading whitespaces in the representation of each node with symbols that denote their hierarchy;       
        for i in reversed(range(len(tree_lines))):
            curr_depth = tree_lines[i]["depth"]
            for j in reversed(range(len(tree_lines[:i]))):
                curr_line_j = tree_lines[j]["repr"]
                if curr_depth > 0 and curr_line_j[curr_depth - 1] == " ":
                    tree_lines[j]["repr"] =\
                        tree_lines[j]["repr"][:(curr_depth-1)] + "│" + tree_lines[j]["repr"][curr_depth:]
                elif curr_depth > 0 and curr_line_j[curr_depth - 1] == "└":
                    tree_lines[j]["repr"] =\
                        tree_lines[j]["repr"][:(curr_depth-1)] + "├" + tree_lines[j]["repr"][curr_depth:]
                else:
                    break
            # Remove └ from nodes that are single children;
            if curr_depth > 0 and tree_lines[i]["repr"][curr_depth-1] == "└" \
                    and tree_lines[i-1]["repr"][curr_depth-1] not in [" ", "│", "├"]:
                tree_lines[i]["repr"] =\
                    tree_lines[i]["repr"][:(curr_depth-1)] + " " + tree_lines[i]["repr"][curr_depth:]
        self._repr = "\n".join([x["repr"] for x in tree_lines])
        return self._repr
        
    def __str__(self) -> str:
        return self.__repr__()
