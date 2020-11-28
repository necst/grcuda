#ifndef _FIX_OOB_FUNCTION_PROVIDER
#define _FIX_OOB_FUNCTION_PROVIDER

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Analysis/OrderedInstructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Dominators.h"
#include "cxxabi.h"
#include <string>
#include <iostream>
#include <set>
#include <queue>

#include "../../utils.hpp"
#include "../array_access.hpp"

////////////////////////////////
////////////////////////////////

namespace llvm {
    struct FixOOBFunctionProvider {

        FixOOBFunctionProvider(bool debug = 0): debug(debug) { }

        // Process the argument list:
        // assume that each input pointer/array has its size, expressed as integer, after it.
        // Return false if 1! pointer argument was found, 
        // i.e. the sizes array is missing, or it is present but no pointer argument exist;
        virtual bool parse_argument_list(Function &F, std::vector<Value *> &array_arguments);

        // Insert at the beginning of the kernel code a sequence of instructions 
        // that loads each array value and stores it in a local value;
        virtual void insert_load_sizes_instructions(Function &F, std::vector<Value *> &array_arguments, std::map<Value *, Value *> &input_array_sizes);

        // Handle array/pointer accesses: if the index used to access the array is CUDA-dependent,
        // keep track of the array access;
        bool handle_array_access(
            Instruction &I,
            std::map<Value *, Value *> &input_array_sizes,
            std::map<Value *, Value *> &input_alias_map,
            DominatorTree *DT,
            std::vector<ArrayAccess *> &array_accesses
        );

        // Find the last instruction in the scope of a given array access, whose corresponding GEP is "getI", by exploring the dominator tree.
        // This is the last instruction that will be covered by the boundary check;
        virtual Instruction* find_array_access_end(ArrayAccess *access, DominatorTree *DT, GetElementPtrInst *getI);

        // Identify which array accesses are already covered by a boundary check identical to what we are going to add, and discard them from further analyses;
        void filter_accesses_with_protection(
            std::vector<ArrayAccess *> &array_accesses,
            std::vector<ArrayAccess *> &array_accesses_to_be_protected,
            std::map<Value *, Value *> &input_alias_sizes_map
        );

        // Check if two arrays x and y have the same size (checking also if they are an alias);
        bool check_if_same_size(Value *x, Value *y, std::map<Value *, Value *> &input_alias_sizes_map);

        // Modify the existing IR with additional boundary checks;
        virtual bool add_array_access_protection(
            LLVMContext &context,
            std::vector<ArrayAccess *> &array_accesses_postprocessed,
            bool protect_lower_bounds,
            std::set<Value *> &positive_cuda_dependent_values);

        // Reference to the array that stores sizes;
        Value* sizes_array;

        // If true, print stuff;
        bool debug;
    };
}

#endif