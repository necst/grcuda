#ifndef _FIX_OOB_CUDA_PASS
#define _FIX_OOB_CUDA_PASS

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

#include "../utils.hpp"
#include "array_access.hpp"
#include "function_provider/fix_oob_function_provider.hpp"
#include "function_provider/track_oob_function_provider.hpp"

////////////////////////////////
////////////////////////////////

namespace llvm {

#define NUM_PROTECTION_TYPES 3
enum OOB_PROTECTION_TYPE {
    PREVENT = 0,
    TRACK = 1,
    PREVENT_AND_TRACK = 2
};

struct TestPass : public FunctionPass {
    TestPass();

    OOB_PROTECTION_TYPE protection_type = PREVENT;

    virtual bool runOnFunction(Function &F);

    // Check if an input array is stored inside some other value.
    // If so, mark this other value as alias.
    // Also, propagate aliases;
    bool check_if_input_array_alias(Instruction &I);
    // Same thing, with array sizes;
    bool check_if_input_array_size_alias(Instruction &I);

    // Check if the specified call instruction refers to a CUDA positional index lookup;
    bool is_instruction_call_to_cuda_position(CallInst &I);

    bool instruction_has_positive_result(Instruction &I);
    bool is_positive_constant(Value *V);

    // Check if the provided instruction depends (directly or indirectly) from a CUDA-dependent index.
    // This happens if the instruction is a function call to a CUDA-dependent index, or if one of the instruction
    // operands is CUDA-dependent. If the instruction is CUDA-dependent, store its value in a set.
    // In the case of "store" instructions, keep track of the destination operand instead of the instruction itself;
    void handle_instruction_if_cuda_dependent(Instruction &I);

    void check_if_positive_cuda_dependent(Instruction &I);

    // Check if the span of instructions covered by access x
    // is included inside the span covered by y;
    bool inside_access(ArrayAccess *x, ArrayAccess *y);

    // Check if the instruction set of one array access is a subset of the set of another array access.
    // Indeed, if the "larger" access is not performed, the "smaller" access also won't be performed.
    // We can add the boundary condition of the "smaller" array access into the "larger" access,
    // and process only the "larger" access;
    void simplify_array_accesses();

    Instruction* obtain_array_access_start(ArrayAccess* access);

    // Printing function used for debugging.
    // Print all the array accesses done by the kernel;
    void print_array_accesses(std::vector<ArrayAccess *> &array_accesses);

    // Return true if "second" appears as the followers of "first", using a BFS;
    bool check_if_bb_follows_another(BasicBlock* first, BasicBlock* second);

    ////////////////////////////////
    ////////////////////////////////

    bool is_value_cuda_dependent(Value &V) {
        return is_in_set(&cuda_dependent_values, V);
    }

    ////////////////////////////////
    ////////////////////////////////

    // The address of this member is used to uniquely identify the class. This is
    // used by LLVM's own RTTI mechanism;
    static char ID;
    std::string kernel_name;
    std::set<Value *> cuda_dependent_values{};
    std::set<Value *> positive_cuda_dependent_values{};
    std::vector<ArrayAccess *> array_accesses{};
    std::vector<ArrayAccess *> array_accesses_to_be_protected{};
    std::vector<ArrayAccess *> array_accesses_postprocessed{};
    // Dominator tree of the examined kernel;
    DominatorTree *DT;

    // List of array parameters in the kernel signature;
    std::vector<Value *> array_arguments;
    // Map that associates each input array to its size;
    std::map<Value *, Value *> input_array_sizes;
    // Map that associates input arrays to aliases, other instructions that contains the same value;
    std::map<Value *, Value *> input_alias_map;
    // Map that associates input array sizes to aliases, other instructions that contains the same value;
    std::map<Value *, Value *> input_alias_sizes_map;
    // Set of values used to store array sizes;
    std::set<Value *> array_sizes_set;

    // Reference to the class that implements the function required for OOB detection & protection;
    FixOOBFunctionProvider *function_provider;
};

} // namespace llvm

#endif