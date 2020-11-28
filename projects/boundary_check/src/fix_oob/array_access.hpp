#ifndef _ARRAY_ACCESS
#define _ARRAY_ACCESS

#include "llvm/IR/Instructions.h"
#include <vector>

////////////////////////////////
////////////////////////////////

using namespace llvm;

////////////////////////////////
////////////////////////////////

enum ArrayAccessType { LOAD,
                       STORE,
                       CALL };

// Each array access is associated to a GetElementPointer instruction,
// i.e. an address computation. A single computation might be used for
// multiple array accesses, especially if the code has been optimized;
struct ArrayAccess {
    // Type of array accesses;
    std::vector<ArrayAccessType> access_type;
    // Size of the array being accessed;
    Value *array_size;
    // These instructions compose an array/pointer access building block;
    Value *array_load;
    Instruction *index_expression;
    // Casting is optional;
    CastInst *index_casting;
    GetElementPtrInst *get_array_value;
    // List of load/store instructions that use a given array access.
    // Multiple load/stores could be associated to the same address,
    // especially if the code has been optimized:
    std::vector<Instruction *> array_access;
    // An array load can have impact over a large portion of the code.
    // Keep track of what is the first instruction of the access (usually "array_load"),
    // and of the last instruction that uses a value loaded by the access;
    Instruction *start;
    Instruction *end;

    // Store the sequence of instructions used to check the size of the array in a list.
    // After merging overlapping array accesses, the boundary check instruction
    // might be merged with other instructions;
    std::vector<std::pair<Value *, Value *>> extended_cmp_inst_for_size_check;

    // True if we need to add a boundary check for this access;
    bool requires_protection = true;
};

#endif
