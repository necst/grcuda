#ifndef _TRACK_OOB_FUNCTION_PROVIDER
#define _TRACK_OOB_FUNCTION_PROVIDER

#include "fix_oob_function_provider.hpp"

////////////////////////////////
////////////////////////////////

namespace llvm {
    struct TrackOOBFunctionProvider : public FixOOBFunctionProvider {

        TrackOOBFunctionProvider(bool debug = 0): FixOOBFunctionProvider(debug) { }

        // Process the argument list:
        // in this case, assume we have 2 additional arrays, one containing sizes and the other used to count OOB accesses;
        bool parse_argument_list(Function &F, std::vector<Value *> &array_arguments) override;

        // In this case, the start and end of the array coincide with the GEP itself, as we are simply adding a boundary check before the instruction;
        Instruction* find_array_access_end(ArrayAccess *access, DominatorTree *DT, GetElementPtrInst *getI) override;

        // Add a simple if-statement before the GEP that will check if the value used for the GEP is not OOB.
        // If it is, increment the OOB counter for the array that presents an OOB access, and print a warning;
        bool add_array_access_protection(
            LLVMContext &context,
            std::vector<ArrayAccess *> &array_accesses_postprocessed,
            bool protect_lower_bounds,
            std::set<Value *> &positive_cuda_dependent_values) override;

        // Reference to the array that tracks OOB accesses;
        Value* track_oob_array;
    };
}

#endif