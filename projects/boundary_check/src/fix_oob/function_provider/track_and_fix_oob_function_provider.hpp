#ifndef _FIX_AND_TRACK_OOB_FUNCTION_PROVIDER
#define _FIX_AND_TRACK_OOB_FUNCTION_PROVIDER

#include "track_oob_function_provider.hpp"

////////////////////////////////
////////////////////////////////

namespace llvm {
    struct TrackAndFixOOBFunctionProvider : public TrackOOBFunctionProvider {

        TrackAndFixOOBFunctionProvider(bool debug = 0, bool print_oob_accesses = 0): TrackOOBFunctionProvider(debug, print_oob_accesses) { }

        // Use the same behaviour as FixOOBFunctionProvider;
        Instruction* find_array_access_end(ArrayAccess *access, DominatorTree *DT, GetElementPtrInst *getI) override;

        // Do a first pass with the same behaviour as FixOOBFunctionProvider, 
        // but split the target end block to add an "else" block with tracking;
        bool add_array_access_protection(
            LLVMContext &context,
            std::vector<ArrayAccess *> &array_accesses_postprocessed,
            bool protect_lower_bounds,
            std::set<Value *> &positive_cuda_dependent_values) override;

        // Add a sequence of instructions that update the OOB tracking array for a given access.
        // If we have an OOB access on the array at the i-th position, we update the i-th position in the tracking array with an atomic add;
        // void add_update_to_oob_tracking_array(LLVMContext &context, ArrayAccess *array_access, BasicBlock *start_if_block);
    };
}

#endif