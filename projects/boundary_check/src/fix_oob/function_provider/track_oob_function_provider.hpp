#ifndef _TRACK_OOB_FUNCTION_PROVIDER
#define _TRACK_OOB_FUNCTION_PROVIDER

#include "fix_oob_function_provider.hpp"

////////////////////////////////
////////////////////////////////

namespace llvm {
    struct TrackOOBFunctionProvider : public FixOOBFunctionProvider {

        TrackOOBFunctionProvider(bool debug = 0, bool print_oob_accesses = 0): FixOOBFunctionProvider(debug), print_oob_accesses(print_oob_accesses) { }

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

        // Add a sequence of instructions that update the OOB tracking array for a given access.
        // If we have an OOB access on the array at the i-th position, we update the i-th position in the tracking array with an atomic add;
        void add_update_to_oob_tracking_array(LLVMContext &context, ArrayAccess *array_access, BasicBlock *start_if_block);

        // Reference to the array that tracks OOB accesses;
        Value* track_oob_array;

        // If true, add print instruction to kernels to track OOB accesses;
        bool print_oob_accesses;

        // Reference to printf function that can be used inside CUDA kernels;
        Function *printf_function;
        // Global string pointer used in printf;
        Value *printf_string;
        Type *printf_alloca_type;
        Value *printf_alloca;

        // Obtain the CUDA printf function, https://clang.llvm.org/doxygen/CGCUDABuiltin_8cpp_source.html
        static inline llvm::Function *get_vprintf_declaration(llvm::Module &M) {
            llvm::Type *ArgTypes[] = {llvm::Type::getInt8PtrTy(M.getContext()),
                                      llvm::Type::getInt8PtrTy(M.getContext())};
            llvm::FunctionType *VprintfFuncType = llvm::FunctionType::get(
                llvm::Type::getInt32Ty(M.getContext()), ArgTypes, false);
            
            if (auto* F = M.getFunction("vprintf")) {
                // Our CUDA system header declares vprintf with the right signature, so
                // nobody else should have been able to declare vprintf with a bogus
                // signature.
                assert(F->getFunctionType() == VprintfFuncType);
                return F;
            }
            
            // vprintf doesn't already exist; create a declaration and insert it into the
            // module.
            return llvm::Function::Create(VprintfFuncType, llvm::GlobalVariable::ExternalLinkage, "vprintf", &M);
        }
    };
}

#endif