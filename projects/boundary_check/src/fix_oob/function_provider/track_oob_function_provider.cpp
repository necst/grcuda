#include "track_oob_function_provider.hpp"

namespace llvm {

////////////////////////////////
////////////////////////////////

bool TrackOOBFunctionProvider::parse_argument_list(Function &F, std::vector<Value *> &array_arguments) {
    // Process the function arguments to obtain a list of arrays and sizes;
    std::vector<Argument *> args;
    for (auto &arg : F.args()) {
        if (debug) {
            outs() << "  argument: ";
            arg.print(outs());
            outs() << "\n";
        }
        args.push_back(&arg);
    }
    // Process the argument list:
    // assume that each input pointer/array has its size, expressed as integer, at the end of the signature.
    std::vector<Value *> all_arguments;
    for (auto &arg : F.args()) {
        if (arg.getType()->isArrayTy() || arg.getType()->isPointerTy()) {
            array_arguments.push_back(&arg);
        }
        all_arguments.push_back(&arg);
    }
    // Exclude the last array, which holds the sizes.
    // Keep a reference to this array, for later use;
    if (array_arguments.size() == 2) {
        if (debug) {
            outs() << "WARNING: only " << array_arguments.size() << " extra pointer argument(s) found!\n";
        }
        return false;
    }
    track_oob_array = array_arguments.back();
    array_arguments.pop_back();
    sizes_array = array_arguments.back();
    array_arguments.pop_back();

    return true;
}

////////////////////////////////
////////////////////////////////

Instruction* TrackOOBFunctionProvider::find_array_access_end(ArrayAccess *access, DominatorTree *DT, GetElementPtrInst *getI) {
    return access->get_array_value;
}

////////////////////////////////
////////////////////////////////

bool TrackOOBFunctionProvider::add_array_access_protection(
    LLVMContext &context,
    std::vector<ArrayAccess *> &array_accesses_postprocessed,
    bool protect_lower_bounds,
    std::set<Value *> &positive_cuda_dependent_values
    ) {

    bool ir_updated = false;

    // Do a pass over the array accesses, and add "if" statements before each GEP instruction to track them and update the OOB debug array if any OOB accesses occur;
    int access_number = 0;
    for (auto access : array_accesses_postprocessed) {

        // We insert an if-statement before the GEP, so we split the basic block where the GEP appears right before the GEP itself,
        // and add a simple basic block where we jump if we have an OOB and update the OOB debug array;

        // Start a new block right before the first instruction of the array access (i.e. the GEP).
        // This represents the start of the "if" statement;
        Instruction *start_if_instruction = access->start; // obtain_array_access_start(access);
        BasicBlock *start_if_block = SplitBlock(start_if_instruction->getParent(), start_if_instruction);
        start_if_block->setName(formatv("start_array_access_tracking_{0}", access_number));

        // The end instruction is the GEP itself, we split the basic block right before it;
        Instruction *end_if_instruction = access->end;
        // Create a new block after the array access instruction.
        // This represents the end of the "if" statement;
        BasicBlock *end_if_block = SplitBlock(end_if_instruction->getParent(), end_if_instruction);
        
        if (end_if_block) {
            end_if_block->setName(formatv("end_array_access_tracking_{0}", access_number));
        }
/*
        // Delete the unconditional branch added by the block creation;
        start_if_instruction->getParent()->getPrevNode()->getTerminator()->eraseFromParent();

        // Add the sequence of instructions required to check the array size;
        // The instructions are added at the end of the block before the start of the "then" block;
        IRBuilder<> builder(start_if_instruction->getParent()->getPrevNode());
        std::vector<ICmpInst *> icmp_list;
        Value *last_and = nullptr;

        // Set that stores array indices, it avoids the addition of redundant >= accesses;
        std::set<Value *> indices_set;

        for (uint i = 0; i < access->extended_cmp_inst_for_size_check.size(); i++) {
            auto size_check = access->extended_cmp_inst_for_size_check[i];

            Value *first_operand = nullptr;
            // If the first part of the size check is a pointer, add a load instruction to read it;
            if (size_check.first->getType()->isPointerTy()) {
                first_operand = builder.CreateLoad(size_check.first);
            } else {
                first_operand = size_check.first;
            }

            // Add "i >= 0" protection, if we didn't already and if the index is not guaranteed to be positive;
            if (protect_lower_bounds && !is_in_set(&indices_set, size_check.first) && !is_in_set(&positive_cuda_dependent_values, size_check.first)) {
                icmp_list.push_back(cast<ICmpInst>(builder.CreateICmp(
                    CmpInst::Predicate::ICMP_SGE,
                    first_operand,
                    ConstantInt::get(first_operand->getType(), 0, true))));
                indices_set.insert(size_check.first);
            }

            // Array sizes are stored as int64. If the index is an int32, use the existing index casting to 64 bit,
            // or add a new one. The only exception is when comparing against a fixed size array, which might have an int32 size.
            // The i == 0 ensures that the existing casting is re-used only for the first boundary check in the case of merged accesses,
            // as the existing casting of subsequent accesses might be much further in the code;
            if (first_operand->getType()->isIntegerTy(32) && size_check.second->getType()->isIntegerTy(64)) {
                if (access->index_casting && i == 0) {
                    first_operand = access->index_casting;
                } else {
                    first_operand = builder.CreateIntCast(first_operand, size_check.second->getType(), true, formatv("cast_index_{0}", i));
                }
            }
            // Create an "index < array_size" instruction;
            icmp_list.push_back(cast<ICmpInst>(builder.CreateICmp(
                CmpInst::Predicate::ICMP_SLT,
                first_operand,
                size_check.second)));
        }

        // Join all the "<" instructions with "and" insturctions;
        for (uint i = 0; i < icmp_list.size() - 1; i++) {
            Value *first_operand = i == 0 ? icmp_list[0] : last_and;
            last_and = builder.CreateAnd(first_operand, icmp_list[i + 1]);
        }
        // Add an "if" statement before the array access is performed, after the last "and",
        // or after the first (and last) "<" instruction if no "and" is present;
        if (last_and) {
            builder.CreateCondBr(last_and, start_if_block, end_if_block);
        } else {
            builder.CreateCondBr(icmp_list.back(), start_if_block, end_if_block);
        }
*/
        ir_updated = true;
        access_number++;
    }
    return ir_updated;
}

}