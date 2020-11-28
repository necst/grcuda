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
        // Add an if-statement: in this case, we jump in "start_if_block" if the index is >= than the array boundary, i.e. if we have an OOB access to track;
        add_array_size_if_statement(access, start_if_block, end_if_block, protect_lower_bounds, positive_cuda_dependent_values, CmpInst::Predicate::ICMP_SGE);

        ir_updated = true;
        access_number++;
    }
    return ir_updated;
}

}