#include "track_and_fix_oob_function_provider.hpp"

namespace llvm {

////////////////////////////////
////////////////////////////////

Instruction* TrackAndFixOOBFunctionProvider::find_array_access_end(ArrayAccess *access, DominatorTree *DT, GetElementPtrInst *getI) {
    // Queue used to explore the instruction users in BFS fashion;
    std::queue<Instruction *> end_array_queue;
    // Set used to avoid processing the same instruction twice;
    std::set<Instruction *> seen_instructions;

    // Add to the queue the index value used to access the array,
    // as it might be used somewhere else;
    if (access->index_casting) {
        end_array_queue.push(access->index_casting);
    }

    Instruction *lastI = access->array_access[0];  // Just pick the first leaf instruction, to have a valid last instruction;
    // Initialize a queue using the array accesses;
    for (auto curr_access : access->array_access) {
        // Update the last instruction that uses a value dependent on the array access;
        if (OrderedInstructions(DT).dfsBefore(lastI, curr_access)) {
            lastI = curr_access;
        }
        // If the array access is a load, look for additional usages of both the loaded value
        // and the index used to access the array.
        // If it is a store, look for additional usages of the index.
        // If it is a function call, look for usages of both the return value and the index;
        if (auto loadI = dyn_cast<LoadInst>(curr_access)) {
            if (auto indexI = dyn_cast<Instruction>(loadI->getPointerOperand())) {
                end_array_queue.push(indexI);
            }
            end_array_queue.push(loadI);
        } else if (auto storeI = dyn_cast<StoreInst>(curr_access)) {
            if (auto indexI = dyn_cast<Instruction>(storeI->getPointerOperand())) {
                end_array_queue.push(indexI);
            }
        } else if (auto callI = dyn_cast<CallInst>(curr_access)) {
            for (auto &dataOp : callI->data_ops()) {
                // Add pointers used in the call;
                if (auto gepI = dyn_cast<GetElementPtrInst>(dataOp)) {
                    end_array_queue.push(gepI);
                }
            }
            // Add the return value to the queue, if not void;
            if (!callI->isReturnNonNull()) {
                end_array_queue.push(callI);
            }
        }
    }

    // Now we have a queue with all the pre-existing leaves, do a DSF to find further leaves and identify a common end;
    while (!end_array_queue.empty()) {
        Instruction *currI = end_array_queue.front();
        end_array_queue.pop();

        // Update the last instruction that uses a value dependent on the array access;
        if (OrderedInstructions(DT).dfsBefore(lastI, currI)) {
            lastI = currI;
        } else {
            // Check if lastI is in a predecessor block of currI;
            if (auto uniqueSucc = lastI->getParent()->getUniqueSuccessor()) {
                if (uniqueSucc == currI->getParent()) {
                    lastI = currI;
                }
            }
        }

        // Add the users of the current instruction to the queue;
        for (auto child : currI->users()) {

            Instruction *childI = dyn_cast<Instruction>(child);

            if (childI && !is_in_set(&seen_instructions, childI)) {
                end_array_queue.push(childI);
                seen_instructions.insert(childI);
            }
        }
        // If the current instruction is a store, add the index of the array being accessed.
        // We need to protect further array accesses to that array, as the value might have not be stored;
        if (auto storeI = dyn_cast<StoreInst>(currI)) {
            if (auto pointer_storeI = dyn_cast<Instruction>(storeI->getPointerOperand())) {
                if (!is_in_set(&seen_instructions, pointer_storeI)) {
                    end_array_queue.push(pointer_storeI);
                    seen_instructions.insert(pointer_storeI);
                }
            }
        }
    }
    // If the last instruction is in a different basic block, we need to cover the whole block
    // not to split it in 2 parts, where the second part might be executed but the first isn't;
    if (getI->getParent() != lastI->getParent()) {
        lastI = lastI->getParent()->getTerminator();
    }
    return lastI;
}

////////////////////////////////
////////////////////////////////

bool TrackAndFixOOBFunctionProvider::add_array_access_protection(
    LLVMContext &context,
    std::vector<ArrayAccess *> &array_accesses_postprocessed,
    bool protect_lower_bounds,
    std::set<Value *> &positive_cuda_dependent_values
    ) {

    bool ir_updated = false;

    // Do a pass over the array accesses, and add "if" statements to protect them;
    int access_number = 0;
    for (auto access : array_accesses_postprocessed) {

        // This is the basic block where the boundary check ends.
        // If the array access is not performed, or if it performed, jump to this block;
        BasicBlock *end_if_block = nullptr;

        // First, we need to understand where the boundary check is going to end.
        // If the end instruction is in the same basic block as the start, we simply split twice this block
        // (starting from the end, then the start), and replace the first unconditional branch with the boundary check.
        // If the end instruction is in a different basic block, the end instruction is a terminator of the basic block
        // (to avoid splitting it in two, which creates an unreachable scenario). In this second case,
        // the ending basic block will be the basic block that follows the one where the end instruction is;

        // 1) The end instruction is a terminator IFF the ending basic block is different from the start;
        if (access->end->isTerminator()) {
            // 1.1) The end instruction is not a return statement: jump at the next BB;
            if (!isa<ReturnInst>(access->end)) {

                // Obtain the basic block where the end instruction is;
                BasicBlock *end_inst_block = access->end->getParent();
                // Obtain the basic block that follows the end instruction.
                // We take the first basic block that follows the current one, and is different from it;
                Instruction *terminatorI = access->end->getParent()->getTerminator();
                if (auto branchI = dyn_cast<BranchInst>(terminatorI)) {
                    for (auto successor_block : branchI->successors()) {
                        if (successor_block != end_inst_block) {
                            end_if_block = successor_block;
                            break;
                        }
                    }
                }
                if (end_if_block) {
                    // Note: splitting the edge guarantees, on paper, a unique place to end the loop.
                    // Based on the tests so far, this doesn't seem necessary however!
                    // SplitEdge(end_inst_block, end_if_block);
                } else {
                    outs() << "WARNING: skipping processing of access " << access_number << ", no next BasicBlock found!";
                    access_number++;
                    continue;
                }
            } else {
                // 1.2) If the last instruction is a return statement, jump right before it;
                end_if_block = SplitBlock(access->end->getParent(), access->end);
            }
        } else {
            // 2) Handle the standard situation where the start and end of the array access are in the same basic block.
            // In this case, split again the current basic block to create the area protected by the boundary check;

            // Get the instruction after the last instruction affected by the array access;
            Instruction *end_if_instruction = access->end->getNextNode();
            // Create a new block after the array access instruction.
            // This represents the end of the "if" statement;
            end_if_block = SplitBlock(end_if_instruction->getParent(), end_if_instruction);
        }
        if (end_if_block) {
            end_if_block->setName(formatv("end_array_access_{0}", access_number));
        }

        BasicBlock *else_block = BasicBlock::Create(context, formatv("else_block_array_access_{0}", access_number), end_if_block->getParent(), end_if_block);
        // Insert unconditional jump in basick block, as terminator: we always jump to the original end_array_access block;
        IRBuilder<> builder(else_block);
        builder.CreateBr(end_if_block);

        // Start a new block right before the first instruction of the array access.
        // This represents the start of the "if" statement;
        Instruction *start_if_instruction = access->start; // obtain_array_access_start(access);
        BasicBlock *start_if_block = SplitBlock(start_if_instruction->getParent(), start_if_instruction);
        start_if_block->setName(formatv("start_array_access_{0}", access_number));

        add_array_size_if_statement(access, start_if_block, else_block, protect_lower_bounds, positive_cuda_dependent_values);

        // Add instructions that track OOB accesses, printf and incremenent debug array;
        add_update_to_oob_tracking_array(context, access, else_block);

        ir_updated = true;
        access_number++;
    }
    return ir_updated;
}

}