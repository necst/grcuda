#include "fix_oob_function_provider.hpp"

namespace llvm {

bool FixOOBFunctionProvider::parse_argument_list(Function &F, std::vector<Value *> &array_arguments) {
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
    if (array_arguments.size() == 1) {
        if (debug) {
            outs() << "WARNING: only " << array_arguments.size() << " pointer argument(s) found!\n";
        }
        return false;
    }
    sizes_array = array_arguments.back();
    array_arguments.pop_back();

    return true;
}

////////////////////////////////
////////////////////////////////

void FixOOBFunctionProvider::insert_load_sizes_instructions(Function &F, std::vector<Value *> &array_arguments, std::map<Value *, Value *> &input_array_sizes) {
    // Obtain a reference to the start of the function, that's were we add new instructions;
    Instruction &first_inst = F.getBasicBlockList().front().front();
    IRBuilder<> builder(&first_inst);

    // Load the value of each array size;
    for (uint i = 0; i < array_arguments.size(); i++) {
        // Compute the pointer value;
        Value *ptrI = builder.CreateGEP(sizes_array, ConstantInt::get(IntegerType::getInt64Ty(F.getContext()), i),
                                        formatv("compute_ptr_size_{0}", i));
        // Add a load instruction;
        Value *array_size = builder.CreateLoad(ptrI, formatv("array_size_{0}", i));
        // Associate the array size with the corresponding array;
        input_array_sizes[array_arguments[i]] = array_size;
        if (debug) {
            outs() << "Loading size of array " << i << "\n\t";
            ptrI->print(outs());
            outs() << "\n\t";
            array_size->print(outs());
            outs() << "\n";
        }
    }
}

////////////////////////////////
////////////////////////////////

bool FixOOBFunctionProvider::handle_array_access(
    Instruction &I,
    std::map<Value *, Value *> &input_array_sizes,
    std::map<Value *, Value *> &input_alias_map,
    DominatorTree *DT,
    std::vector<ArrayAccess *> &array_accesses
    ) {

    bool array_access_found = false;

    if (auto getI = dyn_cast<GetElementPtrInst>(&I)) {

        // Ignore accesses with all constant indices;
        if (getI->hasAllConstantIndices()) {
            return false;
        }

        // Create a data structure to store array access data;
        ArrayAccess *access = new ArrayAccess();
        access->get_array_value = getI;
        access->array_load = getI->getPointerOperand();

        // DEBUG PRINTING;
        if (debug) {
            outs() << "pointer operand: ";
            getI->getPointerOperand()->print(outs());
            outs() << "\n";
        }
        // Look for the index used to access the array;
        for (auto index = getI->indices().begin(); index != getI->indices().end(); index++) {
            // If the index was obtained with a cast, retrieve the original index;
            if (auto castI = dyn_cast<CastInst>(index->get())) {
                access->index_casting = castI;

                // DEBUG PRINTING;
                if (debug) {
                    outs() << "\t\tindex: ";
                    index->get()->print(outs());
                    outs() << "\n";
                }
                // Retrieve the original index load, before casting;
                auto index_expression = dyn_cast<Instruction>(&*(castI->getOperand(0)));
                if (index_expression) { // && is_value_cuda_dependent(*index_expression)) {
                    access->index_expression = index_expression;

                    // DEBUG PRINTING;
                    if (debug) {
                        outs() << "\t\toriginal index: ";
                        access->index_expression->print(outs());
                        outs() << "\n";
                    }
                }
            } else if (auto index_expression = dyn_cast<Instruction>(index->get())) {
                access->index_expression = index_expression;
            }
        }

        // Obtain the array real size if the array is given as input;
        // If the array is allocated with a fixed size, we can obtain the size at compile time;
        if (auto arrayT = dyn_cast<ArrayType>(getI->getSourceElementType())) {
            access->array_size = ConstantInt::get(access->index_expression->getType(), arrayT->getNumElements());
        } else if (is_in_map(&input_array_sizes, getI->getPointerOperand())) {
            // If the accessed pointer is an input argument, use its size;
            access->array_size = input_array_sizes[getI->getPointerOperand()];
        } else if (is_in_map(&input_alias_map, getI->getPointerOperand()) &&
                   is_in_map(&input_array_sizes, input_alias_map[getI->getPointerOperand()])) {
            // The accessed pointer is an alias for an input argument;
            access->array_size = input_array_sizes[input_alias_map[getI->getPointerOperand()]];
        } else {
            // Use an undefined value;
            access->array_size = UndefValue::get(access->index_expression->getType());
            outs() << "WARNING: detected array access to array with undefined size!\n";
        }

        // Find the start of the array access (either the index casting or the address computation);
        // if (access->index_casting) {
        //     access->start = access->index_casting;
        // } else {
        //     access->start = access->get_array_value;
        // }
        access->start = access->get_array_value;

        // Obtain the load/store instruction associated to the array access.
        // Do this by looking at the set of users of the GEP instruction;
        std::queue<Value *> user_queue;
        user_queue.push(getI);
        while (!user_queue.empty()) {
            Value *currI = user_queue.front();
            user_queue.pop();

            // If the current instruction is a load/store stop the search,
            // else add its users.
            // Consider also function calls that use a pointer as input argument;
            if (auto loadI = dyn_cast<LoadInst>(currI)) {
                access->access_type.push_back(ArrayAccessType::LOAD);
                access->array_access.push_back(loadI);
            } else if (auto storeI = dyn_cast<StoreInst>(currI)) {
                access->access_type.push_back(ArrayAccessType::STORE);
                access->array_access.push_back(storeI);
            } else if (auto callI = dyn_cast<CallInst>(currI)) {
                access->access_type.push_back(ArrayAccessType::CALL);
                access->array_access.push_back(callI);
            } else {
                for (auto child : currI->users()) {
                    user_queue.push(child);
                }
            }
        }
        if (access->array_access.size() == 0) {
            outs() << "WARNING: No load/store/call instruction found for array access, instruction: ";
            getI->print(outs());
            outs() << "\n";
        }
        // Initialize the extended list of boundary check inspections.
        // It contains only a tuple with the array size and the index used in the access.
        // If the index is represented by a "load" instruction, store the pointer instead:
        //   this is useful to check if multiple boundary checks are using the same index;
        Value *updated_index_expression = access->index_expression;
        if (auto *loadI = dyn_cast<LoadInst>(access->index_expression)) {
            updated_index_expression = loadI->getPointerOperand();
        }
        access->extended_cmp_inst_for_size_check =
            std::vector<std::pair<Value *, Value *>>{
                std::pair<Value *, Value *>(updated_index_expression, access->array_size)};

        // Compute the last instruction "touched" by the array access, by inspecting all the leaf load/store instructions and finding a common basic block to all of them;
        if (access->array_access.size() > 0) {
            access->end = find_array_access_end(access, DT, getI);
            array_accesses.push_back(access);
        }
    }
    return array_access_found;
} 

////////////////////////////////
////////////////////////////////

Instruction* FixOOBFunctionProvider::find_array_access_end(ArrayAccess *access, DominatorTree *DT, GetElementPtrInst *getI) {
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

void FixOOBFunctionProvider::filter_accesses_with_protection(
    std::vector<ArrayAccess *> &array_accesses,
    std::vector<ArrayAccess *> &array_accesses_to_be_protected,
    std::map<Value *, Value *> &input_alias_sizes_map
    ) {
    for (auto access : array_accesses) {
        bool requires_protection = true;

        // Look at the list of predecessors Basic Blocks, and check if any of them ends with an "if".
        // We cannot look just at the immediate predecessor, as we might have nested "ifs".
        // We process the BB with a queue;
        std::queue<BasicBlock *> predecessors_queue;
        std::set<BasicBlock *> seen_blocks;
        // Initialize the queue;
        for (auto predBB : predecessors(access->start->getParent())) {
            predecessors_queue.push(predBB);
        }
        while (!predecessors_queue.empty() && requires_protection) {
            BasicBlock *BB = predecessors_queue.front();
            predecessors_queue.pop();
            seen_blocks.insert(BB);

            // Check if the BB ends with a conditional branch, whose condition is a compare instruction;
            if (auto branchI = dyn_cast<BranchInst>(BB->getTerminator())) {
                if (branchI->isConditional()) {

                    // If the condition is a comparison, process it directly.
                    // If it's an AND, look for a list of ICMP from which the AND was built;
                    std::vector<ICmpInst *> comparison_list;
                    if (auto ifI = dyn_cast<ICmpInst>(branchI->getCondition())) {
                        comparison_list.push_back(ifI);
                    } else if (auto andI = dyn_cast<BinaryOperator>(branchI->getCondition())) {
                        // If it's an AND;
                        if (std::string(andI->getOpcodeName()) == "and") {
                            std::queue<Value *> icmp_queue;
                            icmp_queue.push(andI->getOperand(0));
                            icmp_queue.push(andI->getOperand(1));
                            while (!icmp_queue.empty()) {
                                Value *tempI = icmp_queue.front();
                                icmp_queue.pop();
                                // Add ICMP to the list of comparisons to be processed below;
                                if (auto icmpI = dyn_cast<ICmpInst>(tempI)) {
                                    comparison_list.push_back(icmpI);
                                } else if (auto andI2 = dyn_cast<BinaryOperator>(tempI)) {
                                    // If the instruction is an AND, add the operands;
                                    if (std::string(andI2->getOpcodeName()) == "and") {
                                        icmp_queue.push(andI2->getOperand(0));
                                        icmp_queue.push(andI2->getOperand(1));
                                    }
                                }
                            }
                        }
                    }

                    for (auto ifI : comparison_list) {

                        // Check if the LHS and RHS of the compare instruction are the same of our boundary check.
                        Value *index_expr_aa = access->extended_cmp_inst_for_size_check[0].first;
                        Value *array_size_aa = access->extended_cmp_inst_for_size_check[0].second;
                        Value *LHS = ifI->getOperand(0);
                        Value *RHS = ifI->getOperand(1);

                        bool same_index = check_if_same_index(index_expr_aa, LHS);

                        // Compare also the array sizes, using aliases if necessary;
                        bool same_size = false;
                        if (same_index) {
                            same_size = check_if_same_size(array_size_aa, RHS, input_alias_sizes_map);
                        }

                        // Check if the termination of the existing check is after the "span" of the array access;
                        bool included_end = false;
                        if (same_index && same_size) {

                            BasicBlock *destination = dyn_cast<BasicBlock>(branchI->getSuccessor(1));
                            if (destination) {
                                // Check if the BB of the access end is a predecessor of the destination block;
                                std::queue<BasicBlock *> temp_pred_queue;
                                std::set<BasicBlock *> temp_seen_blocks;
                                for (auto p : predecessors(destination)) {
                                    temp_pred_queue.push(p);
                                }
                                while (!temp_pred_queue.empty()) {
                                    BasicBlock *temp_pred = temp_pred_queue.front();
                                    temp_pred_queue.pop();
                                    temp_seen_blocks.insert(temp_pred);
                                    if (temp_pred == access->end->getParent()) {
                                        included_end = true;
                                        break;
                                    } else {
                                        for (auto p : predecessors(temp_pred)) {
                                            if (!is_in_set(&temp_seen_blocks, p)) {
                                                temp_pred_queue.push(p);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (same_index && same_size && included_end) {
                            if (debug) {
                                outs() << "found array access with exising protection: ";
                                access->get_array_value->print(outs());
                                outs() << "\n";
                            }
                            requires_protection = false;
                        }
                    }
                }
            }

            // Add to the queue the predecessors of the current BB;
            if (requires_protection) {
                for (auto predBB : predecessors(BB)) {
                    if (!is_in_set(&seen_blocks, predBB)) {
                        predecessors_queue.push(predBB);
                    }
                }
            }
        }

        if (requires_protection) {
            array_accesses_to_be_protected.push_back(access);
        } else {
            access->requires_protection = false;
        }
    }
}


////////////////////////////////
////////////////////////////////

bool FixOOBFunctionProvider::add_array_access_protection(
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

        // Start a new block right before the first instruction of the array access.
        // This represents the start of the "if" statement;
        Instruction *start_if_instruction = access->start; // obtain_array_access_start(access);
        BasicBlock *start_if_block = SplitBlock(start_if_instruction->getParent(), start_if_instruction);
        start_if_block->setName(formatv("start_array_access_{0}", access_number));

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

        ir_updated = true;
        access_number++;
    }
    return ir_updated;
}

////////////////////////////////
////////////////////////////////

bool FixOOBFunctionProvider::check_if_same_size(Value *x, Value *y, std::map<Value *, Value *> &input_alias_sizes_map) {
    bool same_size = x == y;
    if (!same_size) {
        Value *x_orig = is_in_map(&input_alias_sizes_map, x) ? input_alias_sizes_map[x] : x;
        Value *y_orig = is_in_map(&input_alias_sizes_map, y) ? input_alias_sizes_map[y] : y;
        same_size = x_orig == y_orig;
    }
    return same_size;
}

}