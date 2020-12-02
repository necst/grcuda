#include "fix_oob_cuda_pass.hpp"

////////////////////////////////
////////////////////////////////

// Set of functions that express a CUDA-dependent positional index;
std::set<StringRef> CUDA_REF_EXPR = {
    "llvm.nvvm.read.ptx.sreg.ntid",
    "llvm.nvvm.read.ptx.sreg.ctaid",
    "llvm.nvvm.read.ptx.sreg.tid",
    "llvm.nvvm.read.ptx.sreg.nctaid",
    "llvm.nvvm.read.ptx.sreg.gridid"};

////////////////////////////////
////////////////////////////////

cl::opt<std::string> InputFile("input_file", cl::desc("Specify the filename of the file to be analyzed"), cl::value_desc("file_name"));
cl::opt<std::string> InputKernel("kernel", cl::desc("Specify the name of the CUDA kernel to be analyzed"), cl::value_desc("kernel_name"));
cl::opt<bool> NoInputSizes("no_input_sizes", cl::desc("If present, assume that the function signature does not contain the size of each input array"));
cl::opt<bool> SimplifyAccesses("simplify_accesses", cl::desc("If present, try to optimize nested boundary checks"));
cl::opt<bool> LowerBounds("lower_bounds", cl::desc("If present, add >= 0 boundary checks"));
cl::opt<bool> TestKernels("test", cl::desc("If present, apply the optimization pass to a few sample kernels and check if the output IR is valid"));
cl::opt<bool> Debug("debug", cl::desc("If present, print debug messages during the transformation pass"));
cl::opt<bool> DumpKernel("dump_updated_kernel", cl::desc("If present, print the updated kernel IR"));
cl::opt<int> Mode("oob_protection_type", cl::desc("Specify the type of protection: [0] Prevent OOB accesses [1] Track OOB accesses [2] Prevent and track OOB accesses"));
cl::opt<bool> PrintOOB("print_oob", cl::desc("If true, add debug print to kernels for OOB accesses, when tracking accesses"));

namespace llvm {

TestPass::TestPass() : FunctionPass(ID) {
    if (!TestKernels && InputKernel.getNumOccurrences() > 0) {
        kernel_name = InputKernel;
    } else {
        kernel_name = "";
    }
    // Define how this transformation pass should operate;
    protection_type = (Mode.getNumOccurrences() > 0 && Mode.getValue() < NUM_PROTECTION_TYPES) ? OOB_PROTECTION_TYPE(Mode.getValue()) : PREVENT;
    // Define the functions' implementations based on the operation mode;
    switch (protection_type)
    {
    case PREVENT:
        function_provider = new FixOOBFunctionProvider(Debug);
        break;
    case TRACK:
        function_provider = new TrackOOBFunctionProvider(Debug, PrintOOB);
        break;
    case PREVENT_AND_TRACK:
        function_provider = new TrackAndFixOOBFunctionProvider(Debug, PrintOOB);
        break;
    default:
        function_provider = new FixOOBFunctionProvider(Debug);
        break;
    }
}

bool TestPass::runOnFunction(Function &F) {
    // Keep track of IR modifications;
    bool ir_updated = false;

    // Obtain the original function name, without LLVM mangling;
    std::string demangled_name = demangle_function_name(F.getName().str().c_str());
    // Keep processing only the desired kernel;
    if (kernel_name.length() == 0 || kernel_name == demangled_name) {
        // The dominator tree is used to check if an instruction comes before another in the code.
        // We use it to identify if an array access is "included" into another;
        DT = new DominatorTree(F);
        DT->recalculate(F);

        // Process the argument list;
        if (!NoInputSizes) {
            bool sizes_array_found = function_provider->parse_argument_list(F, array_arguments);
            // If the sizes array was found, add instructions that load the array sizes.
            // If not, interrupt the pass;
            if (sizes_array_found) {
                function_provider->insert_load_sizes_instructions(F, array_arguments, input_array_sizes);
            } else {
                return ir_updated;
            }
        }

        // Iterate over the instructions to find CUDA-dependent instructions and array accesses;
        for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; I++) {

            // DEBUG PRINTING;
            if (Debug) {
                outs() << I->getOpcodeName() << ", opcode: " << I->getOpcode() << "\n";
            }

            // Check if the current instruction is storing a reference to an input array inside some other value.
            // If so, keep track of it;
            check_if_input_array_alias(*I);
            // Same thing, but with array sizes;
            // check_if_input_array_size_alias(*I);

            // Check if the current instruction depends, directly or indirectly,
            // from a CUDA-dependent index.
            // If it does, add it to the set of CUDA-dependent values;
            // handle_instruction_if_cuda_dependent(*I);
            // bool cuda_dependent = is_value_cuda_dependent(*I);
            check_if_positive_cuda_dependent(*I);

            // Handle array/pointer accesses: if the index used to access the array is CUDA-dependent,
            // keep track of the array access;
            // if (cuda_dependent) {
            function_provider->handle_array_access(*I, input_array_sizes, input_alias_map, DT, array_accesses);
            // }
        }

        if (Debug) {
            outs() << "\n\n------------------------------\n"
                   << "------- End of IR Pass -------\n"
                   << "------------------------------\n\n";
            for (auto a : positive_cuda_dependent_values) {
                outs() << "POSITIVE CUDA: ";
                a->print(outs());
                outs() << "\n";
            }
        }

        // Check if an array access already has a valid OoB protection.
        // Keep only the accesses without protection;
        function_provider->filter_accesses_with_protection(array_accesses, array_accesses_to_be_protected, input_alias_sizes_map);

        // Simplify the array access list by joining array accesses that are "included" into another access;
        if (SimplifyAccesses) {
            simplify_array_accesses();
        } else {
            array_accesses_postprocessed = array_accesses_to_be_protected;
        }

        // Do a pass over the array accesses, and add "if" statements to protect them;
        ir_updated |= function_provider->add_array_access_protection(F.getContext(), array_accesses_postprocessed, LowerBounds, positive_cuda_dependent_values);

        // DEBUG PRINTING;
        if (Debug) {
            outs() << "\n\tOriginal number of accesses: " << array_accesses.size() << "\n";
            outs() << "\tNumber of accesses to be protected: " << array_accesses_to_be_protected.size() << "\n";
            if (SimplifyAccesses) {
                outs() << "\tNumber of accesses after simplification: " << array_accesses_postprocessed.size() << "\n";
            }

            print_array_accesses(array_accesses_postprocessed);
        }

        // Remove the "optnone" function attribute to ensure that the function can be optimized;
        F.removeFnAttr(Attribute::OptimizeNone);
    }

    // DEBUG PRINTING of the updated kernel IR;
    if (ir_updated && DumpKernel) {
        outs() << "\n\n------------------------------\n"
               << "--------- Dumping IR ---------\n"
               << "------------------------------\n\n";
        F.print(outs());
        outs() << "\n";
    }

    return ir_updated;
}

////////////////////////////////
////////////////////////////////

bool TestPass::check_if_input_array_alias(Instruction &I) {
    bool added_alias = false;
    if (auto storeI = dyn_cast<StoreInst>(&I)) {
        Value *value = storeI->getValueOperand();
        Value *pointer = storeI->getPointerOperand();
        if (is_in_map(&input_array_sizes, value)) {
            input_alias_map[pointer] = value;
            added_alias = true;
        } else if (is_in_map(&input_alias_map, value)) {
            // Propagate aliases;
            input_alias_map[pointer] = input_alias_map[value];
            added_alias = true;
        }
    } else if (auto loadI = dyn_cast<LoadInst>(&I)) {
        // Also, we could load a value that represents an input array, or an alias.
        // That's also an alias;
        Value *pointer = loadI->getPointerOperand();
        if (is_in_map(&input_alias_map, pointer)) {
            input_alias_map[loadI] = input_alias_map[pointer];
        } else if (is_in_map(&input_array_sizes, pointer)) {
            input_alias_map[loadI] = pointer;
        }
    }
    return added_alias;
}

bool TestPass::check_if_input_array_size_alias(Instruction &I) {
    bool added_alias = false;
    if (auto storeI = dyn_cast<StoreInst>(&I)) {
        Value *value = storeI->getValueOperand();
        Value *pointer = storeI->getPointerOperand();
        if (is_in_set(&array_sizes_set, value)) {
            input_alias_sizes_map[pointer] = value;
            added_alias = true;
        } else if (is_in_map(&input_alias_sizes_map, value)) {
            // Propagate aliases;
            input_alias_sizes_map[pointer] = input_alias_sizes_map[value];
            added_alias = true;
        }
    } else if (auto loadI = dyn_cast<LoadInst>(&I)) {
        // Also, we could load a value that represents an input array, or an alias.
        // That's also an alias;
        Value *pointer = loadI->getPointerOperand();
        if (is_in_map(&input_alias_sizes_map, pointer)) {
            input_alias_sizes_map[loadI] = input_alias_sizes_map[pointer];
        } else if (is_in_set(&array_sizes_set, pointer)) {
            input_alias_sizes_map[loadI] = pointer;
        }
    }
    return added_alias;
}

////////////////////////////////
////////////////////////////////

bool TestPass::is_instruction_call_to_cuda_position(CallInst &I) {
    for (auto s : CUDA_REF_EXPR) {
        if (I.getCalledFunction()->getName().contains(s)) {
            return true;
        }
    }
    return false;
}

void TestPass::handle_instruction_if_cuda_dependent(Instruction &I) {
    // Check if we are accessing a CUDA-dependent index;
    if (auto *cI = dyn_cast<CallInst>(&I)) {
        if (is_instruction_call_to_cuda_position(*cI)) {
            cuda_dependent_values.insert(&I);

            // DEBUG PRINTING;
            if (Debug) {
                StringRef function_name = cI->getCalledFunction()->getName();
                outs() << "CUDA-dependent instruction, addr: " << cI->getOperand(0) << ", function called: " << function_name << "\n";
                outs() << "\t Instruction dump: ";
                cI->print(outs());
                outs() << "\n";
            }
        }
    }
    // Check if a CUDA-dependent value is stored in another value.
    // If so, add the current instruction to this set too.
    // Store instructions are handled separately as they don't have a "return" value;
    if (auto storeI = dyn_cast<StoreInst>(&I)) {
        if (is_value_cuda_dependent(*(storeI->getValueOperand()))) {
            cuda_dependent_values.insert(storeI->getPointerOperand());

            // DEBUG PRINTING;
            if (Debug) {
                outs() << "Found instruction that stores a CUDA-dependent value:\n\t";
                I.print(outs());
                outs() << "\n";
            }
        }
    } else {
        // Check if the instruction operands belong to the set of CUDA-dependent values.
        // If so, add the current instruction to this set too;
        for (auto operand = I.operands().begin(); operand != I.operands().end(); operand++) {
            if (is_value_cuda_dependent(*(operand->get()))) {
                cuda_dependent_values.insert(&I);

                // DEBUG PRINTING;
                if (Debug) {
                    outs() << "Found instruction that uses CUDA-dependent value:\n\t";
                    I.print(outs());
                    outs() << "\n";
                }

                // Don't inspect the other operands;
                break;
            }
        }
    }
}

// Check if the given instruction gives a positive result, starting
// from positive operands;
bool TestPass::instruction_has_positive_result(Instruction &I) {
    std::set<std::string> positive_opcodes = {
        "load", "add", "fadd", "mul", "fmul", "udiv", "sdiv", "fdiv", "urem", "srem", "frem"};
    return is_in_set(&positive_opcodes, std::string(I.getOpcodeName()));
}

// Check if the input value is a >= 0 constant int or float;
bool TestPass::is_positive_constant(Value *V) {
    if (auto constant = dyn_cast<ConstantInt>(V)) {
        if (constant->getSExtValue() > 0) {
            return true;
        }
    } else if (auto constant = dyn_cast<ConstantFP>(V)) {
        if (!constant->isNegative()) {
            return true;
        }
    }
    return false;
}

void TestPass::check_if_positive_cuda_dependent(Instruction &I) {
    bool positive_dependent_cuda_val = false;
    // Check if we are accessing a CUDA-dependent index;
    if (auto *cI = dyn_cast<CallInst>(&I)) {
        if (is_instruction_call_to_cuda_position(*cI)) {
            positive_cuda_dependent_values.insert(&I);
            positive_dependent_cuda_val = true;

            // DEBUG PRINTING;
            if (Debug) {
                StringRef function_name = cI->getCalledFunction()->getName();
                outs() << "CUDA-dependent instruction, addr: " << cI->getOperand(0) << ", function called: " << function_name << "\n";
                outs() << "\t Instruction dump: ";
                cI->print(outs());
                outs() << "\n";
            }
        }
    }
    // Check if a CUDA-dependent value is stored in another value.
    // If so, add the current instruction to this set too.
    // Store instructions are handled separately as they don't have a "return" value;
    if (auto storeI = dyn_cast<StoreInst>(&I)) {
        if (is_in_set(&positive_cuda_dependent_values, storeI->getValueOperand())) {
            positive_cuda_dependent_values.insert(storeI->getPointerOperand());
            positive_dependent_cuda_val = true;

            // DEBUG PRINTING;
            if (Debug) {
                outs() << "Found instruction that stores a positive CUDA-dependent value:\n\t";
                I.print(outs());
                outs() << "\n";
            }
        }
        // Check if we are storing a constant positive value;
        else if (is_positive_constant(storeI->getValueOperand())) {
            positive_cuda_dependent_values.insert(storeI->getPointerOperand());
            positive_dependent_cuda_val = true;
            // DEBUG PRINTING;
            if (Debug) {
                outs() << "Found instruction that stores a positive CONSTANT value:\n\t";
                I.print(outs());
                outs() << "\n";
            }
        }
    } else {
        // Check if all operands are positive-CUDA-dep, and instruction is among sum/prod/etc
        bool all_operands_cuda_dep = true;
        for (auto operand = I.operands().begin(); operand != I.operands().end(); operand++) {
            if (!is_in_set(&positive_cuda_dependent_values, operand->get()) && !is_positive_constant(operand->get())) {
                all_operands_cuda_dep = false;
            }
        }
        if (!instruction_has_positive_result(I)) {
            all_operands_cuda_dep = false;
        }
        if (all_operands_cuda_dep) {
            positive_cuda_dependent_values.insert(&I);
            positive_dependent_cuda_val = true;

            // DEBUG PRINTING;
            if (Debug) {
                outs() << "Found instruction that uses positive CUDA-dependent value:\n\t";
                I.print(outs());
                outs() << "\n";
            }
        }
    }
    // Keep track if instruction was CUDA dep. If not, check if it belongs to the set: if so, remove it.
    // If I == store, look at pointer operand;
    if (!positive_dependent_cuda_val) {
        if (Value *v = dyn_cast<Value>(&I)) {
            if (is_in_set(&positive_cuda_dependent_values, v)) {
                positive_cuda_dependent_values.erase(v);
            }
        }
        if (auto storeI = dyn_cast<StoreInst>(&I)) {
            if (is_in_set(&positive_cuda_dependent_values, storeI->getPointerOperand())) {
                positive_cuda_dependent_values.erase(storeI->getPointerOperand());
            }
        }
    }
}

////////////////////////////////
////////////////////////////////

// Check if x is inside y;
bool TestPass::inside_access(ArrayAccess *x, ArrayAccess *y) {
    // Check if the y.start is <= x.start;
    bool y_start_larger = y->start == x->start || OrderedInstructions(DT).dfsBefore(y->start, x->start);
    bool y_end_larger = y->end == x->end || OrderedInstructions(DT).dfsBefore(x->end, y->end);

    bool inside = y_start_larger && y_end_larger;

    // If the index computation of x is after the start of y,
    // we cannot merge the boundary access of x with y (as we wouldn't have the value required for the check);
    if (inside) {
        for (auto boundary_access : x->extended_cmp_inst_for_size_check) {
            if (OrderedInstructions(DT).dfsBefore(y->start, cast<Instruction>(boundary_access.first))) {
                inside = false;
                break;
            }

            // Check if among the users of y there is a store on the value used to index x,
            // and the store is before the start of x. No merging can be done in this case,
            // as x needs the updated index;
            // Queue used to explore the instruction users in BFS fashion;
            std::queue<Instruction *> users_queue;
            // Set used to avoid processing the same instruction twice;
            std::set<Instruction *> seen_instructions;
            Instruction *x_index_storage = cast<Instruction>(boundary_access.first);
            for (auto user : y->start->users()) {
                if (auto I = dyn_cast<Instruction>(user)) {
                    users_queue.push(I);
                    seen_instructions.insert(I);
                }
            }
            while (!users_queue.empty()) {
                Value *currI = users_queue.front();
                users_queue.pop();

                // Check if the current instruction is a store on the same value as the index of x;
                if (auto storeI = dyn_cast<StoreInst>(currI)) {
                    if (storeI->getPointerOperand() == x_index_storage && OrderedInstructions(DT).dfsBefore(storeI, x->start)) {
                        inside = false;
                        break;
                    }
                }

                // Add the users of the current instruction to the queue;
                for (auto child : currI->users()) {
                    if (auto childI = dyn_cast<Instruction>(child)) {
                        if (!is_in_set(&seen_instructions, childI) && OrderedInstructions(DT).dfsBefore(childI, x->start)) {
                            users_queue.push(childI);
                            seen_instructions.insert(childI);
                        }
                    }
                }
            }
        }
    }

    return inside;
}

void TestPass::simplify_array_accesses() {

    std::set<uint> added_accesses{};
    std::set<uint> merged_accesses{};

    for (uint i = 0; i < array_accesses_to_be_protected.size(); i++) {
        // Never merge array accesses where the index refers to another pointer expression,
        // to avoid having to add "recursive" boundary checks.
        // For example x[y[index]] would require this simplified check: index < y_size && y[index] < x_size,
        // which would require the addition of another index < y_size (we are not doing shortcircuiting);
        if (auto *loadI = dyn_cast<LoadInst>(array_accesses_to_be_protected[i]->index_expression)) {
            if (isa<GetElementPtrInst>(loadI->getPointerOperand())) {
                continue;
            }
        }
        bool add_access = true;
        for (uint j = 0; j < array_accesses_to_be_protected.size(); j++) {
            if (i != j && inside_access(array_accesses_to_be_protected[i], array_accesses_to_be_protected[j])) {
                add_access = false;
                // When dealing with optimized LLVM code, the start and end of an array access might be identical:
                // in this case, add the first access and merge the second with the first.
                // Obviously, add only one access if > 2 accesses have the same start & end;
                if (i < j &&
                    array_accesses_to_be_protected[j]->start == array_accesses_to_be_protected[i]->start &&
                    array_accesses_to_be_protected[j]->end == array_accesses_to_be_protected[i]->end &&
                    !is_in_set(&added_accesses, i) &&
                    !is_in_set(&merged_accesses, i)) {
                    add_access = true;
                    merged_accesses.insert(j);
                }

                // Add to the extended comparison list the information regarding the "included" access;
                for (auto access_to_insert : array_accesses_to_be_protected[i]->extended_cmp_inst_for_size_check) {
                    // Avoid the insertion of duplicate boundary checks;
                    bool unique_access = true;
                    for (auto existing_access : array_accesses_to_be_protected[j]->extended_cmp_inst_for_size_check) {
                        if (existing_access.first == access_to_insert.first && existing_access.second == access_to_insert.second) {
                            unique_access = false;
                        }
                    }
                    if (unique_access) {
                        array_accesses_to_be_protected[j]->extended_cmp_inst_for_size_check.push_back(access_to_insert);
                    }
                }
            }
        }
        if (add_access) {
            array_accesses_postprocessed.push_back(array_accesses_to_be_protected[i]);
            added_accesses.insert(i);
        }
    }
}

////////////////////////////////
////////////////////////////////

/*
Instruction *TestPass::obtain_array_access_start(ArrayAccess *access) {
    Instruction *current_start = access->start;
    GetElementPtrInst *gepI = access->get_array_value;

    // If the pointer computation and the array start are in the same BB, keep the current start;
    if (current_start->getParent() == gepI->getParent()) {
        return current_start;
    } else {
        // If not, we need to find a BB where we are sure that both the index computation and the actual
        // array access will be performed,
        // otherwise we could reach the array access without performing the index computation.
        // This can happen if we have already split an existing BB and the index computation falls
        // in a "boundary-check" protected BB;

        // Obtain the set of BB that follows the start BB, not including children BB of the array access BB;
        BasicBlock *current_startBB = current_start->getParent();
        BasicBlock *gepBB = gepI->getParent();
        std::set<BasicBlock *> follower_set;
        follower_set.insert(gepBB);
        std::queue<BasicBlock *> follower_queue;
        follower_queue.push(current_startBB);

        while (!follower_queue.empty()) {
            BasicBlock *temp_BB = follower_queue.front();
            follower_queue.pop();
            follower_set.insert(temp_BB);

            for (auto f : successors(temp_BB)) {
                if (!is_in_set(&follower_set, f)) {
                    follower_queue.push(f);
                }
            }
        }

        // Visit all predecessors of the BB where the index is computed. If they are followers of the BB
        // where the current start is, simply add their predecessors and continue the visit.
        // If we find a predecessor BB which is not a follower of the start BB, check if the start BB is reachable from it.
        // If so, we can use this BB as new start and stop the search;
        std::queue<BasicBlock *> pred_queue;
        std::set<BasicBlock *> visited_pred_set;
        visited_pred_set.insert(gepBB);
        for (auto p : predecessors(gepBB)) {
            pred_queue.push(p);
            visited_pred_set.insert(p);
        }
        while (!pred_queue.empty()) {
            BasicBlock *temp_BB = pred_queue.front();
            pred_queue.pop();

            // If the current predecessor is not a follower of the start BB, check if it covers it;
            if (!is_in_set(&follower_set, temp_BB)) {
                bool start_found = check_if_bb_follows_another(temp_BB, current_startBB);
                if (start_found) {
                    // Return the new start, i.e. the end of this BB;
                    return temp_BB->getTerminator();
                }
            }

            for (auto p : predecessors(temp_BB)) {
                if (!is_in_set(&visited_pred_set, p)) {
                    pred_queue.push(p);
                    visited_pred_set.insert(p);
                }
            }
        }

        // Fallback to the existing start;
        return current_start;
    }
}

// Return true if BB "second" follows "first", by doing a DFS from "first";
bool TestPass::check_if_bb_follows_another(BasicBlock *first, BasicBlock *second) {
    std::set<BasicBlock *> follower_set;
    follower_set.insert(first);
    std::queue<BasicBlock *> follower_queue;
    follower_queue.push(first);

    while (!follower_queue.empty()) {
        BasicBlock *follower_BB = follower_queue.front();
        follower_queue.pop();

        if (follower_BB == second) {
            return true;
        }
        for (auto f : successors(follower_BB)) {
            if (!is_in_set(&follower_set, f)) {
                follower_queue.push(f);
                follower_set.insert(f);
            }
        }
    }
    return false;
}

*/

////////////////////////////////
////////////////////////////////

void TestPass::print_array_accesses(std::vector<ArrayAccess *> &array_accesses) {
    outs() << "CUDA-dependent array accesses: " << array_accesses.size() << "\n";
    for (uint i = 0; i < array_accesses.size(); i++) {

        outs() << i << ") access types: \n  ";
        for (auto t : array_accesses[i]->access_type) {
            if (t == ArrayAccessType::LOAD) {
                outs() << "load\n";
            } else if (t == ArrayAccessType::STORE) {
                outs() << "store\n";
            } else if (t == ArrayAccessType::CALL) {
                outs() << "call\n";
            }
        }

        array_accesses[i]->array_load->print(outs());
        outs() << "\n  ";
        array_accesses[i]->index_expression->print(outs());

        outs() << "\n  ";
        if (array_accesses[i]->index_casting) {
            array_accesses[i]->index_casting->print(outs());
            outs() << "\n  ";
        } else {
            outs() << "no index casting\n  ";
        }
        array_accesses[i]->get_array_value->print(outs());

        outs() << "\n  accesses:\n    ";
        for (auto a : array_accesses[i]->array_access) {
            a->print(outs());
            outs() << "\n    ";
        }

        outs() << "\n    start: ";
        array_accesses[i]->start->print(outs());

        outs() << "\n    end: ";
        array_accesses[i]->end->print(outs());

        outs() << "\n    size: ";
        array_accesses[i]->array_size->print(outs());

        outs() << "\n    requires protection: " << (array_accesses[i]->requires_protection ? "true" : "false") << "\n";

        outs() << "\n    list of accesses: ";
        for (auto access : array_accesses[i]->extended_cmp_inst_for_size_check) {
            outs() << "\n        index: ";
            access.first->print(outs());
            outs() << "\n        size: ";
            access.second->print(outs());
        }

        outs() << "\n\n";
    }
}

char TestPass::ID = 0;

static RegisterPass<TestPass> PassRegistration("fix_oob_cuda_pass", "CUDA - Fix OoB Array Accesses",
                                               false /* Only looks at CFG */,
                                               false /* Analysis Pass */);

} // namespace llvm