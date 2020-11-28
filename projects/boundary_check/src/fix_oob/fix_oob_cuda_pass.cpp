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

namespace llvm {

TestPass::TestPass() : FunctionPass(ID) {
    if (!TestKernels && InputKernel.getNumOccurrences() > 0) {
        kernel_name = InputKernel;
    } else {
        kernel_name = "";
    }
    // Define how this transformation pass should operate;
    protection_type = (Mode.getNumOccurrences() > 0 && Mode.getValue() < NUM_PROTECTION_TYPES) ? (OOB_PROTECTION_TYPE) Mode.getValue() : PREVENT;
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
            bool sizes_array_found = parse_argument_list(F);
            // If the sizes array was found, add instructions that load the array sizes.
            // If not, interrupt the pass;
            if (sizes_array_found) {
                insert_load_sizes_instructions(F);
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
            handle_array_access(*I);
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
        filter_accesses_with_protection();

        // Simplify the array access list by joining array accesses that are "included" into another access;
        if (SimplifyAccesses) {
            simplify_array_accesses();
        } else {
            array_accesses_postprocessed = array_accesses_to_be_protected;
        }

        // Do a pass over the array accesses, and add "if" statements to protect them;
        ir_updated |= add_array_access_protection(F.getContext());

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

bool TestPass::parse_argument_list(Function &F) {
    // Process the function arguments to obtain a list of arrays and sizes;
    std::vector<Argument *> args;
    for (auto &arg : F.args()) {
        if (Debug) {
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
        if (Debug) {
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

void TestPass::insert_load_sizes_instructions(Function &F) {
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
        if (Debug) {
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

bool TestPass::handle_array_access(Instruction &I) {

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
        if (Debug) {
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
                if (Debug) {
                    outs() << "\t\tindex: ";
                    index->get()->print(outs());
                    outs() << "\n";
                }
                // Retrieve the original index load, before casting;
                auto index_expression = dyn_cast<Instruction>(&*(castI->getOperand(0)));
                if (index_expression) { // && is_value_cuda_dependent(*index_expression)) {
                    access->index_expression = index_expression;

                    // DEBUG PRINTING;
                    if (Debug) {
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

        // Compute the last instruction "touched" by the array access;
        if (access->array_access.size() > 0) {

            // Queue used to explore the instruction users in BFS fashion;
            std::queue<Instruction *> end_array_queue;
            // Set used to avoid processing the same instruction twice;
            std::set<Instruction *> seen_instructions;

            // Add to the queue the index value used to access the array,
            // as it might be used somewhere else;
            if (access->index_casting) {
                end_array_queue.push(access->index_casting);
            }

            Instruction *lastI = access->array_access[0];
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
            access->end = lastI;

            array_accesses.push_back(access);
        }
    }
    return array_access_found;
} // namespace llvm

////////////////////////////////
////////////////////////////////

bool TestPass::check_if_same_index(Value *x, Value *y) {
    bool same_index = x == y;
    // If the index refers to a load instruction, check the original values;
    if (!same_index) {
        Value *x_orig = x;
        Value *y_orig = y;
        if (auto loadI1 = dyn_cast<LoadInst>(x)) {
            x_orig = loadI1->getPointerOperand();
        }
        if (auto loadI2 = dyn_cast<LoadInst>(y)) {
            y_orig = loadI2->getPointerOperand();
        }
        same_index = x_orig == y_orig;
    }
    return same_index;
}

bool TestPass::check_if_same_size(Value *x, Value *y) {
    bool same_size = x == y;
    if (!same_size) {
        Value *x_orig = is_in_map(&input_alias_sizes_map, x) ? input_alias_sizes_map[x] : x;
        Value *y_orig = is_in_map(&input_alias_sizes_map, y) ? input_alias_sizes_map[y] : y;
        same_size = x_orig == y_orig;
    }
    return same_size;
}

void TestPass::filter_accesses_with_protection() {
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
                            same_size = check_if_same_size(array_size_aa, RHS);
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
                            if (Debug) {
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

bool TestPass::add_array_access_protection(LLVMContext &context) {

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
            if (LowerBounds && !is_in_set(&indices_set, size_check.first) && !is_in_set(&positive_cuda_dependent_values, size_check.first)) {
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