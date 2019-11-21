#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"
#include "cxxabi.h"
#include <string>
#include <iostream>

#include "../utils.hpp"

////////////////////////////////
////////////////////////////////

using namespace llvm;
using std::cout;
using std::endl;

////////////////////////////////
////////////////////////////////

cl::opt<std::string> InputFile("input_file", cl::desc("Specify the filename of the file to be analyzed"), cl::value_desc("file_name"));
cl::opt<std::string> InputKernel("kernel", cl::desc("Specify the name of the CUDA kernel to be analyzed"), cl::value_desc("kernel_name"));
cl::opt<bool> TestKernels("test", cl::desc("If present, apply the optimization pass to a few sample kernels and check if the output IR is valid"));
cl::opt<bool> Debug("debug", cl::desc("If present, print debug messages during the transformation pass"));
cl::opt<bool> DumpKernel("dump_updated_kernel", cl::desc("If present, print the updated kernel IR"));

////////////////////////////////
////////////////////////////////

std::string examples_folder = "examples/truffle_kernels/llvm/original/";

std::vector<std::pair<std::string, std::string>> example_kernels = {
    std::pair<std::string, std::string>("axpy-O0.ll", "axpy"),
    std::pair<std::string, std::string>("axpy-O1.ll", "axpy"),
    std::pair<std::string, std::string>("dot_product-O0.ll", "dot_product"),
    std::pair<std::string, std::string>("dot_product-O1.ll", "dot_product"),
    std::pair<std::string, std::string>("convolution-O0.ll", "convolution"),
    std::pair<std::string, std::string>("convolution-O1.ll", "convolution"),
    std::pair<std::string, std::string>("hotspot-O0.ll", "calculate_temp"),
    std::pair<std::string, std::string>("hotspot-O1.ll", "calculate_temp")};

////////////////////////////////
////////////////////////////////

struct CudaAddSizesPass : public ModulePass {
    CudaAddSizesPass() : ModulePass(ID) {
        if (!TestKernels && InputKernel.getNumOccurrences() > 0) {
            kernel_name = InputKernel;
        } else {
            kernel_name = "";
        }
    }

    ////////////////////////////////
    ////////////////////////////////

    virtual bool runOnModule(Module &M) {

        // Keep track of IR modifications;
        bool ir_updated = false;

        // First, we need to find the desired CUDA kernel;
        for (auto &F : M.getFunctionList()) {
            // Obtain the original function name, without LLVM mangling;
            std::string demangled_name = demangle_function_name(F.getName().str().c_str());
            // Keep processing only the desired kernel;
            if (kernel_name.length() == 0 || kernel_name == demangled_name) {
                original_kernel = &F;
                break;
            }
        }

        // If the desired kernel was found, start processing it;
        if (original_kernel) {
            // Obtain the signature of the original kernel;
            FunctionType *original_signature = original_kernel->getFunctionType();

            // Copy the argument types;
            std::vector<Type *> params(original_signature->param_begin(), original_signature->param_end());

            // Look at each argument of the kernel. If any is a pointer, or an array,
            // add a pointer argument at the end, it will be an array containing sizes;
            for (auto &p : original_kernel->args()) {
                if (p.getType()->isArrayTy() || p.getType()->isPointerTy()) {
                    params.push_back(Type::getInt64PtrTy(original_kernel->getContext()));

                    ir_updated = true;
                    break;
                }
            }

            // We need to preserve the kernel entrypoint. For now, obtain a reference to the metadata where
            // the original kernel is listed as entrypoint;
            NamedMDNode *nvvm_metadata = M.getNamedMetadata("nvvm.annotations");
            // Iterate the nvvm annotations;
            for (auto op : nvvm_metadata->operands()) {
                // Iterate the operands of each annotation;
                int op2_num = 0;
                for (auto &op2 : op->operands()) {
                    if (op2) {
                        // Look for metadata whose operand is the original kernel;
                        if (auto mdnode = dyn_cast<ConstantAsMetadata>(op2)) {
                            if (auto temp_function = dyn_cast<Function>(mdnode->getValue())) {
                                if (temp_function == original_kernel) {
                                    metadata_to_update[op].push_back(op2_num);
                                }
                            }
                        }
                    }
                    op2_num++;
                }
            }

            // Stop processing the kernel if no pointer argument was found;
            if (ir_updated) {

                // Note: if using a struct to pass sizes, the struct must have the byval attribute,
                // so that it's copied on the stack. See http://lists.llvm.org/pipermail/llvm-dev/2008-September/016990.html
                // Do it with: original_kernel->getAttributes().addParamAttribute(original_kernel->getContext(), params.size(), llvm::Attribute::ByVal);
                // "params.size()" is used because we add the attribute to the new parameter we have created,
                // in position params.size();

                // The new return type is the same as before;
                Type *return_type = original_kernel->getReturnType();

                // Create the new function signature;
                FunctionType *new_kernel_signature = FunctionType::get(return_type, params, original_kernel->isVarArg());

                // Create the new function body and insert it into the module;
                new_kernel = Function::Create(new_kernel_signature, original_kernel->getLinkage(), original_kernel->getAddressSpace());
                // Keep the same function attributes;
                new_kernel->copyAttributesFrom(original_kernel);
                new_kernel->setComdat(original_kernel->getComdat());
                // Insert the function where the previous one was;
                original_kernel->getParent()->getFunctionList().insert(original_kernel->getIterator(), new_kernel);
                // Copy the name of the existing function;
                new_kernel->takeName(original_kernel);

                // Replace calls to the original function with calls to the new function.
                // In theory, this step is not required, as we process modules containing a single CUDA kernel.
                // We still do this step to guarantee that we obtain valid LLVM code.
                // The additional size parameters are passed with value equal to 0;
                update_callers(*original_signature);

                // Since we have now created the new function, splice the body of the old
                // function right into the new function, leaving the old function empty.
                new_kernel->getBasicBlockList().splice(new_kernel->begin(), original_kernel->getBasicBlockList());

                // Loop over the argument list, transferring uses of the old arguments over to
                // the new arguments, also transferring over the names as well.
                for (Function::arg_iterator oldI = original_kernel->arg_begin(), E = original_kernel->arg_end(),
                                            newI = new_kernel->arg_begin();
                     oldI != E; ++oldI, ++newI) {
                    // Move the name and users over to the new version.
                    oldI->replaceAllUsesWith(&*newI);
                    newI->takeName(&*oldI);
                }

                // Small optimizaztion: specify the sizes array to be read only, and "no capture",
                // i.e. no copies of this pointer are done in the function and outlive the function call;
                auto new_attr = new_kernel->getAttributes().addParamAttribute(new_kernel->getContext(), params.size() - 1, llvm::Attribute::ReadOnly);
                new_kernel->setAttributes(new_attr);
                new_attr = new_kernel->getAttributes().addParamAttribute(new_kernel->getContext(), params.size() - 1, llvm::Attribute::NoCapture);
                new_kernel->setAttributes(new_attr);

                // Cleanup steps, taken from: https://llvm.org/doxygen/DeadArgumentElimination_8cpp_source.html
                cleanup();

                // Delete the old function
                original_kernel->eraseFromParent();

                // Remove the "optnone" function attribute to ensure that the function can be optimized;
                new_kernel->removeFnAttr(Attribute::OptimizeNone);
            }
        }

        if (ir_updated && DumpKernel) {
            outs() << "\n\n------------------------------\n"
                   << "--------- Dumping IR ---------\n"
                   << "------------------------------\n\n";
            //new_kernel->print(outs(), nullptr);
            M.print(outs(), nullptr);
            outs() << "\n";
        }

        return ir_updated;
    }

    ////////////////////////////////
    ////////////////////////////////

    // Replace calls to the original function with calls to the new function.
    // In theory, this step is not required, as we process modules containing a single CUDA kernel.
    // We still do this step to guarantee that we obtain valid LLVM code.
    // The additional size parameters are passed with value equal to 0;
    void update_callers(FunctionType &original_signature) {
        for (Value::user_iterator I = original_kernel->user_begin(), E = original_kernel->user_end(); I != E;) {
            CallSite CS(*I++);
            if (CS) {
                Instruction *call = CS.getInstruction();

                std::vector<Value *> new_args;
                new_args.reserve(CS.arg_size());
                CallSite::arg_iterator AI = CS.arg_begin();
                CallSite::arg_iterator AE = CS.arg_end();
                // First, copy regular arguments;
                for (uint i = 0, e = original_signature.getNumParams(); i != e; ++i, ++AI) {
                    new_args.push_back(*AI);
                }
                // Then, insert for each array size that we have added, a constant size of 0;
                for (uint i = 0; i < number_of_added_args; i++) {
                    new_args.push_back(ConstantInt::get(Type::getInt32Ty(original_kernel->getContext()), 0));
                }
                // Lastly, copy any remaining varargs;
                for (; AI != AE; ++AI) {
                    new_args.push_back(*AI);
                }

                SmallVector<OperandBundleDef, 1> OpBundles;
                CS.getOperandBundlesAsDefs(OpBundles);
                CallSite newCS;
                if (InvokeInst *II = dyn_cast<InvokeInst>(call)) {
                    newCS = InvokeInst::Create(new_kernel, II->getNormalDest(), II->getUnwindDest(), new_args, OpBundles, "", call);
                } else {
                    newCS = CallInst::Create(new_kernel, new_args, OpBundles, "", call);
                    cast<CallInst>(newCS.getInstruction())->setTailCallKind(cast<CallInst>(call)->getTailCallKind());
                }
                newCS.setCallingConv(CS.getCallingConv());
                newCS.setAttributes(CS.getAttributes());
                newCS->setDebugLoc(call->getDebugLoc());
                uint64_t w;
                if (call->extractProfTotalWeight(w)) {
                    newCS->setProfWeight(w);
                }
                if (!call->use_empty()) {
                    call->replaceAllUsesWith(newCS.getInstruction());
                }
                newCS->takeName(call);

                // Finally, remove the old call from the program, reducing the use-count of the original function;
                call->eraseFromParent();
            }
        }
    }

    ////////////////////////////////
    ////////////////////////////////

    // Cleanup steps, taken from: https://llvm.org/doxygen/DeadArgumentElimination_8cpp_source.html
    void cleanup() {
        // Clone metadatas from the old function, including debug info descriptor;
        SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
        original_kernel->getAllMetadata(MDs);
        for (auto MD : MDs) {
            new_kernel->addMetadata(MD.first, *MD.second);
        }
        // Fix up any BlockAddresses that refer to the function;
        original_kernel->replaceAllUsesWith(ConstantExpr::getBitCast(new_kernel, original_kernel->getType()));
        // Delete the bitcast that we just created, so that new_kernel does not
        // appear to be address-taken;
        new_kernel->removeDeadConstantUsers();

        // Update the metadata where the original kernel is referenced.
        // Make sure that the new kernel is an entrypoint;
        for (auto pair : metadata_to_update) {
            for (int operand_index : pair.second) {
                pair.first->replaceOperandWith(operand_index, ConstantAsMetadata::get(new_kernel));
            }
        }
    }

    ////////////////////////////////
    ////////////////////////////////

    static char ID;
    std::string kernel_name;
    Function *original_kernel = nullptr;
    Function *new_kernel = nullptr;
    uint number_of_added_args = 0;
    std::map<MDNode *, std::vector<int>> metadata_to_update;
};

char CudaAddSizesPass::ID = 0;

static RegisterPass<CudaAddSizesPass> PassRegistration("add_sizes_cuda_pass", "CUDA - Add Array Sizes Pass",
                                                       false /* Only looks at CFG */,
                                                       false /* Analysis Pass */);

////////////////////////////////
////////////////////////////////

int main(int argc, char **argv) {

    cl::ParseCommandLineOptions(argc, argv);

    std::vector<std::pair<std::string, std::string>> kernels_to_process;

    // Process the test kernels, or a given input;
    if (TestKernels) {
        for (auto pair : example_kernels) {
            kernels_to_process.push_back(std::pair<std::string, std::string>(examples_folder + pair.first, pair.second));
        }
    } else {
        std::string filename = "";
        if (InputFile.getNumOccurrences() > 0) {
            filename = InputFile;
        } else {
            outs() << "no input file given!\n";
            return -1;
        }
        std::string kernel_name = "ALL";
        if (InputKernel.getNumOccurrences() > 0) {
            kernel_name = InputKernel;
        }
        kernels_to_process.push_back(std::pair<std::string, std::string>(filename, kernel_name));
    }

    outs() << "Processing " << kernels_to_process.size() << " kernels\n\n";

    // Keep track of how many kernels are correct;
    int correct_kernels = 0;

    for (uint i = 0; i < kernels_to_process.size(); i++) {

        std::string filename = kernels_to_process[i].first;
        std::string kernel_name = kernels_to_process[i].second;

        // Parse the input LLVM IR file into a module;
        outs() << i + 1 << "/" << kernels_to_process.size() << " parsing file: " << filename << ", kernel: " << kernel_name << "\n";

        SMDiagnostic Err;
        LLVMContext Context;
        std::unique_ptr<Module> Mod(parseIRFile(filename, Err, Context));
        if (!Mod) {
            Err.print(argv[0], errs());
            return 1;
        }

        // Create a pass manager and fill it with the passes we want to run;
        legacy::PassManager PM;
        CudaAddSizesPass *cuda_pass = new CudaAddSizesPass();
        cuda_pass->kernel_name = kernel_name;
        PM.add(cuda_pass);
        PM.run(*Mod);

        bool is_module_broken = verifyModule(*Mod, (DumpKernel ? &outs() : nullptr));

        outs() << "\n--Is the IR valid? " << (is_module_broken ? "No" : "Yes") << "\n\n";

        correct_kernels += is_module_broken ? 0 : 1;
    }
    outs() << "Number of processed kernels with valid IR: " << correct_kernels << "/" << kernels_to_process.size() << "\n";

    return 0;
}