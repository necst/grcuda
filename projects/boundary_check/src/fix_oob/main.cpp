#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/Verifier.h"
#include "cxxabi.h"
#include <string>
#include <iostream>

#include "fix_oob_cuda_pass.hpp"

////////////////////////////////
////////////////////////////////

using namespace llvm;
using std::cout;
using std::endl;

////////////////////////////////
////////////////////////////////

std::string examples_folder = "benchmark/llvm/added_size/O0/no_simplification/";

std::vector<std::pair<std::string, std::string>> example_kernels = {
    std::pair<std::string, std::string>("axpy.ll", "axpy"),
    std::pair<std::string, std::string>("dot_product.ll", "dot_product"),
    std::pair<std::string, std::string>("convolution.ll", "convolution"),
    std::pair<std::string, std::string>("calculate_temp.ll", "calculate_temp"),
    std::pair<std::string, std::string>("backprop.ll", "backprop"),
    std::pair<std::string, std::string>("backprop2.ll", "backprop2"),
    std::pair<std::string, std::string>("bfs.ll", "bfs"),
    std::pair<std::string, std::string>("pr.ll", "pr")
    };

////////////////////////////////
////////////////////////////////

int main(int argc, char **argv) {

    extern cl::opt<std::string> InputFile;
    extern cl::opt<std::string> InputKernel;
    extern cl::opt<bool> UseInputSizes;
    extern cl::opt<bool> SimplifyAccesses;
    extern cl::opt<bool> TestKernels;
    extern cl::opt<bool> Debug;
    extern cl::opt<bool> DumpKernel;
    extern cl::opt<int> Mode;

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
        TestPass *test_pass = new TestPass();
        test_pass->kernel_name = kernel_name;
        PM.add(test_pass);
        PM.run(*Mod);

        bool is_module_broken = verifyModule(*Mod, (DumpKernel ? &outs() : nullptr));

        outs() << "\n--Is the IR valid? " << (is_module_broken ? "No" : "Yes") << "\n\n";

        correct_kernels += is_module_broken ? 0 : 1;
    }
    outs() << "Number of processed kernels with valid IR: " << correct_kernels << "/" << kernels_to_process.size() << "\n";

    return 0;
}