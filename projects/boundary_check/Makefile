
ifndef LLVM_BUILD_DIR
$(error LLVM_BUILD_DIR should be set to the build directory of LLVM)
endif

ifndef CLANG_PATH
$(error CLANG_PATH should point to where the clang executable is)
endif

BUILDDIR := build
LLVM_BIN_PATH 	:= $(LLVM_BUILD_DIR)/bin

CXX := $(CLANG_PATH)/clang++
CXXFLAGS := -fno-rtti -O0 -g

LLVM_CXXFLAGS := `$(LLVM_BIN_PATH)/llvm-config --cxxflags`
LLVM_LDFLAGS := `$(LLVM_BIN_PATH)/llvm-config --ldflags --libs --system-libs`

.PHONY: all
all: \
	add_sizes_cuda_pass \
	fix_oob_cuda_pass

fix_oob_cuda_pass: src/fix_oob/fix_oob_cuda_pass.cpp src/fix_oob/main.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $^ $(LLVM_LDFLAGS) -o $(BUILDDIR)/$@

add_sizes_cuda_pass: src/add_sizes/add_sizes_cuda_pass.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $^ $(LLVM_LDFLAGS) -o $(BUILDDIR)/$@
