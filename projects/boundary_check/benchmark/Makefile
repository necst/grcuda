ifndef CLANG_DIR
$(error CLANG_DIR should point to where the clang executable is)
endif

ifndef CUDA_DIR
$(error CUDA_DIR should point to where CUDA is installed (e.g. /usr/local/cuda-10.1))
endif

CXX := $(CLANG_PATH)/clang++

LLVM_LIB_DIR = $(CLANG_DIR)/../lib
ADD_SIZE_CUDA_PASS = $(LLVM_LIB_DIR)/LLVMCudaAddSizePass.so
FIX_OOB_CUDA_PASS = $(LLVM_LIB_DIR)/LLVMCudaFixOoBPass.so

OPT_LEVEL ?= O0
OPT_LEVEL_FLAG := -$(OPT_LEVEL)

ifdef DUMP
ADD_SIZE_PASS_FLAGS := --dump_updated_kernel
FIX_OOB_PASS_FLAGS := --dump_updated_kernel
endif

# Track OOB accesses in kernel;
ifeq ($(EXECUTION_MODE),1)
ADD_SIZE_PASS_FLAGS := $(ADD_SIZE_PASS_FLAGS) --add_debug_array
FIX_OOB_PASS_FLAGS := $(FIX_OOB_PASS_FLAGS) --oob_protection_type 1
	# Add prints to kernel;
	ifeq ($(PRINT_OOB_ACCESSES),1)
	FIX_OOB_PASS_FLAGS := $(FIX_OOB_PASS_FLAGS) --print_oob
	endif
endif

CXX_FLAGS := --cuda-gpu-arch=sm_70  -pthread -std=c++11 -S -emit-llvm --cuda-device-only
PTXAS_FLAGS := -m64 --gpu-name sm_70

# Folders where files are located/placed;
BIN_DIR := cubin
UNMODIFIED_KERNEL_BIN_DIR := unmodified_kernel_cubin
CUDA_SRC_FOLDER := cuda
LLVM_ORIGINAL_FOLDER := llvm/original
LLVM_ADDED_SIZE_FOLDER := llvm/added_size
LLVM_FIX_OOB_FOLDER := llvm/oob_fixed

# Create the full paths;
BIN_DIR := $(BIN_DIR)/$(OPT_LEVEL)
UNMODIFIED_KERNEL_BIN_DIR := $(UNMODIFIED_KERNEL_BIN_DIR)/$(OPT_LEVEL)
LLVM_ORIGINAL_FOLDER := $(LLVM_ORIGINAL_FOLDER)/$(OPT_LEVEL)
LLVM_ADDED_SIZE_FOLDER := $(LLVM_ADDED_SIZE_FOLDER)/$(OPT_LEVEL)
LLVM_FIX_OOB_FOLDER := $(LLVM_FIX_OOB_FOLDER)/$(OPT_LEVEL)

ifdef SIMPLIFY
FIX_OOB_PASS_FLAGS := $(FIX_OOB_PASS_FLAGS) --simplify_accesses
BIN_DIR := $(BIN_DIR)/simplify_accesses
LLVM_ORIGINAL_FOLDER := $(LLVM_ORIGINAL_FOLDER)/simplify_accesses
LLVM_ADDED_SIZE_FOLDER := $(LLVM_ADDED_SIZE_FOLDER)/simplify_accesses
LLVM_FIX_OOB_FOLDER := $(LLVM_FIX_OOB_FOLDER)/simplify_accesses
endif 

ifndef SIMPLIFY
BIN_DIR := $(BIN_DIR)/no_simplification
LLVM_ORIGINAL_FOLDER := $(LLVM_ORIGINAL_FOLDER)/no_simplification
LLVM_ADDED_SIZE_FOLDER := $(LLVM_ADDED_SIZE_FOLDER)/no_simplification
LLVM_FIX_OOB_FOLDER := $(LLVM_FIX_OOB_FOLDER)/no_simplification
endif

$(echo BIN_DIR)

ifneq ($(OPT_LEVEL),O0)
	OPT_COMMAND_LEVEL = $(CLANG_DIR)/opt $(OPT_LEVEL_FLAG) -S -o $(LLVM_FIX_OOB_FOLDER)/$(DEVICE_LL_NAME) $(LLVM_FIX_OOB_FOLDER)/$(DEVICE_LL_NAME)
endif

# Define a function used to compile each kernel;
define compile_kernel
	mkdir -p $(BIN_DIR)
	mkdir -p $(LLVM_ORIGINAL_FOLDER)
	mkdir -p $(LLVM_ADDED_SIZE_FOLDER)
	mkdir -p $(LLVM_FIX_OOB_FOLDER)
	$(eval CUDA_FILE_NAME = $(1))
	$(eval KERNEL_NAME = $(2))
	$(eval DEVICE_LL_NAME = $(KERNEL_NAME).ll)
	$(eval BIN_DEVICE_FILE = $(KERNEL_NAME).cubin)

	# Obtain the original LLVM code;
	$(CLANG_DIR)/clang++ $(CXX_FLAGS) $(CUDA_FILE_NAME) -o $(LLVM_ORIGINAL_FOLDER)/$(DEVICE_LL_NAME) 
	# Add the array sizes to the function signature;
	$(CLANG_DIR)/opt  -S -o $(LLVM_ADDED_SIZE_FOLDER)/$(DEVICE_LL_NAME) -load $(ADD_SIZE_CUDA_PASS) -add_sizes_cuda_pass < $(LLVM_ORIGINAL_FOLDER)/$(DEVICE_LL_NAME) --kernel $(KERNEL_NAME) $(ADD_SIZE_PASS_FLAGS) 
	# Add out-of-boundary checks;
	$(CLANG_DIR)/opt  -S -o $(LLVM_FIX_OOB_FOLDER)/$(DEVICE_LL_NAME) -load $(FIX_OOB_CUDA_PASS) -fix_oob_cuda_pass < $(LLVM_ADDED_SIZE_FOLDER)/$(DEVICE_LL_NAME) --kernel $(KERNEL_NAME) $(FIX_OOB_PASS_FLAGS) --lower_bounds
	# Perform optimization on LLVM code if required;
	$(OPT_COMMAND_LEVEL)	
	# Turn the LLVM core into an .s file;
	$(CLANG_DIR)/llc $(LLVM_FIX_OOB_FOLDER)/$(DEVICE_LL_NAME) -o $(BIN_DIR)/$(KERNEL_NAME).s
	# Generate the object file;
	$(CUDA_DIR)/bin/ptxas $(PTXAS_FLAGS) $(OPT_LEVEL_FLAG) --output-file $(BIN_DIR)/$(BIN_DEVICE_FILE) $(BIN_DIR)/$(KERNEL_NAME).s
	# Remove the temporary files;
	rm $(BIN_DIR)/$(KERNEL_NAME).s
endef

# Function used to compile CUDA kernels without the additional LLVM passes;
define compile_original_kernel
	mkdir -p $(UNMODIFIED_KERNEL_BIN_DIR)
	mkdir -p $(LLVM_ORIGINAL_FOLDER)
	$(eval CUDA_FILE_NAME := $(1))
	$(eval KERNEL_NAME := $(2))
	$(eval DEVICE_LL_NAME := $(KERNEL_NAME).ll)
	$(eval BIN_DEVICE_FILE := $(KERNEL_NAME).cubin)

	# Obtain the original LLVM code;
	$(CLANG_DIR)/clang++ $(CXX_FLAGS) $(OPT_LEVEL_FLAG) $(CUDA_FILE_NAME) -o $(LLVM_ORIGINAL_FOLDER)/$(DEVICE_LL_NAME)
	# Turn the LLVM core into an .s file;
	$(CLANG_DIR)/llc $(LLVM_ORIGINAL_FOLDER)/$(DEVICE_LL_NAME) -o $(UNMODIFIED_KERNEL_BIN_DIR)/$(KERNEL_NAME).s
	# Generate the object file;
	$(CUDA_DIR)/bin/ptxas $(PTXAS_FLAGS) $(OPT_LEVEL_FLAG) --output-file $(UNMODIFIED_KERNEL_BIN_DIR)/$(BIN_DEVICE_FILE) $(UNMODIFIED_KERNEL_BIN_DIR)/$(KERNEL_NAME).s
	# Remove the temporary files;
	rm $(UNMODIFIED_KERNEL_BIN_DIR)/$(KERNEL_NAME).s
endef


.PHONY: all
all: \
	axpy \
	dot_product \
	convolution \
	hotspot \
	mmul \
	autocov \
	backprop \
	backprop2 \
	bfs \
	gaussian \
	pr \
	hotspot3d \
	histogram \
	lud \
	needle \
	nested

axpy: cuda/axpy.cu
	$(call compile_kernel,$^,axpy)

dot_product: cuda/dot_product.cu
	$(call compile_kernel,$^,dot_product)

convolution: cuda/convolution.cu
	$(call compile_kernel,$^,convolution)

hotspot: cuda/hotspot.cu
	$(call compile_kernel,$^,calculate_temp)

mmul: cuda/mmul.cu
	$(call compile_kernel,$^,mmul)

autocov: cuda/autocov.cu
	$(call compile_kernel,$^,autocov) 

backprop: cuda/backprop.cu
	$(call compile_kernel,$^,backprop) 

backprop2: cuda/backprop2.cu
	$(call compile_kernel,$^,backprop2) 

bfs: cuda/bfs.cu
	$(call compile_kernel,$^,bfs) 

gaussian: cuda/gaussian.cu
	$(call compile_kernel,$^,gaussian) 

pr: cuda/pr.cu
	$(call compile_kernel,$^,pr) 

hotspot3d: cuda/hotspot3d.cu
	$(call compile_kernel,$^,hotspot3d) 

huffman: cuda/huffman.cu
	$(call compile_kernel,$^,huffman) 

histogram: cuda/histogram.cu
	$(call compile_kernel,$^,histogram) 

lud: cuda/lud.cu
	$(call compile_kernel,$^,lud_perimeter) 

needle: cuda/needle.cu
	$(call compile_kernel,$^,needle) 

nested: cuda/nested.cu
	$(call compile_kernel,$^,nested) 

spmv: cuda/spmv.cu
	$(call compile_kernel,$^,spmv) 

.PHONY: unmodified_kernels
unmodified_kernels: \
	axpy_u \
	dot_product_u \
	convolution_u \
	hotspot_u \
	mmul_u \
	autocov_u \
	backprop_u \
	backprop2_u \
	bfs_u \
	gaussian_u \
	pr_u \
	hotspot3d_u \
	histogram_u \
	lud_u \
	needle_u \
	nested_u

axpy_u: cuda/checked/axpy.cu
	$(call compile_original_kernel,cuda/checked/axpy.cu,axpy_checked)

dot_product_u: cuda/checked/dot_product.cu
	$(call compile_original_kernel,cuda/checked/dot_product.cu,dot_product_checked)

convolution_u: cuda/checked/convolution.cu
	$(call compile_original_kernel,cuda/checked/convolution.cu,convolution_checked)

hotspot_u: cuda/checked/hotspot.cu
	$(call compile_original_kernel,cuda/checked/hotspot.cu,calculate_temp_checked)

mmul_u: cuda/checked/mmul.cu
	$(call compile_original_kernel,cuda/checked/mmul.cu,mmul_checked)

autocov_u: cuda/checked/autocov.cu
	$(call compile_original_kernel,cuda/checked/autocov.cu,autocov_checked)

backprop_u: cuda/checked/backprop.cu
	$(call compile_original_kernel,cuda/checked/backprop.cu,backprop_checked)

backprop2_u: cuda/checked/backprop2.cu
	$(call compile_original_kernel,cuda/checked/backprop2.cu,backprop2)

bfs_u: cuda/checked/bfs.cu
	$(call compile_original_kernel,cuda/checked/bfs.cu,bfs_checked)

gaussian_u: cuda/checked/gaussian.cu
	$(call compile_original_kernel,cuda/checked/gaussian.cu,gaussian)

pr_u: cuda/checked/pr.cu
	$(call compile_original_kernel,cuda/checked/pr.cu,pr_checked)

hotspot3d_u: cuda/checked/hotspot3d.cu
	$(call compile_original_kernel,cuda/checked/hotspot3d.cu,hotspot3d_checked)

histogram_u: cuda/checked/histogram.cu
	$(call compile_original_kernel,cuda/checked/histogram.cu,histogram)

lud_u: cuda/checked/lud.cu
	$(call compile_original_kernel,cuda/checked/lud.cu,lud_perimeter)

needle_u: cuda/checked/needle.cu
	$(call compile_original_kernel,cuda/checked/needle.cu,needle)

nested_u: cuda/checked/nested.cu
	$(call compile_original_kernel,cuda/checked/nested.cu,nested)

unsafe_kernels: cuda/axpy.cu cuda/bfs.cu cuda/pr.cu cuda/nested.cu cuda/hotspot3d.cu cuda/backprop.cu
	$(call compile_original_kernel,cuda/axpy.cu,axpy)
	$(call compile_original_kernel,cuda/bfs.cu,bfs)
	$(call compile_original_kernel,cuda/pr.cu,pr)
	$(call compile_original_kernel,cuda/nested.cu,nested)
	$(call compile_original_kernel,cuda/hotspot3d.cu,hotspot3d)
	$(call compile_original_kernel,cuda/backprop.cu,backprop)
