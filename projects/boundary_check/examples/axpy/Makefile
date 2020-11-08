CLANG_DIR = ~/shared_home/Documents/gpudrano-static-analysis_v1.0/build/bin
CUDA_DIR = /usr/local/cuda-10.1
LLVM_LIB_DIR = $(CLANG_DIR)/../lib
CUDA_PASS_PATH = $(LLVM_LIB_DIR)/LLVMCudaTestPass.so

PASS_FLAGS := --dump_updated_kernel

BIN_DIR = ../bin/modified

OPT_LEVEL ?= O0
OPT_LEVEL_FLAG := -$(OPT_LEVEL)

BIN_DIR := $(BIN_DIR)/$(OPT_LEVEL)

ifdef SIMPLIFY
PASS_FLAGS := $(PASS_FLAGS) --simplify_accesses
BIN_DIR := $(BIN_DIR)/simplify_accesses
endif 

ifndef SIMPLIFY
BIN_DIR := $(BIN_DIR)/no_simplification
endif

$(echo BIN_DIR)

.PHONY: all
all: \
	axpy
	
axpy: axpy.cu
	mkdir -p $(BIN_DIR)
	$(eval KERNEL_NAME = axpy_with_args)
	$(eval CUDA_FILE_NAME = axpy.cu)
	$(eval DEVICE_LL_NAME = $(KERNEL_NAME)-d.ll)
	$(eval DEVICE_LL_NAME_FIXED = $(KERNEL_NAME)-d-fixed.ll)
	$(eval BIN_DEVICE_FILE = $(KERNEL_NAME)-d.o)
	$(eval BIN_HOST_FILE = $(KERNEL_NAME)-h.o)
	$(eval FATBIN_NAME = $(KERNEL_NAME).fatbin)
	$(CLANG_DIR)/clang++  $(CUDA_FILE_NAME) --cuda-gpu-arch=sm_60  -pthread -std=c++11 -S -emit-llvm --cuda-device-only -o $(DEVICE_LL_NAME) $(OPT_LEVEL_FLAG)

	$(CLANG_DIR)/opt --debug-pass=Structure -S -o $(DEVICE_LL_NAME_FIXED) -load $(CUDA_PASS_PATH) -cuda_test_pass < $(DEVICE_LL_NAME) --kernel $(KERNEL_NAME) $(PASS_FLAGS)

	$(CLANG_DIR)/llc $(DEVICE_LL_NAME_FIXED) -o $(KERNEL_NAME).s

	$(CUDA_DIR)/bin/ptxas -m64 -O0 --gpu-name sm_60 --output-file $(BIN_DEVICE_FILE) $(KERNEL_NAME).s

	$(CUDA_DIR)/bin/fatbinary -64 --create $(FATBIN_NAME) --image=profile=sm_60,file=$(BIN_DEVICE_FILE) --image=profile=compute_60,file=$(KERNEL_NAME).s

	$(CLANG_DIR)/clang++ -cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=10.1 -aux-triple nvptx64-nvidia-cuda -emit-obj \
	 -main-file-name $(CUDA_FILE_NAME) -target-cpu x86-64 \
	 -internal-isystem $(LLVM_LIB_DIR)/clang/10.0.0/include/cuda_wrappers \
	 -internal-isystem /usr/local/cuda-10.1/include -include __clang_cuda_runtime_wrapper.h \
	 -internal-isystem /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../include/c++/8 \
	 -internal-isystem /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../include/c++/8/x86_64-redhat-linux \
	 -internal-isystem /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../include/c++/8/backward \
	 -internal-isystem /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../include/c++/8 \
	 -internal-isystem /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../include/c++/8/x86_64-redhat-linux \
	 -internal-isystem /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../include/c++/8/backward \
	 -internal-isystem /usr/local/include -internal-isystem $(LLVM_LIB_DIR)/clang/10.0.0/include \
	 -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include \
	 -internal-isystem $(LLVM_LIB_DIR)/clang/10.0.0/include \
	 -internal-externc-isystem /include -internal-externc-isystem /usr/include \
	 -std=c++11 -pthread -fobjc-runtime=gcc -fcuda-include-gpubinary $(FATBIN_NAME) -o $(BIN_HOST_FILE) -x cuda $(CUDA_FILE_NAME)

	ld --hash-style=gnu --no-add-needed --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 \
	 -o $(BIN_DIR)/$(KERNEL_NAME) /lib/../lib64/crt1.o /lib/../lib64/crti.o /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/crtbegin.o \
	 -L/usr/local/cuda/lib64 -L/opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8 \
	 -L/opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 \
	 -L/opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/../../.. -L $(LLVM_LIB_DIR) \
	 -L/lib -L/usr/lib $(BIN_HOST_FILE) -lcudart_static -ldl -lrt -lstdc++ -lm -lgcc_s -lgcc -lpthread -lc -lgcc_s \
	 -lgcc /opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/crtend.o /lib/../lib64/crtn.o
	
	rm $(DEVICE_LL_NAME) $(DEVICE_LL_NAME_FIXED) $(BIN_DEVICE_NAME) $(BIN_HOST_FILE) $(FATBIN_NAME) 

