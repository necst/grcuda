add_llvm_library( LLVMCudaAddSizePass MODULE
  add_sizes_cuda_pass.cpp

  PLUGIN_TOOL
  opt
  )
