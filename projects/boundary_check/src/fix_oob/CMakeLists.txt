add_llvm_library( LLVMCudaFixOoBPass MODULE
  fix_oob_cuda_pass.cpp
  main.cpp

  PLUGIN_TOOL
  opt
  )

