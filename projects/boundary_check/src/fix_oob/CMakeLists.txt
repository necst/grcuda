add_llvm_library( LLVMCudaFixOoBPass MODULE
  fix_oob_cuda_pass.cpp
  fix_oob_function_provider.cpp
  main.cpp

  PLUGIN_TOOL
  opt
  )

