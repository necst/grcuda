add_llvm_library( LLVMCudaFixOoBPass MODULE
  fix_oob_cuda_pass.cpp
  function_provider/fix_oob_function_provider.cpp
  function_provider/track_oob_function_provider.cpp
  function_provider/track_and_fix_oob_function_provider.cpp
  main.cpp

  PLUGIN_TOOL
  opt
  )

