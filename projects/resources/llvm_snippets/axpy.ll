; Extracted from `axpy-cuda-nvptx64-nvidia-cuda-sm_60.ll`, using `~/Documents/gpudrano-static-analysis_v1.0/build/bin/clang++  axpy.cu  --cuda-gpu-arch=sm_60  -pthread -std=c++11  -S -emit-llvm`

; LLVM generates 2 files when compiling a .cu file:
;   1. axpy.ll is the host code. GPU kernels have a corresponding function in the host code IR, which takes care of
;     passing parameters and configurations.
;   The idea of having a memory area where array sizes are stored might be difficult to implement,
;     unless we transfer it to GPU before calling the kernels.
;   2. axpy-cuda-nvptx64-nvidia-cuda-sm_60.ll is the device code, which we want to modify.


; Standard axpy kernel

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z4axpyPfS_fiS_(float*, float*, float, i32, float*) #0 {
  ; Allocate local registers for input parameters;
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float, align 4
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  ; Pre-allocate a register for index i;
  %11 = alloca i32, align 4
  ; Store input values into local registers;
  store float* %0, float** %6, align 8
  store float* %1, float** %7, align 8
  store float %2, float* %8, align 4
  store i32 %3, i32* %9, align 4
  store float* %4, float** %10, align 8

  ; Compute the current thread id. CUDA position identifiers: tid, ntid, ctaid, nctaid (number of blocks), and gridid.
  ; ctaid = Cooperative Thread Array Id, i.e. blockId;
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !11
  ; ntid = number of threads per block, i.e. blockDim;
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !12
  %14 = mul i32 %12, %13
  ; tid = Thread index, i.e. threadIdx;
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !13
  %16 = add i32 %14, %15
  store i32 %16, i32* %11, align 4

  ; Perform the main computation;
  %17 = load float, float* %8, align 4 ; load "a"
  %18 = load float*, float** %6, align 8 ; load vector "x"
  %19 = load i32, i32* %11, align 4 ; load index "i"
  %20 = sext i32 %19 to i64 ; sext: casting operation, in this case cast "i" from i32 to i64
  %21 = getelementptr inbounds float, float* %18, i64 %20 ; load x[i]
  %22 = load float, float* %21, align 4
  %23 = fmul contract float %17, %22 ; a * x[i]
  %24 = load float*, float** %7, align 8
  %25 = load i32, i32* %11, align 4 ; load index "i" again (this instruction might be removed by an optimization pass)
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds float, float* %24, i64 %26 ; y[i]
  %28 = load float, float* %27, align 4
  %29 = fadd contract float %23, %28 ; (a * x[i]) + y
  %30 = load float*, float** %10, align 8
  %31 = load i32, i32* %11, align 4 ; load index "i" again (this instruction might be removed by an optimization pass)
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds float, float* %30, i64 %32 ; store result in res[i]
  store float %29, float* %33, align 4
  ret void
}

; axpy kernel with OoB check

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z12axpy_checkedPfS_fiS_(float*, float*, float, i32, float*) #0 {
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float, align 4
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca i32, align 4
  store float* %0, float** %6, align 8
  store float* %1, float** %7, align 8
  store float %2, float* %8, align 4
  store i32 %3, i32* %9, align 4
  store float* %4, float** %10, align 8
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !11
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !12
  %14 = mul i32 %12, %13
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !13
  %16 = add i32 %14, %15
  store i32 %16, i32* %11, align 4

  ; Compare "i" to "n" (i < n)
  %17 = load i32, i32* %11, align 4
  %18 = load i32, i32* %9, align 4
  %19 = icmp slt i32 %17, %18 ; strict less than
  br i1 %19, label %20, label %38

; Registers 12, 13, 15 depends on CUDA indices. So do 14 and 16, as they store results of binary operations dependent on them.
; 11 is also flagged as we store 16 in 11. This can be done with a linear scan over the instructions;

20:                                               ; preds = %5
  %21 = load float, float* %8, align 4
  %22 = load float*, float** %6, align 8 ; 3 operations before the array access -> how standard is this? Probably the 2 loads are standard
  %23 = load i32, i32* %11, align 4
  %24 = sext i32 %23 to i64

; "getelementptr" denotes a pointer/array access. The access is done using register 24, which would be flagged.
; Flagging CUDA-dependent accesses is also doable in the same linear scan.

; Array size: backtrack %22 to %6, keep track of what each register "alloca" contains
;   (in this case, input %0, which we also keep track of, along with other arrays/pointers)

; How do we understand where an "if" check should be placed?
;   Worst case: wrap each "getelementptr" (100% safety)
;   Better solution: find a common flagged register that is used in the access
;     we can create a DAG among flagged registers, and process the DAG backwards to understand how many checks we need;
  %25 = getelementptr inbounds float, float* %22, i64 %24
  %26 = load float, float* %25, align 4
  %27 = fmul contract float %21, %26
  %28 = load float*, float** %7, align 8
  %29 = load i32, i32* %11, align 4
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds float, float* %28, i64 %30
  %32 = load float, float* %31, align 4
  %33 = fadd contract float %27, %32
  %34 = load float*, float** %10, align 8
  %35 = load i32, i32* %11, align 4
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds float, float* %34, i64 %36
  store float %33, float* %37, align 4
  br label %38 ; jump to end

38:                                               ; preds = %20, %5
  ret void
}