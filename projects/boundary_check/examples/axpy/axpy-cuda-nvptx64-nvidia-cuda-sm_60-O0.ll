; ModuleID = 'axpy.cu'
source_filename = "axpy.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaMalloc(i8**, i64) #0 {
  %3 = alloca i8**, align 8
  %4 = alloca i64, align 8
  store i8** %0, i8*** %3, align 8
  store i64 %1, i64* %4, align 8
  ret i32 30
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaFuncGetAttributes(%struct.cudaFuncAttributes*, i8*) #0 {
  %3 = alloca %struct.cudaFuncAttributes*, align 8
  %4 = alloca i8*, align 8
  store %struct.cudaFuncAttributes* %0, %struct.cudaFuncAttributes** %3, align 8
  store i8* %1, i8** %4, align 8
  ret i32 30
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaDeviceGetAttribute(i32*, i32, i32) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32* %0, i32** %4, align 8
  store i32 %1, i32* %5, align 4
  store i32 %2, i32* %6, align 4
  ret i32 30
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaGetDevice(i32*) #0 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  ret i32 30
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(i32*, i8*, i32, i64) #0 {
  %5 = alloca i32*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i64, align 8
  store i32* %0, i32** %5, align 8
  store i8* %1, i8** %6, align 8
  store i32 %2, i32* %7, align 4
  store i64 %3, i64* %8, align 8
  ret i32 30
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(i32*, i8*, i32, i64, i32) #0 {
  %6 = alloca i32*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i32, align 4
  %9 = alloca i64, align 8
  %10 = alloca i32, align 4
  store i32* %0, i32** %6, align 8
  store i8* %1, i8** %7, align 8
  store i32 %2, i32* %8, align 4
  store i64 %3, i64* %9, align 8
  store i32 %4, i32* %10, align 4
  ret i32 30
}

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z4axpyPfS_fiS_(float*, float*, float, i32, float*) #0 {
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
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !13
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !14
  %14 = mul i32 %12, %13
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !15
  %16 = add i32 %14, %15
  store i32 %16, i32* %11, align 4
  %17 = load float, float* %8, align 4
  %18 = load float*, float** %6, align 8
  %19 = load i32, i32* %11, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds float, float* %18, i64 %20
  %22 = load float, float* %21, align 4
  %23 = fmul contract float %17, %22
  %24 = load float*, float** %7, align 8
  %25 = load i32, i32* %11, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds float, float* %24, i64 %26
  %28 = load float, float* %27, align 4
  %29 = fadd contract float %23, %28
  %30 = load float*, float** %10, align 8
  %31 = load i32, i32* %11, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds float, float* %30, i64 %32
  store float %29, float* %33, align 4
  ret void
}

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
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !13
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !14
  %14 = mul i32 %12, %13
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !15
  %16 = add i32 %14, %15
  store i32 %16, i32* %11, align 4
  %17 = load i32, i32* %11, align 4
  %18 = load i32, i32* %9, align 4
  %19 = icmp slt i32 %17, %18
  br i1 %19, label %20, label %38

20:                                               ; preds = %5
  %21 = load float, float* %8, align 4
  %22 = load float*, float** %6, align 8
  %23 = load i32, i32* %11, align 4
  %24 = sext i32 %23 to i64
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
  br label %38

38:                                               ; preds = %20, %5
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z14axpy_with_argsPfiS_ifS_i(float*, i32, float*, i32, float, float*, i32) #0 {
  %8 = alloca float*, align 8
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca i32, align 4
  %12 = alloca float, align 4
  %13 = alloca float*, align 8
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  store float* %0, float** %8, align 8
  store i32 %1, i32* %9, align 4
  store float* %2, float** %10, align 8
  store i32 %3, i32* %11, align 4
  store float %4, float* %12, align 4
  store float* %5, float** %13, align 8
  store i32 %6, i32* %14, align 4
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !13
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !14
  %18 = mul i32 %16, %17
  %19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !15
  %20 = add i32 %18, %19
  store i32 %20, i32* %15, align 4
  %21 = load float, float* %12, align 4
  %22 = load float*, float** %8, align 8
  %23 = load i32, i32* %15, align 4
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds float, float* %22, i64 %24
  %26 = load float, float* %25, align 4
  %27 = fmul contract float %21, %26
  %28 = load float*, float** %10, align 8
  %29 = load i32, i32* %15, align 4
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds float, float* %28, i64 %30
  %32 = load float, float* %31, align 4
  %33 = fadd contract float %27, %32
  %34 = load float*, float** %13, align 8
  %35 = load i32, i32* %15, align 4
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds float, float* %34, i64 %36
  store float %33, float* %37, align 4
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z22axpy_with_args_checkedPfiS_ifS_i(float*, i32, float*, i32, float, float*, i32) #0 {
  %8 = alloca float*, align 8
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca i32, align 4
  %12 = alloca float, align 4
  %13 = alloca float*, align 8
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  store float* %0, float** %8, align 8
  store i32 %1, i32* %9, align 4
  store float* %2, float** %10, align 8
  store i32 %3, i32* %11, align 4
  store float %4, float* %12, align 4
  store float* %5, float** %13, align 8
  store i32 %6, i32* %14, align 4
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !13
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !14
  %18 = mul i32 %16, %17
  %19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !15
  %20 = add i32 %18, %19
  store i32 %20, i32* %15, align 4
  %21 = load i32, i32* %15, align 4
  %22 = load i32, i32* %9, align 4
  %23 = icmp slt i32 %21, %22
  br i1 %23, label %24, label %50

24:                                               ; preds = %7
  %25 = load i32, i32* %15, align 4
  %26 = load i32, i32* %11, align 4
  %27 = icmp slt i32 %25, %26
  br i1 %27, label %28, label %50

28:                                               ; preds = %24
  %29 = load i32, i32* %15, align 4
  %30 = load i32, i32* %14, align 4
  %31 = icmp slt i32 %29, %30
  br i1 %31, label %32, label %50

32:                                               ; preds = %28
  %33 = load float, float* %12, align 4
  %34 = load float*, float** %8, align 8
  %35 = load i32, i32* %15, align 4
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds float, float* %34, i64 %36
  %38 = load float, float* %37, align 4
  %39 = fmul contract float %33, %38
  %40 = load float*, float** %10, align 8
  %41 = load i32, i32* %15, align 4
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds float, float* %40, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fadd contract float %39, %44
  %46 = load float*, float** %13, align 8
  %47 = load i32, i32* %15, align 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds float, float* %46, i64 %48
  store float %45, float* %49, align 4
  br label %50

50:                                               ; preds = %32, %28, %24, %7
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !6, !7, !8, !7, !9, !9, !9, !9, !10, !10, !9}
!llvm.ident = !{!11}
!nvvmir.version = !{!12}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (float*, float*, float, i32, float*)* @_Z4axpyPfS_fiS_, !"kernel", i32 1}
!4 = !{void (float*, float*, float, i32, float*)* @_Z12axpy_checkedPfS_fiS_, !"kernel", i32 1}
!5 = !{void (float*, i32, float*, i32, float, float*, i32)* @_Z14axpy_with_argsPfiS_ifS_i, !"kernel", i32 1}
!6 = !{void (float*, i32, float*, i32, float, float*, i32)* @_Z22axpy_with_args_checkedPfiS_ifS_i, !"kernel", i32 1}
!7 = !{null, !"align", i32 8}
!8 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!9 = !{null, !"align", i32 16}
!10 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!11 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!12 = !{i32 1, i32 4}
!13 = !{i32 0, i32 2147483647}
!14 = !{i32 1, i32 1025}
!15 = !{i32 0, i32 1024}
