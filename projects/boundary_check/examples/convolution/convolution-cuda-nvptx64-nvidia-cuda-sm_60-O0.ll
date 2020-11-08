; ModuleID = 'convolution.cu'
source_filename = "convolution.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ11convolutionPfS_iS_E5cache = internal addrspace(3) global [128 x float] undef, align 4
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ19convolution_checkedPfS_iS_E5cache = internal addrspace(3) global [128 x float] undef, align 4

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
define dso_local void @_Z11convolutionPfS_iS_(float*, float*, i32, float*) #0 {
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i32, align 4
  %8 = alloca float*, align 8
  %9 = alloca i32, align 4
  store float* %0, float** %5, align 8
  store float* %1, float** %6, align 8
  store i32 %2, i32* %7, align 4
  store float* %3, float** %8, align 8
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !11
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !12
  %12 = mul i32 %10, %11
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %14 = add i32 %12, %13
  store i32 %14, i32* %9, align 4
  %15 = load float*, float** %5, align 8
  %16 = load i32, i32* %9, align 4
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds float, float* %15, i64 %17
  %19 = load float, float* %18, align 4
  %20 = load float*, float** %6, align 8
  %21 = load i32, i32* %7, align 4
  %22 = load i32, i32* %9, align 4
  %23 = sub nsw i32 %21, %22
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds float, float* %20, i64 %24
  %26 = load float, float* %25, align 4
  %27 = fmul contract float %19, %26
  %28 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache to [128 x float]*), i64 0, i64 %29
  store float %27, float* %30, align 4
  call void @llvm.nvvm.barrier0()
  store i32 64, i32* %9, align 4
  br label %31

31:                                               ; preds = %34, %4
  %32 = load i32, i32* %9, align 4
  %33 = icmp sgt i32 %32, 0
  br i1 %33, label %34, label %48

34:                                               ; preds = %31
  %35 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %36 = load i32, i32* %9, align 4
  %37 = add i32 %35, %36
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache to [128 x float]*), i64 0, i64 %38
  %40 = load float, float* %39, align 4
  %41 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache to [128 x float]*), i64 0, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fadd contract float %44, %40
  store float %45, float* %43, align 4
  call void @llvm.nvvm.barrier0()
  %46 = load i32, i32* %9, align 4
  %47 = sdiv i32 %46, 2
  store i32 %47, i32* %9, align 4
  br label %31

48:                                               ; preds = %31
  %49 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %55

51:                                               ; preds = %48
  %52 = load float*, float** %8, align 8
  %53 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4
  %54 = call float @_ZL9atomicAddPff(float* %52, float %53) #1
  br label %55

55:                                               ; preds = %51, %48
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: convergent noinline nounwind optnone
define internal float @_ZL9atomicAddPff(float*, float) #0 {
  %3 = alloca float*, align 8
  %4 = alloca float, align 4
  %5 = alloca float*, align 8
  %6 = alloca float, align 4
  store float* %0, float** %5, align 8
  store float %1, float* %6, align 4
  %7 = load float*, float** %5, align 8
  %8 = load float, float* %6, align 4
  store float* %7, float** %3, align 8
  store float %8, float* %4, align 4
  %9 = load float*, float** %3, align 8
  %10 = load float, float* %4, align 4
  %11 = atomicrmw fadd float* %9, float %10 seq_cst
  ret float %11
}

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z19convolution_checkedPfS_iS_(float*, float*, i32, float*) #0 {
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i32, align 4
  %8 = alloca float*, align 8
  %9 = alloca i32, align 4
  store float* %0, float** %5, align 8
  store float* %1, float** %6, align 8
  store i32 %2, i32* %7, align 4
  store float* %3, float** %8, align 8
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !11
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !12
  %12 = mul i32 %10, %11
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %14 = add i32 %12, %13
  store i32 %14, i32* %9, align 4
  %15 = load i32, i32* %9, align 4
  %16 = load i32, i32* %7, align 4
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %18, label %35

18:                                               ; preds = %4
  %19 = load float*, float** %5, align 8
  %20 = load i32, i32* %9, align 4
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds float, float* %19, i64 %21
  %23 = load float, float* %22, align 4
  %24 = load float*, float** %6, align 8
  %25 = load i32, i32* %7, align 4
  %26 = load i32, i32* %9, align 4
  %27 = sub nsw i32 %25, %26
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds float, float* %24, i64 %28
  %30 = load float, float* %29, align 4
  %31 = fmul contract float %23, %30
  %32 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %33 = zext i32 %32 to i64
  %34 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 %33
  store float %31, float* %34, align 4
  br label %35

35:                                               ; preds = %18, %4
  call void @llvm.nvvm.barrier0()
  store i32 64, i32* %9, align 4
  br label %36

36:                                               ; preds = %55, %35
  %37 = load i32, i32* %9, align 4
  %38 = icmp sgt i32 %37, 0
  br i1 %38, label %39, label %58

39:                                               ; preds = %36
  %40 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %41 = load i32, i32* %9, align 4
  %42 = icmp ult i32 %40, %41
  br i1 %42, label %43, label %55

43:                                               ; preds = %39
  %44 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %45 = load i32, i32* %9, align 4
  %46 = add i32 %44, %45
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 %47
  %49 = load float, float* %48, align 4
  %50 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 %51
  %53 = load float, float* %52, align 4
  %54 = fadd contract float %53, %49
  store float %54, float* %52, align 4
  br label %55

55:                                               ; preds = %43, %39
  call void @llvm.nvvm.barrier0()
  %56 = load i32, i32* %9, align 4
  %57 = sdiv i32 %56, 2
  store i32 %57, i32* %9, align 4
  br label %36

58:                                               ; preds = %36
  %59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %61, label %65

61:                                               ; preds = %58
  %62 = load float*, float** %8, align 8
  %63 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4
  %64 = call float @_ZL9atomicAddPff(float* %62, float %63) #1
  br label %65

65:                                               ; preds = %61, %58
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !6, !5, !7, !7, !7, !7, !8, !8, !7}
!llvm.ident = !{!9}
!nvvmir.version = !{!10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (float*, float*, i32, float*)* @_Z11convolutionPfS_iS_, !"kernel", i32 1}
!4 = !{void (float*, float*, i32, float*)* @_Z19convolution_checkedPfS_iS_, !"kernel", i32 1}
!5 = !{null, !"align", i32 8}
!6 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!7 = !{null, !"align", i32 16}
!8 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!9 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!10 = !{i32 1, i32 4}
!11 = !{i32 0, i32 2147483647}
!12 = !{i32 1, i32 1025}
!13 = !{i32 0, i32 1024}
