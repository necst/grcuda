; ModuleID = 'dot_product.cu'
source_filename = "dot_product.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ11dot_productPfS_iS_E5cache = internal addrspace(3) global [128 x float] undef, align 4
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ19dot_product_checkedPfS_iS_E5cache = internal addrspace(3) global [128 x float] undef, align 4

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
define dso_local void @_Z11dot_productPfS_iS_(float*, float*, i32, float*) #0 {
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
  %21 = load i32, i32* %9, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds float, float* %20, i64 %22
  %24 = load float, float* %23, align 4
  %25 = fmul contract float %19, %24
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache to [128 x float]*), i64 0, i64 %27
  store float %25, float* %28, align 4
  call void @llvm.nvvm.barrier0()
  store i32 64, i32* %9, align 4
  br label %29

29:                                               ; preds = %32, %4
  %30 = load i32, i32* %9, align 4
  %31 = icmp sgt i32 %30, 0
  br i1 %31, label %32, label %46

32:                                               ; preds = %29
  %33 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %34 = load i32, i32* %9, align 4
  %35 = add i32 %33, %34
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache to [128 x float]*), i64 0, i64 %36
  %38 = load float, float* %37, align 4
  %39 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache to [128 x float]*), i64 0, i64 %40
  %42 = load float, float* %41, align 4
  %43 = fadd contract float %42, %38
  store float %43, float* %41, align 4
  call void @llvm.nvvm.barrier0()
  %44 = load i32, i32* %9, align 4
  %45 = sdiv i32 %44, 2
  store i32 %45, i32* %9, align 4
  br label %29

46:                                               ; preds = %29
  %47 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %49, label %53

49:                                               ; preds = %46
  %50 = load float*, float** %8, align 8
  %51 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4
  %52 = call float @_ZL9atomicAddPff(float* %50, float %51) #1
  br label %53

53:                                               ; preds = %49, %46
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
define dso_local void @_Z19dot_product_checkedPfS_iS_(float*, float*, i32, float*) #0 {
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
  br i1 %17, label %18, label %33

18:                                               ; preds = %4
  %19 = load float*, float** %5, align 8
  %20 = load i32, i32* %9, align 4
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds float, float* %19, i64 %21
  %23 = load float, float* %22, align 4
  %24 = load float*, float** %6, align 8
  %25 = load i32, i32* %9, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds float, float* %24, i64 %26
  %28 = load float, float* %27, align 4
  %29 = fmul contract float %23, %28
  %30 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 %31
  store float %29, float* %32, align 4
  br label %33

33:                                               ; preds = %18, %4
  call void @llvm.nvvm.barrier0()
  store i32 64, i32* %9, align 4
  br label %34

34:                                               ; preds = %53, %33
  %35 = load i32, i32* %9, align 4
  %36 = icmp sgt i32 %35, 0
  br i1 %36, label %37, label %56

37:                                               ; preds = %34
  %38 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %39 = load i32, i32* %9, align 4
  %40 = icmp ult i32 %38, %39
  br i1 %40, label %41, label %53

41:                                               ; preds = %37
  %42 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %43 = load i32, i32* %9, align 4
  %44 = add i32 %42, %43
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 %45
  %47 = load float, float* %46, align 4
  %48 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds [128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 %49
  %51 = load float, float* %50, align 4
  %52 = fadd contract float %51, %47
  store float %52, float* %50, align 4
  br label %53

53:                                               ; preds = %41, %37
  call void @llvm.nvvm.barrier0()
  %54 = load i32, i32* %9, align 4
  %55 = sdiv i32 %54, 2
  store i32 %55, i32* %9, align 4
  br label %34

56:                                               ; preds = %34
  %57 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %58 = icmp eq i32 %57, 0
  br i1 %58, label %59, label %63

59:                                               ; preds = %56
  %60 = load float*, float** %8, align 8
  %61 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4
  %62 = call float @_ZL9atomicAddPff(float* %60, float %61) #1
  br label %63

63:                                               ; preds = %59, %56
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
!3 = !{void (float*, float*, i32, float*)* @_Z11dot_productPfS_iS_, !"kernel", i32 1}
!4 = !{void (float*, float*, i32, float*)* @_Z19dot_product_checkedPfS_iS_, !"kernel", i32 1}
!5 = !{null, !"align", i32 8}
!6 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!7 = !{null, !"align", i32 16}
!8 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!9 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!10 = !{i32 1, i32 4}
!11 = !{i32 0, i32 2147483647}
!12 = !{i32 1, i32 1025}
!13 = !{i32 0, i32 1024}
