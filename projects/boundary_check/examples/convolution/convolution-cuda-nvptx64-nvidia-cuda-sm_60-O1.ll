; ModuleID = 'convolution.cu'
source_filename = "convolution.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ11convolutionPfS_iS_E5cache = internal unnamed_addr addrspace(3) global [128 x float] undef, align 4
@_ZZ19convolution_checkedPfS_iS_E5cache = internal unnamed_addr addrspace(3) global [128 x float] undef, align 4

; Function Attrs: nounwind
define weak dso_local i32 @cudaMalloc(i8**, i64) local_unnamed_addr #0 {
  ret i32 30
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaFuncGetAttributes(%struct.cudaFuncAttributes*, i8*) local_unnamed_addr #0 {
  ret i32 30
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaDeviceGetAttribute(i32*, i32, i32) local_unnamed_addr #0 {
  ret i32 30
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaGetDevice(i32*) local_unnamed_addr #0 {
  ret i32 30
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(i32*, i8*, i32, i64) local_unnamed_addr #0 {
  ret i32 30
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(i32*, i8*, i32, i64, i32) local_unnamed_addr #0 {
  ret i32 30
}

; Function Attrs: convergent nounwind
define dso_local void @_Z11convolutionPfS_iS_(float* nocapture readonly, float* nocapture readonly, i32, float* nocapture) local_unnamed_addr #1 {
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !range !11
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !range !12
  %7 = mul i32 %6, %5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !range !13
  %9 = add i32 %7, %8
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds float, float* %0, i64 %10
  %12 = load float, float* %11, align 4, !tbaa !14
  %13 = sub nsw i32 %2, %9
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds float, float* %1, i64 %14
  %16 = load float, float* %15, align 4, !tbaa !14
  %17 = fmul contract float %12, %16
  %18 = zext i32 %8 to i64
  %19 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache, i64 0, i64 %18
  %20 = addrspacecast float addrspace(3)* %19 to float*
  store float %17, float* %20, align 4, !tbaa !14
  tail call void @llvm.nvvm.barrier0()
  br label %21

21:                                               ; preds = %4, %21
  %22 = phi i32 [ 64, %4 ], [ %30, %21 ]
  %23 = add nuw nsw i32 %22, %8
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache, i64 0, i64 %24
  %26 = addrspacecast float addrspace(3)* %25 to float*
  %27 = load float, float* %26, align 4, !tbaa !14
  %28 = load float, float* %20, align 4, !tbaa !14
  %29 = fadd contract float %27, %28
  store float %29, float* %20, align 4, !tbaa !14
  tail call void @llvm.nvvm.barrier0()
  %30 = lshr i32 %22, 1
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %21

32:                                               ; preds = %21
  %33 = icmp eq i32 %8, 0
  br i1 %33, label %34, label %36

34:                                               ; preds = %32
  %35 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11convolutionPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4, !tbaa !14
  tail call fastcc void @_ZL9atomicAddPff(float* %3, float %35) #5
  br label %36

36:                                               ; preds = %34, %32
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: inlinehint nofree norecurse nounwind
define internal fastcc void @_ZL9atomicAddPff(float* nocapture, float) unnamed_addr #3 {
  %3 = atomicrmw fadd float* %0, float %1 seq_cst
  ret void
}

; Function Attrs: convergent nounwind
define dso_local void @_Z19convolution_checkedPfS_iS_(float* nocapture readonly, float* nocapture readonly, i32, float* nocapture) local_unnamed_addr #1 {
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !range !11
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !range !12
  %7 = mul i32 %6, %5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !range !13
  %9 = add i32 %7, %8
  %10 = icmp slt i32 %9, %2
  br i1 %10, label %11, label %23

11:                                               ; preds = %4
  %12 = sext i32 %9 to i64
  %13 = getelementptr inbounds float, float* %0, i64 %12
  %14 = load float, float* %13, align 4, !tbaa !14
  %15 = sub nsw i32 %2, %9
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds float, float* %1, i64 %16
  %18 = load float, float* %17, align 4, !tbaa !14
  %19 = fmul contract float %14, %18
  %20 = zext i32 %8 to i64
  %21 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache, i64 0, i64 %20
  %22 = addrspacecast float addrspace(3)* %21 to float*
  store float %19, float* %22, align 4, !tbaa !14
  br label %23

23:                                               ; preds = %11, %4
  tail call void @llvm.nvvm.barrier0()
  %24 = zext i32 %8 to i64
  %25 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache, i64 0, i64 %24
  %26 = addrspacecast float addrspace(3)* %25 to float*
  br label %27

27:                                               ; preds = %23, %38
  %28 = phi i32 [ 64, %23 ], [ %39, %38 ]
  %29 = icmp ult i32 %8, %28
  br i1 %29, label %30, label %38

30:                                               ; preds = %27
  %31 = add nuw nsw i32 %28, %8
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache, i64 0, i64 %32
  %34 = addrspacecast float addrspace(3)* %33 to float*
  %35 = load float, float* %34, align 4, !tbaa !14
  %36 = load float, float* %26, align 4, !tbaa !14
  %37 = fadd contract float %35, %36
  store float %37, float* %26, align 4, !tbaa !14
  br label %38

38:                                               ; preds = %30, %27
  tail call void @llvm.nvvm.barrier0()
  %39 = lshr i32 %28, 1
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %41, label %27

41:                                               ; preds = %38
  %42 = icmp eq i32 %8, 0
  br i1 %42, label %43, label %45

43:                                               ; preds = %41
  %44 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19convolution_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4, !tbaa !14
  tail call fastcc void @_ZL9atomicAddPff(float* %3, float %44) #5
  br label %45

45:                                               ; preds = %43, %41
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }
attributes #3 = { inlinehint nofree norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind }

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
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
