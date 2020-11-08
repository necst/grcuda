; ModuleID = 'dot_product.cu'
source_filename = "dot_product.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ11dot_productPfS_iS_E5cache = internal unnamed_addr addrspace(3) global [128 x float] undef, align 4
@_ZZ19dot_product_checkedPfS_iS_E5cache = internal unnamed_addr addrspace(3) global [128 x float] undef, align 4

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
define dso_local void @_Z11dot_productPfS_iS_(float* nocapture readonly, float* nocapture readonly, i32, float* nocapture) local_unnamed_addr #1 {
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !range !11
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !range !12
  %7 = mul i32 %6, %5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !range !13
  %9 = add i32 %7, %8
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds float, float* %0, i64 %10
  %12 = load float, float* %11, align 4, !tbaa !14
  %13 = getelementptr inbounds float, float* %1, i64 %10
  %14 = load float, float* %13, align 4, !tbaa !14
  %15 = fmul contract float %12, %14
  %16 = zext i32 %8 to i64
  %17 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache, i64 0, i64 %16
  %18 = addrspacecast float addrspace(3)* %17 to float*
  store float %15, float* %18, align 4, !tbaa !14
  tail call void @llvm.nvvm.barrier0()
  br label %19

19:                                               ; preds = %4, %19
  %20 = phi i32 [ 64, %4 ], [ %28, %19 ]
  %21 = add nuw nsw i32 %20, %8
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache, i64 0, i64 %22
  %24 = addrspacecast float addrspace(3)* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !14
  %26 = load float, float* %18, align 4, !tbaa !14
  %27 = fadd contract float %25, %26
  store float %27, float* %18, align 4, !tbaa !14
  tail call void @llvm.nvvm.barrier0()
  %28 = lshr i32 %20, 1
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %19 

30:                                               ; preds = %19
  %31 = icmp eq i32 %8, 0
  br i1 %31, label %32, label %34

32:                                               ; preds = %30
  %33 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ11dot_productPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4, !tbaa !14
  tail call fastcc void @_ZL9atomicAddPff(float* %3, float %33) #5
  br label %34

34:                                               ; preds = %32, %30
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
define dso_local void @_Z19dot_product_checkedPfS_iS_(float* nocapture readonly, float* nocapture readonly, i32, float* nocapture) local_unnamed_addr #1 {
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !range !11
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !range !12
  %7 = mul i32 %6, %5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !range !13
  %9 = add i32 %7, %8
  %10 = icmp slt i32 %9, %2
  %11 = sext i32 %9 to i64
  br i1 %10, label %12, label %21

12:                                               ; preds = %4
  %13 = zext i32 %8 to i64
  %14 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache, i64 0, i64 %13
  %15 = addrspacecast float addrspace(3)* %14 to float*
  %16 = getelementptr inbounds float, float* %1, i64 %11
  %17 = getelementptr inbounds float, float* %0, i64 %11
  %18 = load float, float* %17, align 4, !tbaa !14
  %19 = load float, float* %16, align 4, !tbaa !14
  %20 = fmul contract float %18, %19
  store float %20, float* %15, align 4, !tbaa !14
  br label %21

21:                                               ; preds = %12, %4
  tail call void @llvm.nvvm.barrier0()
  %22 = zext i32 %8 to i64
  %23 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache, i64 0, i64 %22
  %24 = addrspacecast float addrspace(3)* %23 to float*
  br label %25

25:                                               ; preds = %21, %36
  %26 = phi i32 [ 64, %21 ], [ %37, %36 ]
  %27 = icmp ult i32 %8, %26
  br i1 %27, label %28, label %36

28:                                               ; preds = %25
  %29 = add nuw nsw i32 %26, %8
  %30 = zext i32 %29 to i64
  %31 = getelementptr inbounds [128 x float], [128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache, i64 0, i64 %30
  %32 = addrspacecast float addrspace(3)* %31 to float*
  %33 = load float, float* %32, align 4, !tbaa !14
  %34 = load float, float* %24, align 4, !tbaa !14
  %35 = fadd contract float %33, %34
  store float %35, float* %24, align 4, !tbaa !14
  br label %36

36:                                               ; preds = %28, %25
  tail call void @llvm.nvvm.barrier0()
  %37 = lshr i32 %26, 1
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %39, label %25

39:                                               ; preds = %36
  %40 = icmp eq i32 %8, 0
  br i1 %40, label %41, label %43

41:                                               ; preds = %39
  %42 = load float, float* getelementptr inbounds ([128 x float], [128 x float]* addrspacecast ([128 x float] addrspace(3)* @_ZZ19dot_product_checkedPfS_iS_E5cache to [128 x float]*), i64 0, i64 0), align 4, !tbaa !14
  tail call fastcc void @_ZL9atomicAddPff(float* %3, float %42) #5
  br label %43

43:                                               ; preds = %41, %39
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
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
