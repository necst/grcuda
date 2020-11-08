; ModuleID = 'axpy.cu'
source_filename = "axpy.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

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

; Function Attrs: nofree nounwind
define dso_local void @_Z4axpyPfS_fiS_(float* nocapture readonly, float* nocapture readonly, float, i32, float* nocapture) local_unnamed_addr #1 {
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !13
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !14
  %8 = mul i32 %7, %6
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !15
  %10 = add i32 %8, %9
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds float, float* %0, i64 %11
  %13 = load float, float* %12, align 4, !tbaa !16
  %14 = fmul contract float %13, %2
  %15 = getelementptr inbounds float, float* %1, i64 %11
  %16 = load float, float* %15, align 4, !tbaa !16
  %17 = fadd contract float %14, %16
  %18 = getelementptr inbounds float, float* %4, i64 %11
  store float %17, float* %18, align 4, !tbaa !16
  ret void
}

; Function Attrs: nofree nounwind
define dso_local void @_Z12axpy_checkedPfS_fiS_(float* nocapture readonly, float* nocapture readonly, float, i32, float* nocapture) local_unnamed_addr #1 {
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !13
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !14
  %8 = mul i32 %7, %6
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !15
  %10 = add i32 %8, %9
  %11 = icmp slt i32 %10, %3
  br i1 %11, label %12, label %21

12:                                               ; preds = %5
  %13 = sext i32 %10 to i64
  %14 = getelementptr inbounds float, float* %0, i64 %13
  %15 = load float, float* %14, align 4, !tbaa !16
  %16 = fmul contract float %15, %2
  %17 = getelementptr inbounds float, float* %1, i64 %13
  %18 = load float, float* %17, align 4, !tbaa !16
  %19 = fadd contract float %16, %18
  %20 = getelementptr inbounds float, float* %4, i64 %13
  store float %19, float* %20, align 4, !tbaa !16
  br label %21

21:                                               ; preds = %12, %5
  ret void
}

; Function Attrs: nofree nounwind
define dso_local void @_Z14axpy_with_argsPfiS_ifS_i(float* nocapture readonly, i32, float* nocapture readonly, i32, float, float* nocapture, i32) local_unnamed_addr #1 {
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !13
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !14
  %10 = mul i32 %9, %8
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !15
  %12 = add i32 %10, %11
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds float, float* %0, i64 %13
  %15 = load float, float* %14, align 4, !tbaa !16
  %16 = fmul contract float %15, %4
  %17 = getelementptr inbounds float, float* %2, i64 %13
  %18 = load float, float* %17, align 4, !tbaa !16
  %19 = fadd contract float %16, %18
  %20 = getelementptr inbounds float, float* %5, i64 %13
  store float %19, float* %20, align 4, !tbaa !16
  ret void
}

; Function Attrs: nofree nounwind
define dso_local void @_Z22axpy_with_args_checkedPfiS_ifS_i(float* nocapture readonly, i32, float* nocapture readonly, i32, float, float* nocapture, i32) local_unnamed_addr #1 {
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !13
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !14
  %10 = mul i32 %9, %8
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !15
  %12 = add i32 %10, %11
  %13 = icmp slt i32 %12, %1
  %14 = icmp slt i32 %12, %3
  %15 = and i1 %13, %14
  %16 = icmp slt i32 %12, %6
  %17 = and i1 %16, %15
  br i1 %17, label %18, label %27

18:                                               ; preds = %7
  %19 = sext i32 %12 to i64
  %20 = getelementptr inbounds float, float* %0, i64 %19
  %21 = load float, float* %20, align 4, !tbaa !16
  %22 = fmul contract float %21, %4
  %23 = getelementptr inbounds float, float* %2, i64 %19
  %24 = load float, float* %23, align 4, !tbaa !16
  %25 = fadd contract float %22, %24
  %26 = getelementptr inbounds float, float* %5, i64 %19
  store float %25, float* %26, align 4, !tbaa !16
  br label %27

27:                                               ; preds = %18, %7
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

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
!16 = !{!17, !17, i64 0}
!17 = !{!"float", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C++ TBAA"}
