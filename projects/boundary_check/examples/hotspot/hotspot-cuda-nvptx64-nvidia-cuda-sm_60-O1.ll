; ModuleID = 'hotspot.cu'
source_filename = "hotspot.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE13power_on_cuda = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE13power_on_cuda = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4

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
define dso_local void @_Z22calculate_temp_checkediPfS_S_iiiiffffff(i32, float* nocapture readonly, float* nocapture readonly, float* nocapture, i32, i32, i32, i32, float, float, float, float, float, float) local_unnamed_addr #1 {
  %15 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !12
  %17 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !13
  %18 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !13
  %19 = fdiv float %12, %8
  %20 = fdiv float 1.000000e+00, %9
  %21 = fdiv float 1.000000e+00, %10
  %22 = fdiv float 1.000000e+00, %11
  %23 = shl nsw i32 %0, 1
  %24 = sub nsw i32 16, %23
  %25 = mul nsw i32 %16, %24
  %26 = sub nsw i32 %25, %7
  %27 = mul nsw i32 %15, %24
  %28 = sub nsw i32 %27, %6
  %29 = add nsw i32 %26, 15
  %30 = add nsw i32 %28, 15
  %31 = add nsw i32 %26, %18
  %32 = add nsw i32 %28, %17
  %33 = mul nsw i32 %31, %4
  %34 = add nsw i32 %33, %32
  %35 = icmp sgt i32 %31, -1
  br i1 %35, label %36, label %58

36:                                               ; preds = %14
  %37 = icmp slt i32 %31, %5
  %38 = icmp sgt i32 %32, -1
  %39 = and i1 %38, %37
  %40 = icmp slt i32 %32, %4
  %41 = and i1 %40, %39
  br i1 %41, label %42, label %58

42:                                               ; preds = %36
  %43 = sext i32 %34 to i64
  %44 = getelementptr inbounds float, float* %2, i64 %43
  %45 = bitcast float* %44 to i32*
  %46 = load i32, i32* %45, align 4, !tbaa !14
  %47 = zext i32 %18 to i64
  %48 = zext i32 %17 to i64
  %49 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %47, i64 %48
  %50 = bitcast float addrspace(3)* %49 to i32 addrspace(3)*
  %51 = addrspacecast i32 addrspace(3)* %50 to i32*
  store i32 %46, i32* %51, align 4, !tbaa !14
  %52 = getelementptr inbounds float, float* %1, i64 %43
  %53 = bitcast float* %52 to i32*
  %54 = load i32, i32* %53, align 4, !tbaa !14
  %55 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE13power_on_cuda, i64 0, i64 %47, i64 %48
  %56 = bitcast float addrspace(3)* %55 to i32 addrspace(3)*
  %57 = addrspacecast i32 addrspace(3)* %56 to i32*
  store i32 %54, i32* %57, align 4, !tbaa !14
  br label %58

58:                                               ; preds = %42, %36, %14
  tail call void @llvm.nvvm.barrier0()
  %59 = icmp slt i32 %26, 0
  %60 = sub nsw i32 0, %26
  %61 = select i1 %59, i32 %60, i32 0
  %62 = icmp slt i32 %29, %5
  %63 = sub i32 -15, %26
  %64 = add i32 %5, 14
  %65 = add i32 %64, %63
  %66 = select i1 %62, i32 15, i32 %65
  %67 = icmp slt i32 %28, 0
  %68 = sub nsw i32 0, %28
  %69 = select i1 %67, i32 %68, i32 0
  %70 = icmp slt i32 %30, %4
  %71 = sub i32 -15, %28
  %72 = add i32 %4, 14
  %73 = add i32 %72, %71
  %74 = select i1 %70, i32 15, i32 %73
  %75 = add nsw i32 %18, -1
  %76 = add nuw nsw i32 %18, 1
  %77 = add nsw i32 %17, -1
  %78 = add nuw nsw i32 %17, 1
  %79 = icmp sgt i32 %0, 0
  br i1 %79, label %84, label %80

80:                                               ; preds = %58
  %81 = zext i32 %18 to i64
  %82 = zext i32 %17 to i64
  %83 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %81, i64 %82
  br label %185

84:                                               ; preds = %58
  %85 = icmp sgt i32 %78, %74
  %86 = select i1 %85, i32 %74, i32 %78
  %87 = icmp slt i32 %77, %69
  %88 = select i1 %87, i32 %69, i32 %77
  %89 = icmp sgt i32 %76, %66
  %90 = select i1 %89, i32 %66, i32 %76
  %91 = icmp slt i32 %75, %61
  %92 = select i1 %91, i32 %61, i32 %75
  %93 = icmp slt i32 %17, %69
  %94 = icmp sgt i32 %17, %74
  %95 = icmp slt i32 %18, %61
  %96 = icmp sgt i32 %18, %66
  %97 = zext i32 %18 to i64
  %98 = zext i32 %17 to i64
  %99 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %97, i64 %98
  %100 = addrspacecast float addrspace(3)* %99 to float*
  %101 = fpext float %19 to double
  %102 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE13power_on_cuda, i64 0, i64 %97, i64 %98
  %103 = addrspacecast float addrspace(3)* %102 to float*
  %104 = sext i32 %90 to i64
  %105 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %104, i64 %98
  %106 = addrspacecast float addrspace(3)* %105 to float*
  %107 = sext i32 %92 to i64
  %108 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %107, i64 %98
  %109 = addrspacecast float addrspace(3)* %108 to float*
  %110 = fpext float %21 to double
  %111 = sext i32 %86 to i64
  %112 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %97, i64 %111
  %113 = addrspacecast float addrspace(3)* %112 to float*
  %114 = sext i32 %88 to i64
  %115 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %97, i64 %114
  %116 = addrspacecast float addrspace(3)* %115 to float*
  %117 = fpext float %20 to double
  %118 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %97, i64 %98
  %119 = addrspacecast float addrspace(3)* %118 to float*
  %120 = add nsw i32 %0, -1
  %121 = zext i32 %18 to i64
  %122 = zext i32 %17 to i64
  %123 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %121, i64 %122
  %124 = bitcast float addrspace(3)* %123 to i32 addrspace(3)*
  %125 = addrspacecast i32 addrspace(3)* %124 to i32*
  %126 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %121, i64 %122
  %127 = bitcast float addrspace(3)* %126 to i32 addrspace(3)*
  %128 = addrspacecast i32 addrspace(3)* %127 to i32*
  br label %129

129:                                              ; preds = %179, %84
  %130 = phi i32 [ 0, %84 ], [ %131, %179 ]
  %131 = add nuw nsw i32 %130, 1
  %132 = icmp ugt i32 %17, %130
  br i1 %132, label %133, label %171

133:                                              ; preds = %129
  %134 = sub nsw i32 14, %130
  %135 = icmp sgt i32 %17, %134
  %136 = icmp ule i32 %18, %130
  %137 = or i1 %136, %135
  %138 = icmp sgt i32 %18, %134
  %139 = or i1 %138, %137
  %140 = or i1 %93, %139
  %141 = or i1 %94, %140
  %142 = or i1 %95, %141
  %143 = or i1 %96, %142
  br i1 %143, label %171, label %144

144:                                              ; preds = %133
  %145 = load float, float* %100, align 4, !tbaa !14
  %146 = fpext float %145 to double
  %147 = load float, float* %103, align 4, !tbaa !14
  %148 = fpext float %147 to double
  %149 = load float, float* %106, align 4, !tbaa !14
  %150 = load float, float* %109, align 4, !tbaa !14
  %151 = fadd contract float %149, %150
  %152 = fpext float %151 to double
  %153 = fmul contract double %146, 2.000000e+00
  %154 = fsub contract double %152, %153
  %155 = fmul contract double %154, %110
  %156 = fadd contract double %155, %148
  %157 = load float, float* %113, align 4, !tbaa !14
  %158 = load float, float* %116, align 4, !tbaa !14
  %159 = fadd contract float %157, %158
  %160 = fpext float %159 to double
  %161 = fsub contract double %160, %153
  %162 = fmul contract double %161, %117
  %163 = fadd contract double %156, %162
  %164 = fsub contract float 8.000000e+01, %145
  %165 = fmul contract float %22, %164
  %166 = fpext float %165 to double
  %167 = fadd contract double %163, %166
  %168 = fmul contract double %167, %101
  %169 = fadd contract double %168, %146
  %170 = fptrunc double %169 to float
  store float %170, float* %119, align 4, !tbaa !14
  br label %171

171:                                              ; preds = %133, %144, %129
  %172 = phi i1 [ false, %144 ], [ true, %133 ], [ true, %129 ]
  tail call void @llvm.nvvm.barrier0()
  %173 = icmp eq i32 %130, %120
  br i1 %173, label %174, label %176

174:                                              ; preds = %171
  %175 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %121, i64 %122
  br i1 %172, label %193, label %185

176:                                              ; preds = %171
  br i1 %172, label %179, label %177

177:                                              ; preds = %176
  %178 = load i32, i32* %128, align 4, !tbaa !14
  store i32 %178, i32* %125, align 4, !tbaa !14
  br label %179

179:                                              ; preds = %176, %177
  tail call void @llvm.nvvm.barrier0()
  %180 = icmp eq i32 %131, %0
  br i1 %180, label %181, label %129

181:                                              ; preds = %179
  %182 = zext i32 %18 to i64
  %183 = zext i32 %17 to i64
  %184 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %182, i64 %183
  br i1 %172, label %193, label %185

185:                                              ; preds = %181, %80, %174
  %186 = phi float addrspace(3)* [ %175, %174 ], [ %184, %181 ], [ %83, %80 ]
  %187 = sext i32 %34 to i64
  %188 = bitcast float addrspace(3)* %186 to i32 addrspace(3)*
  %189 = getelementptr inbounds float, float* %3, i64 %187
  %190 = addrspacecast i32 addrspace(3)* %188 to i32*
  %191 = bitcast float* %189 to i32*
  %192 = load i32, i32* %190, align 4, !tbaa !14
  store i32 %192, i32* %191, align 4, !tbaa !14
  br label %193

193:                                              ; preds = %181, %174, %185
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: convergent nounwind
define dso_local void @_Z24calculate_temp_uncheckediPfS_S_iiiiffffff(i32, float* nocapture readonly, float* nocapture readonly, float* nocapture, i32, i32, i32, i32, float, float, float, float, float, float) local_unnamed_addr #1 {
  %15 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !12
  %17 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !13
  %18 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !13
  %19 = fdiv float 1.000000e+00, %11
  %20 = shl nsw i32 %0, 1
  %21 = sub nsw i32 16, %20
  %22 = mul nsw i32 %16, %21
  %23 = sub nsw i32 %22, %7
  %24 = mul nsw i32 %15, %21
  %25 = sub nsw i32 %24, %6
  %26 = add nsw i32 %23, 15
  %27 = add nsw i32 %25, 15
  %28 = add nsw i32 %23, %18
  %29 = add nsw i32 %25, %17
  %30 = mul nsw i32 %28, %4
  %31 = add nsw i32 %29, %30
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds float, float* %2, i64 %32
  %34 = bitcast float* %33 to i32*
  %35 = load i32, i32* %34, align 4, !tbaa !14
  %36 = zext i32 %18 to i64
  %37 = zext i32 %17 to i64
  %38 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %36, i64 %37
  %39 = addrspacecast float addrspace(3)* %38 to float*
  %40 = bitcast float addrspace(3)* %38 to i32 addrspace(3)*
  %41 = addrspacecast i32 addrspace(3)* %40 to i32*
  store i32 %35, i32* %41, align 4, !tbaa !14
  %42 = getelementptr inbounds float, float* %1, i64 %32
  %43 = bitcast float* %42 to i32*
  %44 = load i32, i32* %43, align 4, !tbaa !14
  %45 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE13power_on_cuda, i64 0, i64 %36, i64 %37
  %46 = addrspacecast float addrspace(3)* %45 to float*
  %47 = bitcast float addrspace(3)* %45 to i32 addrspace(3)*
  %48 = addrspacecast i32 addrspace(3)* %47 to i32*
  store i32 %44, i32* %48, align 4, !tbaa !14
  tail call void @llvm.nvvm.barrier0()
  %49 = icmp slt i32 %23, 0
  %50 = sub nsw i32 0, %23
  %51 = select i1 %49, i32 %50, i32 0
  %52 = icmp slt i32 %26, %5
  %53 = sub i32 -15, %23
  %54 = add i32 %5, 14
  %55 = add i32 %54, %53
  %56 = select i1 %52, i32 15, i32 %55
  %57 = icmp slt i32 %25, 0
  %58 = sub nsw i32 0, %25
  %59 = select i1 %57, i32 %58, i32 0
  %60 = icmp slt i32 %27, %4
  %61 = sub i32 -15, %25
  %62 = add i32 %4, 14
  %63 = add i32 %62, %61
  %64 = select i1 %60, i32 15, i32 %63
  %65 = add nsw i32 %18, -1
  %66 = add nuw nsw i32 %18, 1
  %67 = add nsw i32 %17, -1
  %68 = add nuw nsw i32 %17, 1
  %69 = icmp sgt i32 %0, 0
  br i1 %69, label %70, label %159

70:                                               ; preds = %14
  %71 = icmp sgt i32 %68, %64
  %72 = select i1 %71, i32 %64, i32 %68
  %73 = icmp slt i32 %67, %59
  %74 = select i1 %73, i32 %59, i32 %67
  %75 = icmp sgt i32 %66, %56
  %76 = select i1 %75, i32 %56, i32 %66
  %77 = icmp slt i32 %65, %51
  %78 = select i1 %77, i32 %51, i32 %65
  %79 = fdiv float 1.000000e+00, %10
  %80 = fdiv float 1.000000e+00, %9
  %81 = fdiv float %12, %8
  %82 = icmp slt i32 %17, %59
  %83 = icmp sgt i32 %17, %64
  %84 = icmp slt i32 %18, %51
  %85 = icmp sgt i32 %18, %56
  %86 = fpext float %81 to double
  %87 = sext i32 %76 to i64
  %88 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %87, i64 %37
  %89 = addrspacecast float addrspace(3)* %88 to float*
  %90 = sext i32 %78 to i64
  %91 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %90, i64 %37
  %92 = addrspacecast float addrspace(3)* %91 to float*
  %93 = fpext float %79 to double
  %94 = sext i32 %72 to i64
  %95 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %36, i64 %94
  %96 = addrspacecast float addrspace(3)* %95 to float*
  %97 = sext i32 %74 to i64
  %98 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda, i64 0, i64 %36, i64 %97
  %99 = addrspacecast float addrspace(3)* %98 to float*
  %100 = fpext float %80 to double
  %101 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %36, i64 %37
  %102 = addrspacecast float addrspace(3)* %101 to float*
  %103 = add nsw i32 %0, -1
  %104 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %36, i64 %37
  %105 = bitcast float addrspace(3)* %104 to i32 addrspace(3)*
  %106 = addrspacecast i32 addrspace(3)* %105 to i32*
  br label %107

107:                                              ; preds = %156, %70
  %108 = phi i32 [ 0, %70 ], [ %109, %156 ]
  %109 = add nuw nsw i32 %108, 1
  %110 = icmp ugt i32 %17, %108
  br i1 %110, label %111, label %149

111:                                              ; preds = %107
  %112 = sub nsw i32 14, %108
  %113 = icmp sgt i32 %17, %112
  %114 = icmp ule i32 %18, %108
  %115 = or i1 %114, %113
  %116 = icmp sgt i32 %18, %112
  %117 = or i1 %116, %115
  %118 = or i1 %82, %117
  %119 = or i1 %83, %118
  %120 = or i1 %84, %119
  %121 = or i1 %85, %120
  br i1 %121, label %149, label %122

122:                                              ; preds = %111
  %123 = load float, float* %39, align 4, !tbaa !14
  %124 = fpext float %123 to double
  %125 = load float, float* %46, align 4, !tbaa !14
  %126 = fpext float %125 to double
  %127 = load float, float* %89, align 4, !tbaa !14
  %128 = load float, float* %92, align 4, !tbaa !14
  %129 = fadd contract float %127, %128
  %130 = fpext float %129 to double
  %131 = fmul contract double %124, 2.000000e+00
  %132 = fsub contract double %130, %131
  %133 = fmul contract double %132, %93
  %134 = fadd contract double %133, %126
  %135 = load float, float* %96, align 4, !tbaa !14
  %136 = load float, float* %99, align 4, !tbaa !14
  %137 = fadd contract float %135, %136
  %138 = fpext float %137 to double
  %139 = fsub contract double %138, %131
  %140 = fmul contract double %139, %100
  %141 = fadd contract double %134, %140
  %142 = fsub contract float 8.000000e+01, %123
  %143 = fmul contract float %19, %142
  %144 = fpext float %143 to double
  %145 = fadd contract double %141, %144
  %146 = fmul contract double %145, %86
  %147 = fadd contract double %146, %124
  %148 = fptrunc double %147 to float
  store float %148, float* %102, align 4, !tbaa !14
  br label %149

149:                                              ; preds = %111, %122, %107
  %150 = phi i1 [ false, %122 ], [ true, %111 ], [ true, %107 ]
  tail call void @llvm.nvvm.barrier0()
  %151 = icmp eq i32 %108, %103
  br i1 %151, label %152, label %153

152:                                              ; preds = %149
  br i1 %150, label %166, label %159

153:                                              ; preds = %149
  br i1 %150, label %156, label %154

154:                                              ; preds = %153
  %155 = load i32, i32* %106, align 4, !tbaa !14
  store i32 %155, i32* %41, align 4, !tbaa !14
  br label %156

156:                                              ; preds = %153, %154
  tail call void @llvm.nvvm.barrier0()
  %157 = icmp eq i32 %109, %0
  br i1 %157, label %158, label %107

158:                                              ; preds = %156
  br i1 %150, label %166, label %159

159:                                              ; preds = %158, %14, %152
  %160 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t, i64 0, i64 %36, i64 %37
  %161 = bitcast float addrspace(3)* %160 to i32 addrspace(3)*
  %162 = getelementptr inbounds float, float* %3, i64 %32
  %163 = addrspacecast i32 addrspace(3)* %161 to i32*
  %164 = bitcast float* %162 to i32*
  %165 = load i32, i32* %163, align 4, !tbaa !14
  store i32 %165, i32* %164, align 4, !tbaa !14
  br label %166

166:                                              ; preds = %158, %152, %159
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx63,+sm_60" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !6, !5, !7, !7, !7, !7, !8, !8, !7}
!llvm.ident = !{!9}
!nvvmir.version = !{!10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (i32, float*, float*, float*, i32, i32, i32, i32, float, float, float, float, float, float)* @_Z22calculate_temp_checkediPfS_S_iiiiffffff, !"kernel", i32 1}
!4 = !{void (i32, float*, float*, float*, i32, i32, i32, i32, float, float, float, float, float, float)* @_Z24calculate_temp_uncheckediPfS_S_iiiiffffff, !"kernel", i32 1}
!5 = !{null, !"align", i32 8}
!6 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!7 = !{null, !"align", i32 16}
!8 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!9 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!10 = !{i32 1, i32 4}
!11 = !{i32 0, i32 2147483647}
!12 = !{i32 0, i32 65535}
!13 = !{i32 0, i32 1024}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
