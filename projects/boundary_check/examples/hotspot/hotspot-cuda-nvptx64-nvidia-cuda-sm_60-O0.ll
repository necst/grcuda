; ModuleID = 'hotspot.cu'
source_filename = "hotspot.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE13power_on_cuda = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE13power_on_cuda = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t = internal addrspace(3) global [16 x [16 x float]] undef, align 4

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
define dso_local void @_Z22calculate_temp_checkediPfS_S_iiiiffffff(i32, float*, float*, float*, i32, i32, i32, i32, float, float, float, float, float, float) #0 {
  %15 = alloca i32, align 4
  %16 = alloca float*, align 8
  %17 = alloca float*, align 8
  %18 = alloca float*, align 8
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca i32, align 4
  %22 = alloca i32, align 4
  %23 = alloca float, align 4
  %24 = alloca float, align 4
  %25 = alloca float, align 4
  %26 = alloca float, align 4
  %27 = alloca float, align 4
  %28 = alloca float, align 4
  %29 = alloca float, align 4
  %30 = alloca float, align 4
  %31 = alloca float, align 4
  %32 = alloca float, align 4
  %33 = alloca float, align 4
  %34 = alloca i32, align 4
  %35 = alloca i32, align 4
  %36 = alloca i32, align 4
  %37 = alloca i32, align 4
  %38 = alloca i32, align 4
  %39 = alloca i32, align 4
  %40 = alloca i32, align 4
  %41 = alloca i32, align 4
  %42 = alloca i32, align 4
  %43 = alloca i32, align 4
  %44 = alloca i32, align 4
  %45 = alloca i32, align 4
  %46 = alloca i32, align 4
  %47 = alloca i32, align 4
  %48 = alloca i32, align 4
  %49 = alloca i32, align 4
  %50 = alloca i32, align 4
  %51 = alloca i32, align 4
  %52 = alloca i32, align 4
  %53 = alloca i32, align 4
  %54 = alloca i32, align 4
  %55 = alloca i32, align 4
  %56 = alloca i32, align 4
  %57 = alloca i8, align 1
  %58 = alloca i32, align 4
  store i32 %0, i32* %15, align 4
  store float* %1, float** %16, align 8
  store float* %2, float** %17, align 8
  store float* %3, float** %18, align 8
  store i32 %4, i32* %19, align 4
  store i32 %5, i32* %20, align 4
  store i32 %6, i32* %21, align 4
  store i32 %7, i32* %22, align 4
  store float %8, float* %23, align 4
  store float %9, float* %24, align 4
  store float %10, float* %25, align 4
  store float %11, float* %26, align 4
  store float %12, float* %27, align 4
  store float %13, float* %28, align 4
  store float 8.000000e+01, float* %29, align 4
  %59 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !11
  store i32 %59, i32* %34, align 4
  %60 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !12
  store i32 %60, i32* %35, align 4
  %61 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  store i32 %61, i32* %36, align 4
  %62 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !13
  store i32 %62, i32* %37, align 4
  %63 = load float, float* %27, align 4
  %64 = load float, float* %23, align 4
  %65 = fdiv float %63, %64
  store float %65, float* %30, align 4
  %66 = load float, float* %24, align 4
  %67 = fdiv float 1.000000e+00, %66
  store float %67, float* %31, align 4
  %68 = load float, float* %25, align 4
  %69 = fdiv float 1.000000e+00, %68
  store float %69, float* %32, align 4
  %70 = load float, float* %26, align 4
  %71 = fdiv float 1.000000e+00, %70
  store float %71, float* %33, align 4
  %72 = load i32, i32* %15, align 4
  %73 = mul nsw i32 %72, 2
  %74 = sub nsw i32 16, %73
  store i32 %74, i32* %38, align 4
  %75 = load i32, i32* %15, align 4
  %76 = mul nsw i32 %75, 2
  %77 = sub nsw i32 16, %76
  store i32 %77, i32* %39, align 4
  %78 = load i32, i32* %38, align 4
  %79 = load i32, i32* %35, align 4
  %80 = mul nsw i32 %78, %79
  %81 = load i32, i32* %22, align 4
  %82 = sub nsw i32 %80, %81
  store i32 %82, i32* %40, align 4
  %83 = load i32, i32* %39, align 4
  %84 = load i32, i32* %34, align 4
  %85 = mul nsw i32 %83, %84
  %86 = load i32, i32* %21, align 4
  %87 = sub nsw i32 %85, %86
  store i32 %87, i32* %41, align 4
  %88 = load i32, i32* %40, align 4
  %89 = add nsw i32 %88, 16
  %90 = sub nsw i32 %89, 1
  store i32 %90, i32* %42, align 4
  %91 = load i32, i32* %41, align 4
  %92 = add nsw i32 %91, 16
  %93 = sub nsw i32 %92, 1
  store i32 %93, i32* %43, align 4
  %94 = load i32, i32* %40, align 4
  %95 = load i32, i32* %37, align 4
  %96 = add nsw i32 %94, %95
  store i32 %96, i32* %44, align 4
  %97 = load i32, i32* %41, align 4
  %98 = load i32, i32* %36, align 4
  %99 = add nsw i32 %97, %98
  store i32 %99, i32* %45, align 4
  %100 = load i32, i32* %44, align 4
  store i32 %100, i32* %46, align 4
  %101 = load i32, i32* %45, align 4
  store i32 %101, i32* %47, align 4
  %102 = load i32, i32* %19, align 4
  %103 = load i32, i32* %46, align 4
  %104 = mul nsw i32 %102, %103
  %105 = load i32, i32* %47, align 4
  %106 = add nsw i32 %104, %105
  store i32 %106, i32* %48, align 4
  %107 = load i32, i32* %46, align 4
  %108 = icmp sge i32 %107, 0
  br i1 %108, label %109, label %145

109:                                              ; preds = %14
  %110 = load i32, i32* %46, align 4
  %111 = load i32, i32* %20, align 4
  %112 = sub nsw i32 %111, 1
  %113 = icmp sle i32 %110, %112
  br i1 %113, label %114, label %145

114:                                              ; preds = %109
  %115 = load i32, i32* %47, align 4
  %116 = icmp sge i32 %115, 0
  br i1 %116, label %117, label %145

117:                                              ; preds = %114
  %118 = load i32, i32* %47, align 4
  %119 = load i32, i32* %19, align 4
  %120 = sub nsw i32 %119, 1
  %121 = icmp sle i32 %118, %120
  br i1 %121, label %122, label %145

122:                                              ; preds = %117
  %123 = load float*, float** %17, align 8
  %124 = load i32, i32* %48, align 4
  %125 = sext i32 %124 to i64
  %126 = getelementptr inbounds float, float* %123, i64 %125
  %127 = load float, float* %126, align 4
  %128 = load i32, i32* %37, align 4
  %129 = sext i32 %128 to i64
  %130 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %129
  %131 = load i32, i32* %36, align 4
  %132 = sext i32 %131 to i64
  %133 = getelementptr inbounds [16 x float], [16 x float]* %130, i64 0, i64 %132
  store float %127, float* %133, align 4
  %134 = load float*, float** %16, align 8
  %135 = load i32, i32* %48, align 4
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds float, float* %134, i64 %136
  %138 = load float, float* %137, align 4
  %139 = load i32, i32* %37, align 4
  %140 = sext i32 %139 to i64
  %141 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE13power_on_cuda to [16 x [16 x float]]*), i64 0, i64 %140
  %142 = load i32, i32* %36, align 4
  %143 = sext i32 %142 to i64
  %144 = getelementptr inbounds [16 x float], [16 x float]* %141, i64 0, i64 %143
  store float %138, float* %144, align 4
  br label %145

145:                                              ; preds = %122, %117, %114, %109, %14
  call void @llvm.nvvm.barrier0()
  %146 = load i32, i32* %40, align 4
  %147 = icmp slt i32 %146, 0
  br i1 %147, label %148, label %151

148:                                              ; preds = %145
  %149 = load i32, i32* %40, align 4
  %150 = sub nsw i32 0, %149
  br label %152

151:                                              ; preds = %145
  br label %152

152:                                              ; preds = %151, %148
  %153 = phi i32 [ %150, %148 ], [ 0, %151 ]
  store i32 %153, i32* %49, align 4
  %154 = load i32, i32* %42, align 4
  %155 = load i32, i32* %20, align 4
  %156 = sub nsw i32 %155, 1
  %157 = icmp sgt i32 %154, %156
  br i1 %157, label %158, label %164

158:                                              ; preds = %152
  %159 = load i32, i32* %42, align 4
  %160 = load i32, i32* %20, align 4
  %161 = sub nsw i32 %159, %160
  %162 = add nsw i32 %161, 1
  %163 = sub nsw i32 15, %162
  br label %165

164:                                              ; preds = %152
  br label %165

165:                                              ; preds = %164, %158
  %166 = phi i32 [ %163, %158 ], [ 15, %164 ]
  store i32 %166, i32* %50, align 4
  %167 = load i32, i32* %41, align 4
  %168 = icmp slt i32 %167, 0
  br i1 %168, label %169, label %172

169:                                              ; preds = %165
  %170 = load i32, i32* %41, align 4
  %171 = sub nsw i32 0, %170
  br label %173

172:                                              ; preds = %165
  br label %173

173:                                              ; preds = %172, %169
  %174 = phi i32 [ %171, %169 ], [ 0, %172 ]
  store i32 %174, i32* %51, align 4
  %175 = load i32, i32* %43, align 4
  %176 = load i32, i32* %19, align 4
  %177 = sub nsw i32 %176, 1
  %178 = icmp sgt i32 %175, %177
  br i1 %178, label %179, label %185

179:                                              ; preds = %173
  %180 = load i32, i32* %43, align 4
  %181 = load i32, i32* %19, align 4
  %182 = sub nsw i32 %180, %181
  %183 = add nsw i32 %182, 1
  %184 = sub nsw i32 15, %183
  br label %186

185:                                              ; preds = %173
  br label %186

186:                                              ; preds = %185, %179
  %187 = phi i32 [ %184, %179 ], [ 15, %185 ]
  store i32 %187, i32* %52, align 4
  %188 = load i32, i32* %37, align 4
  %189 = sub nsw i32 %188, 1
  store i32 %189, i32* %53, align 4
  %190 = load i32, i32* %37, align 4
  %191 = add nsw i32 %190, 1
  store i32 %191, i32* %54, align 4
  %192 = load i32, i32* %36, align 4
  %193 = sub nsw i32 %192, 1
  store i32 %193, i32* %55, align 4
  %194 = load i32, i32* %36, align 4
  %195 = add nsw i32 %194, 1
  store i32 %195, i32* %56, align 4
  %196 = load i32, i32* %53, align 4
  %197 = load i32, i32* %49, align 4
  %198 = icmp slt i32 %196, %197
  br i1 %198, label %199, label %201

199:                                              ; preds = %186
  %200 = load i32, i32* %49, align 4
  br label %203

201:                                              ; preds = %186
  %202 = load i32, i32* %53, align 4
  br label %203

203:                                              ; preds = %201, %199
  %204 = phi i32 [ %200, %199 ], [ %202, %201 ]
  store i32 %204, i32* %53, align 4
  %205 = load i32, i32* %54, align 4
  %206 = load i32, i32* %50, align 4
  %207 = icmp sgt i32 %205, %206
  br i1 %207, label %208, label %210

208:                                              ; preds = %203
  %209 = load i32, i32* %50, align 4
  br label %212

210:                                              ; preds = %203
  %211 = load i32, i32* %54, align 4
  br label %212

212:                                              ; preds = %210, %208
  %213 = phi i32 [ %209, %208 ], [ %211, %210 ]
  store i32 %213, i32* %54, align 4
  %214 = load i32, i32* %55, align 4
  %215 = load i32, i32* %51, align 4
  %216 = icmp slt i32 %214, %215
  br i1 %216, label %217, label %219

217:                                              ; preds = %212
  %218 = load i32, i32* %51, align 4
  br label %221

219:                                              ; preds = %212
  %220 = load i32, i32* %55, align 4
  br label %221

221:                                              ; preds = %219, %217
  %222 = phi i32 [ %218, %217 ], [ %220, %219 ]
  store i32 %222, i32* %55, align 4
  %223 = load i32, i32* %56, align 4
  %224 = load i32, i32* %52, align 4
  %225 = icmp sgt i32 %223, %224
  br i1 %225, label %226, label %228

226:                                              ; preds = %221
  %227 = load i32, i32* %52, align 4
  br label %230

228:                                              ; preds = %221
  %229 = load i32, i32* %56, align 4
  br label %230

230:                                              ; preds = %228, %226
  %231 = phi i32 [ %227, %226 ], [ %229, %228 ]
  store i32 %231, i32* %56, align 4
  store i32 0, i32* %58, align 4
  br label %232

232:                                              ; preds = %399, %230
  %233 = load i32, i32* %58, align 4
  %234 = load i32, i32* %15, align 4
  %235 = icmp slt i32 %233, %234
  br i1 %235, label %236, label %402

236:                                              ; preds = %232
  store i8 0, i8* %57, align 1
  %237 = load i32, i32* %36, align 4
  %238 = load i32, i32* %58, align 4
  %239 = add nsw i32 %238, 1
  %240 = icmp sge i32 %237, %239
  br i1 %240, label %241, label %375

241:                                              ; preds = %236
  %242 = load i32, i32* %36, align 4
  %243 = load i32, i32* %58, align 4
  %244 = sub nsw i32 16, %243
  %245 = sub nsw i32 %244, 2
  %246 = icmp sle i32 %242, %245
  br i1 %246, label %247, label %375

247:                                              ; preds = %241
  %248 = load i32, i32* %37, align 4
  %249 = load i32, i32* %58, align 4
  %250 = add nsw i32 %249, 1
  %251 = icmp sge i32 %248, %250
  br i1 %251, label %252, label %375

252:                                              ; preds = %247
  %253 = load i32, i32* %37, align 4
  %254 = load i32, i32* %58, align 4
  %255 = sub nsw i32 16, %254
  %256 = sub nsw i32 %255, 2
  %257 = icmp sle i32 %253, %256
  br i1 %257, label %258, label %375

258:                                              ; preds = %252
  %259 = load i32, i32* %36, align 4
  %260 = load i32, i32* %51, align 4
  %261 = icmp sge i32 %259, %260
  br i1 %261, label %262, label %375

262:                                              ; preds = %258
  %263 = load i32, i32* %36, align 4
  %264 = load i32, i32* %52, align 4
  %265 = icmp sle i32 %263, %264
  br i1 %265, label %266, label %375

266:                                              ; preds = %262
  %267 = load i32, i32* %37, align 4
  %268 = load i32, i32* %49, align 4
  %269 = icmp sge i32 %267, %268
  br i1 %269, label %270, label %375

270:                                              ; preds = %266
  %271 = load i32, i32* %37, align 4
  %272 = load i32, i32* %50, align 4
  %273 = icmp sle i32 %271, %272
  br i1 %273, label %274, label %375

274:                                              ; preds = %270
  store i8 1, i8* %57, align 1
  %275 = load i32, i32* %37, align 4
  %276 = sext i32 %275 to i64
  %277 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %276
  %278 = load i32, i32* %36, align 4
  %279 = sext i32 %278 to i64
  %280 = getelementptr inbounds [16 x float], [16 x float]* %277, i64 0, i64 %279
  %281 = load float, float* %280, align 4
  %282 = fpext float %281 to double
  %283 = load float, float* %30, align 4
  %284 = fpext float %283 to double
  %285 = load i32, i32* %37, align 4
  %286 = sext i32 %285 to i64
  %287 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE13power_on_cuda to [16 x [16 x float]]*), i64 0, i64 %286
  %288 = load i32, i32* %36, align 4
  %289 = sext i32 %288 to i64
  %290 = getelementptr inbounds [16 x float], [16 x float]* %287, i64 0, i64 %289
  %291 = load float, float* %290, align 4
  %292 = fpext float %291 to double
  %293 = load i32, i32* %54, align 4
  %294 = sext i32 %293 to i64
  %295 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %294
  %296 = load i32, i32* %36, align 4
  %297 = sext i32 %296 to i64
  %298 = getelementptr inbounds [16 x float], [16 x float]* %295, i64 0, i64 %297
  %299 = load float, float* %298, align 4
  %300 = load i32, i32* %53, align 4
  %301 = sext i32 %300 to i64
  %302 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %301
  %303 = load i32, i32* %36, align 4
  %304 = sext i32 %303 to i64
  %305 = getelementptr inbounds [16 x float], [16 x float]* %302, i64 0, i64 %304
  %306 = load float, float* %305, align 4
  %307 = fadd contract float %299, %306
  %308 = fpext float %307 to double
  %309 = load i32, i32* %37, align 4
  %310 = sext i32 %309 to i64
  %311 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %310
  %312 = load i32, i32* %36, align 4
  %313 = sext i32 %312 to i64
  %314 = getelementptr inbounds [16 x float], [16 x float]* %311, i64 0, i64 %313
  %315 = load float, float* %314, align 4
  %316 = fpext float %315 to double
  %317 = fmul contract double 2.000000e+00, %316
  %318 = fsub contract double %308, %317
  %319 = load float, float* %32, align 4
  %320 = fpext float %319 to double
  %321 = fmul contract double %318, %320
  %322 = fadd contract double %292, %321
  %323 = load i32, i32* %37, align 4
  %324 = sext i32 %323 to i64
  %325 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %324
  %326 = load i32, i32* %56, align 4
  %327 = sext i32 %326 to i64
  %328 = getelementptr inbounds [16 x float], [16 x float]* %325, i64 0, i64 %327
  %329 = load float, float* %328, align 4
  %330 = load i32, i32* %37, align 4
  %331 = sext i32 %330 to i64
  %332 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %331
  %333 = load i32, i32* %55, align 4
  %334 = sext i32 %333 to i64
  %335 = getelementptr inbounds [16 x float], [16 x float]* %332, i64 0, i64 %334
  %336 = load float, float* %335, align 4
  %337 = fadd contract float %329, %336
  %338 = fpext float %337 to double
  %339 = load i32, i32* %37, align 4
  %340 = sext i32 %339 to i64
  %341 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %340
  %342 = load i32, i32* %36, align 4
  %343 = sext i32 %342 to i64
  %344 = getelementptr inbounds [16 x float], [16 x float]* %341, i64 0, i64 %343
  %345 = load float, float* %344, align 4
  %346 = fpext float %345 to double
  %347 = fmul contract double 2.000000e+00, %346
  %348 = fsub contract double %338, %347
  %349 = load float, float* %31, align 4
  %350 = fpext float %349 to double
  %351 = fmul contract double %348, %350
  %352 = fadd contract double %322, %351
  %353 = load float, float* %29, align 4
  %354 = load i32, i32* %37, align 4
  %355 = sext i32 %354 to i64
  %356 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %355
  %357 = load i32, i32* %36, align 4
  %358 = sext i32 %357 to i64
  %359 = getelementptr inbounds [16 x float], [16 x float]* %356, i64 0, i64 %358
  %360 = load float, float* %359, align 4
  %361 = fsub contract float %353, %360
  %362 = load float, float* %33, align 4
  %363 = fmul contract float %361, %362
  %364 = fpext float %363 to double
  %365 = fadd contract double %352, %364
  %366 = fmul contract double %284, %365
  %367 = fadd contract double %282, %366
  %368 = fptrunc double %367 to float
  %369 = load i32, i32* %37, align 4
  %370 = sext i32 %369 to i64
  %371 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t to [16 x [16 x float]]*), i64 0, i64 %370
  %372 = load i32, i32* %36, align 4
  %373 = sext i32 %372 to i64
  %374 = getelementptr inbounds [16 x float], [16 x float]* %371, i64 0, i64 %373
  store float %368, float* %374, align 4
  br label %375

375:                                              ; preds = %274, %270, %266, %262, %258, %252, %247, %241, %236
  call void @llvm.nvvm.barrier0()
  %376 = load i32, i32* %58, align 4
  %377 = load i32, i32* %15, align 4
  %378 = sub nsw i32 %377, 1
  %379 = icmp eq i32 %376, %378
  br i1 %379, label %380, label %381

380:                                              ; preds = %375
  br label %402

381:                                              ; preds = %375
  %382 = load i8, i8* %57, align 1
  %383 = trunc i8 %382 to i1
  br i1 %383, label %384, label %398

384:                                              ; preds = %381
  %385 = load i32, i32* %37, align 4
  %386 = sext i32 %385 to i64
  %387 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t to [16 x [16 x float]]*), i64 0, i64 %386
  %388 = load i32, i32* %36, align 4
  %389 = sext i32 %388 to i64
  %390 = getelementptr inbounds [16 x float], [16 x float]* %387, i64 0, i64 %389
  %391 = load float, float* %390, align 4
  %392 = load i32, i32* %37, align 4
  %393 = sext i32 %392 to i64
  %394 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %393
  %395 = load i32, i32* %36, align 4
  %396 = sext i32 %395 to i64
  %397 = getelementptr inbounds [16 x float], [16 x float]* %394, i64 0, i64 %396
  store float %391, float* %397, align 4
  br label %398

398:                                              ; preds = %384, %381
  call void @llvm.nvvm.barrier0()
  br label %399

399:                                              ; preds = %398
  %400 = load i32, i32* %58, align 4
  %401 = add nsw i32 %400, 1
  store i32 %401, i32* %58, align 4
  br label %232

402:                                              ; preds = %380, %232
  %403 = load i8, i8* %57, align 1
  %404 = trunc i8 %403 to i1
  br i1 %404, label %405, label %417

405:                                              ; preds = %402
  %406 = load i32, i32* %37, align 4
  %407 = sext i32 %406 to i64
  %408 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ22calculate_temp_checkediPfS_S_iiiiffffffE6temp_t to [16 x [16 x float]]*), i64 0, i64 %407
  %409 = load i32, i32* %36, align 4
  %410 = sext i32 %409 to i64
  %411 = getelementptr inbounds [16 x float], [16 x float]* %408, i64 0, i64 %410
  %412 = load float, float* %411, align 4
  %413 = load float*, float** %18, align 8
  %414 = load i32, i32* %48, align 4
  %415 = sext i32 %414 to i64
  %416 = getelementptr inbounds float, float* %413, i64 %415
  store float %412, float* %416, align 4
  br label %417

417:                                              ; preds = %405, %402
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z24calculate_temp_uncheckediPfS_S_iiiiffffff(i32, float*, float*, float*, i32, i32, i32, i32, float, float, float, float, float, float) #0 {
  %15 = alloca i32, align 4
  %16 = alloca float*, align 8
  %17 = alloca float*, align 8
  %18 = alloca float*, align 8
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca i32, align 4
  %22 = alloca i32, align 4
  %23 = alloca float, align 4
  %24 = alloca float, align 4
  %25 = alloca float, align 4
  %26 = alloca float, align 4
  %27 = alloca float, align 4
  %28 = alloca float, align 4
  %29 = alloca float, align 4
  %30 = alloca float, align 4
  %31 = alloca float, align 4
  %32 = alloca float, align 4
  %33 = alloca float, align 4
  %34 = alloca i32, align 4
  %35 = alloca i32, align 4
  %36 = alloca i32, align 4
  %37 = alloca i32, align 4
  %38 = alloca i32, align 4
  %39 = alloca i32, align 4
  %40 = alloca i32, align 4
  %41 = alloca i32, align 4
  %42 = alloca i32, align 4
  %43 = alloca i32, align 4
  %44 = alloca i32, align 4
  %45 = alloca i32, align 4
  %46 = alloca i32, align 4
  %47 = alloca i32, align 4
  %48 = alloca i32, align 4
  %49 = alloca i32, align 4
  %50 = alloca i32, align 4
  %51 = alloca i32, align 4
  %52 = alloca i32, align 4
  %53 = alloca i32, align 4
  %54 = alloca i32, align 4
  %55 = alloca i32, align 4
  %56 = alloca i32, align 4
  %57 = alloca i8, align 1
  %58 = alloca i32, align 4
  store i32 %0, i32* %15, align 4
  store float* %1, float** %16, align 8
  store float* %2, float** %17, align 8
  store float* %3, float** %18, align 8
  store i32 %4, i32* %19, align 4
  store i32 %5, i32* %20, align 4
  store i32 %6, i32* %21, align 4
  store i32 %7, i32* %22, align 4
  store float %8, float* %23, align 4
  store float %9, float* %24, align 4
  store float %10, float* %25, align 4
  store float %11, float* %26, align 4
  store float %12, float* %27, align 4
  store float %13, float* %28, align 4
  store float 8.000000e+01, float* %29, align 4
  %59 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !11
  store i32 %59, i32* %34, align 4
  %60 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !12
  store i32 %60, i32* %35, align 4
  %61 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  store i32 %61, i32* %36, align 4
  %62 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !13
  store i32 %62, i32* %37, align 4
  %63 = load float, float* %27, align 4
  %64 = load float, float* %23, align 4
  %65 = fdiv float %63, %64
  store float %65, float* %30, align 4
  %66 = load float, float* %24, align 4
  %67 = fdiv float 1.000000e+00, %66
  store float %67, float* %31, align 4
  %68 = load float, float* %25, align 4
  %69 = fdiv float 1.000000e+00, %68
  store float %69, float* %32, align 4
  %70 = load float, float* %26, align 4
  %71 = fdiv float 1.000000e+00, %70
  store float %71, float* %33, align 4
  %72 = load i32, i32* %15, align 4
  %73 = mul nsw i32 %72, 2
  %74 = sub nsw i32 16, %73
  store i32 %74, i32* %38, align 4
  %75 = load i32, i32* %15, align 4
  %76 = mul nsw i32 %75, 2
  %77 = sub nsw i32 16, %76
  store i32 %77, i32* %39, align 4
  %78 = load i32, i32* %38, align 4
  %79 = load i32, i32* %35, align 4
  %80 = mul nsw i32 %78, %79
  %81 = load i32, i32* %22, align 4
  %82 = sub nsw i32 %80, %81
  store i32 %82, i32* %40, align 4
  %83 = load i32, i32* %39, align 4
  %84 = load i32, i32* %34, align 4
  %85 = mul nsw i32 %83, %84
  %86 = load i32, i32* %21, align 4
  %87 = sub nsw i32 %85, %86
  store i32 %87, i32* %41, align 4
  %88 = load i32, i32* %40, align 4
  %89 = add nsw i32 %88, 16
  %90 = sub nsw i32 %89, 1
  store i32 %90, i32* %42, align 4
  %91 = load i32, i32* %41, align 4
  %92 = add nsw i32 %91, 16
  %93 = sub nsw i32 %92, 1
  store i32 %93, i32* %43, align 4
  %94 = load i32, i32* %40, align 4
  %95 = load i32, i32* %37, align 4
  %96 = add nsw i32 %94, %95
  store i32 %96, i32* %44, align 4
  %97 = load i32, i32* %41, align 4
  %98 = load i32, i32* %36, align 4
  %99 = add nsw i32 %97, %98
  store i32 %99, i32* %45, align 4
  %100 = load i32, i32* %44, align 4
  store i32 %100, i32* %46, align 4
  %101 = load i32, i32* %45, align 4
  store i32 %101, i32* %47, align 4
  %102 = load i32, i32* %19, align 4
  %103 = load i32, i32* %46, align 4
  %104 = mul nsw i32 %102, %103
  %105 = load i32, i32* %47, align 4
  %106 = add nsw i32 %104, %105
  store i32 %106, i32* %48, align 4
  %107 = load float*, float** %17, align 8
  %108 = load i32, i32* %48, align 4
  %109 = sext i32 %108 to i64
  %110 = getelementptr inbounds float, float* %107, i64 %109
  %111 = load float, float* %110, align 4
  %112 = load i32, i32* %37, align 4
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %113
  %115 = load i32, i32* %36, align 4
  %116 = sext i32 %115 to i64
  %117 = getelementptr inbounds [16 x float], [16 x float]* %114, i64 0, i64 %116
  store float %111, float* %117, align 4
  %118 = load float*, float** %16, align 8
  %119 = load i32, i32* %48, align 4
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds float, float* %118, i64 %120
  %122 = load float, float* %121, align 4
  %123 = load i32, i32* %37, align 4
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE13power_on_cuda to [16 x [16 x float]]*), i64 0, i64 %124
  %126 = load i32, i32* %36, align 4
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds [16 x float], [16 x float]* %125, i64 0, i64 %127
  store float %122, float* %128, align 4
  call void @llvm.nvvm.barrier0()
  %129 = load i32, i32* %40, align 4
  %130 = icmp slt i32 %129, 0
  br i1 %130, label %131, label %134

131:                                              ; preds = %14
  %132 = load i32, i32* %40, align 4
  %133 = sub nsw i32 0, %132
  br label %135

134:                                              ; preds = %14
  br label %135

135:                                              ; preds = %134, %131
  %136 = phi i32 [ %133, %131 ], [ 0, %134 ]
  store i32 %136, i32* %49, align 4
  %137 = load i32, i32* %42, align 4
  %138 = load i32, i32* %20, align 4
  %139 = sub nsw i32 %138, 1
  %140 = icmp sgt i32 %137, %139
  br i1 %140, label %141, label %147

141:                                              ; preds = %135
  %142 = load i32, i32* %42, align 4
  %143 = load i32, i32* %20, align 4
  %144 = sub nsw i32 %142, %143
  %145 = add nsw i32 %144, 1
  %146 = sub nsw i32 15, %145
  br label %148

147:                                              ; preds = %135
  br label %148

148:                                              ; preds = %147, %141
  %149 = phi i32 [ %146, %141 ], [ 15, %147 ]
  store i32 %149, i32* %50, align 4
  %150 = load i32, i32* %41, align 4
  %151 = icmp slt i32 %150, 0
  br i1 %151, label %152, label %155

152:                                              ; preds = %148
  %153 = load i32, i32* %41, align 4
  %154 = sub nsw i32 0, %153
  br label %156

155:                                              ; preds = %148
  br label %156

156:                                              ; preds = %155, %152
  %157 = phi i32 [ %154, %152 ], [ 0, %155 ]
  store i32 %157, i32* %51, align 4
  %158 = load i32, i32* %43, align 4
  %159 = load i32, i32* %19, align 4
  %160 = sub nsw i32 %159, 1
  %161 = icmp sgt i32 %158, %160
  br i1 %161, label %162, label %168

162:                                              ; preds = %156
  %163 = load i32, i32* %43, align 4
  %164 = load i32, i32* %19, align 4
  %165 = sub nsw i32 %163, %164
  %166 = add nsw i32 %165, 1
  %167 = sub nsw i32 15, %166
  br label %169

168:                                              ; preds = %156
  br label %169

169:                                              ; preds = %168, %162
  %170 = phi i32 [ %167, %162 ], [ 15, %168 ]
  store i32 %170, i32* %52, align 4
  %171 = load i32, i32* %37, align 4
  %172 = sub nsw i32 %171, 1
  store i32 %172, i32* %53, align 4
  %173 = load i32, i32* %37, align 4
  %174 = add nsw i32 %173, 1
  store i32 %174, i32* %54, align 4
  %175 = load i32, i32* %36, align 4
  %176 = sub nsw i32 %175, 1
  store i32 %176, i32* %55, align 4
  %177 = load i32, i32* %36, align 4
  %178 = add nsw i32 %177, 1
  store i32 %178, i32* %56, align 4
  %179 = load i32, i32* %53, align 4
  %180 = load i32, i32* %49, align 4
  %181 = icmp slt i32 %179, %180
  br i1 %181, label %182, label %184

182:                                              ; preds = %169
  %183 = load i32, i32* %49, align 4
  br label %186

184:                                              ; preds = %169
  %185 = load i32, i32* %53, align 4
  br label %186

186:                                              ; preds = %184, %182
  %187 = phi i32 [ %183, %182 ], [ %185, %184 ]
  store i32 %187, i32* %53, align 4
  %188 = load i32, i32* %54, align 4
  %189 = load i32, i32* %50, align 4
  %190 = icmp sgt i32 %188, %189
  br i1 %190, label %191, label %193

191:                                              ; preds = %186
  %192 = load i32, i32* %50, align 4
  br label %195

193:                                              ; preds = %186
  %194 = load i32, i32* %54, align 4
  br label %195

195:                                              ; preds = %193, %191
  %196 = phi i32 [ %192, %191 ], [ %194, %193 ]
  store i32 %196, i32* %54, align 4
  %197 = load i32, i32* %55, align 4
  %198 = load i32, i32* %51, align 4
  %199 = icmp slt i32 %197, %198
  br i1 %199, label %200, label %202

200:                                              ; preds = %195
  %201 = load i32, i32* %51, align 4
  br label %204

202:                                              ; preds = %195
  %203 = load i32, i32* %55, align 4
  br label %204

204:                                              ; preds = %202, %200
  %205 = phi i32 [ %201, %200 ], [ %203, %202 ]
  store i32 %205, i32* %55, align 4
  %206 = load i32, i32* %56, align 4
  %207 = load i32, i32* %52, align 4
  %208 = icmp sgt i32 %206, %207
  br i1 %208, label %209, label %211

209:                                              ; preds = %204
  %210 = load i32, i32* %52, align 4
  br label %213

211:                                              ; preds = %204
  %212 = load i32, i32* %56, align 4
  br label %213

213:                                              ; preds = %211, %209
  %214 = phi i32 [ %210, %209 ], [ %212, %211 ]
  store i32 %214, i32* %56, align 4
  store i32 0, i32* %58, align 4
  br label %215

215:                                              ; preds = %382, %213
  %216 = load i32, i32* %58, align 4
  %217 = load i32, i32* %15, align 4
  %218 = icmp slt i32 %216, %217
  br i1 %218, label %219, label %385

219:                                              ; preds = %215
  store i8 0, i8* %57, align 1
  %220 = load i32, i32* %36, align 4
  %221 = load i32, i32* %58, align 4
  %222 = add nsw i32 %221, 1
  %223 = icmp sge i32 %220, %222
  br i1 %223, label %224, label %358

224:                                              ; preds = %219
  %225 = load i32, i32* %36, align 4
  %226 = load i32, i32* %58, align 4
  %227 = sub nsw i32 16, %226
  %228 = sub nsw i32 %227, 2
  %229 = icmp sle i32 %225, %228
  br i1 %229, label %230, label %358

230:                                              ; preds = %224
  %231 = load i32, i32* %37, align 4
  %232 = load i32, i32* %58, align 4
  %233 = add nsw i32 %232, 1
  %234 = icmp sge i32 %231, %233
  br i1 %234, label %235, label %358

235:                                              ; preds = %230
  %236 = load i32, i32* %37, align 4
  %237 = load i32, i32* %58, align 4
  %238 = sub nsw i32 16, %237
  %239 = sub nsw i32 %238, 2
  %240 = icmp sle i32 %236, %239
  br i1 %240, label %241, label %358

241:                                              ; preds = %235
  %242 = load i32, i32* %36, align 4
  %243 = load i32, i32* %51, align 4
  %244 = icmp sge i32 %242, %243
  br i1 %244, label %245, label %358

245:                                              ; preds = %241
  %246 = load i32, i32* %36, align 4
  %247 = load i32, i32* %52, align 4
  %248 = icmp sle i32 %246, %247
  br i1 %248, label %249, label %358

249:                                              ; preds = %245
  %250 = load i32, i32* %37, align 4
  %251 = load i32, i32* %49, align 4
  %252 = icmp sge i32 %250, %251
  br i1 %252, label %253, label %358

253:                                              ; preds = %249
  %254 = load i32, i32* %37, align 4
  %255 = load i32, i32* %50, align 4
  %256 = icmp sle i32 %254, %255
  br i1 %256, label %257, label %358

257:                                              ; preds = %253
  store i8 1, i8* %57, align 1
  %258 = load i32, i32* %37, align 4
  %259 = sext i32 %258 to i64
  %260 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %259
  %261 = load i32, i32* %36, align 4
  %262 = sext i32 %261 to i64
  %263 = getelementptr inbounds [16 x float], [16 x float]* %260, i64 0, i64 %262
  %264 = load float, float* %263, align 4
  %265 = fpext float %264 to double
  %266 = load float, float* %30, align 4
  %267 = fpext float %266 to double
  %268 = load i32, i32* %37, align 4
  %269 = sext i32 %268 to i64
  %270 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE13power_on_cuda to [16 x [16 x float]]*), i64 0, i64 %269
  %271 = load i32, i32* %36, align 4
  %272 = sext i32 %271 to i64
  %273 = getelementptr inbounds [16 x float], [16 x float]* %270, i64 0, i64 %272
  %274 = load float, float* %273, align 4
  %275 = fpext float %274 to double
  %276 = load i32, i32* %54, align 4
  %277 = sext i32 %276 to i64
  %278 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %277
  %279 = load i32, i32* %36, align 4
  %280 = sext i32 %279 to i64
  %281 = getelementptr inbounds [16 x float], [16 x float]* %278, i64 0, i64 %280
  %282 = load float, float* %281, align 4
  %283 = load i32, i32* %53, align 4
  %284 = sext i32 %283 to i64
  %285 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %284
  %286 = load i32, i32* %36, align 4
  %287 = sext i32 %286 to i64
  %288 = getelementptr inbounds [16 x float], [16 x float]* %285, i64 0, i64 %287
  %289 = load float, float* %288, align 4
  %290 = fadd contract float %282, %289
  %291 = fpext float %290 to double
  %292 = load i32, i32* %37, align 4
  %293 = sext i32 %292 to i64
  %294 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %293
  %295 = load i32, i32* %36, align 4
  %296 = sext i32 %295 to i64
  %297 = getelementptr inbounds [16 x float], [16 x float]* %294, i64 0, i64 %296
  %298 = load float, float* %297, align 4
  %299 = fpext float %298 to double
  %300 = fmul contract double 2.000000e+00, %299
  %301 = fsub contract double %291, %300
  %302 = load float, float* %32, align 4
  %303 = fpext float %302 to double
  %304 = fmul contract double %301, %303
  %305 = fadd contract double %275, %304
  %306 = load i32, i32* %37, align 4
  %307 = sext i32 %306 to i64
  %308 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %307
  %309 = load i32, i32* %56, align 4
  %310 = sext i32 %309 to i64
  %311 = getelementptr inbounds [16 x float], [16 x float]* %308, i64 0, i64 %310
  %312 = load float, float* %311, align 4
  %313 = load i32, i32* %37, align 4
  %314 = sext i32 %313 to i64
  %315 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %314
  %316 = load i32, i32* %55, align 4
  %317 = sext i32 %316 to i64
  %318 = getelementptr inbounds [16 x float], [16 x float]* %315, i64 0, i64 %317
  %319 = load float, float* %318, align 4
  %320 = fadd contract float %312, %319
  %321 = fpext float %320 to double
  %322 = load i32, i32* %37, align 4
  %323 = sext i32 %322 to i64
  %324 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %323
  %325 = load i32, i32* %36, align 4
  %326 = sext i32 %325 to i64
  %327 = getelementptr inbounds [16 x float], [16 x float]* %324, i64 0, i64 %326
  %328 = load float, float* %327, align 4
  %329 = fpext float %328 to double
  %330 = fmul contract double 2.000000e+00, %329
  %331 = fsub contract double %321, %330
  %332 = load float, float* %31, align 4
  %333 = fpext float %332 to double
  %334 = fmul contract double %331, %333
  %335 = fadd contract double %305, %334
  %336 = load float, float* %29, align 4
  %337 = load i32, i32* %37, align 4
  %338 = sext i32 %337 to i64
  %339 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %338
  %340 = load i32, i32* %36, align 4
  %341 = sext i32 %340 to i64
  %342 = getelementptr inbounds [16 x float], [16 x float]* %339, i64 0, i64 %341
  %343 = load float, float* %342, align 4
  %344 = fsub contract float %336, %343
  %345 = load float, float* %33, align 4
  %346 = fmul contract float %344, %345
  %347 = fpext float %346 to double
  %348 = fadd contract double %335, %347
  %349 = fmul contract double %267, %348
  %350 = fadd contract double %265, %349
  %351 = fptrunc double %350 to float
  %352 = load i32, i32* %37, align 4
  %353 = sext i32 %352 to i64
  %354 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t to [16 x [16 x float]]*), i64 0, i64 %353
  %355 = load i32, i32* %36, align 4
  %356 = sext i32 %355 to i64
  %357 = getelementptr inbounds [16 x float], [16 x float]* %354, i64 0, i64 %356
  store float %351, float* %357, align 4
  br label %358

358:                                              ; preds = %257, %253, %249, %245, %241, %235, %230, %224, %219
  call void @llvm.nvvm.barrier0()
  %359 = load i32, i32* %58, align 4
  %360 = load i32, i32* %15, align 4
  %361 = sub nsw i32 %360, 1
  %362 = icmp eq i32 %359, %361
  br i1 %362, label %363, label %364

363:                                              ; preds = %358
  br label %385

364:                                              ; preds = %358
  %365 = load i8, i8* %57, align 1
  %366 = trunc i8 %365 to i1
  br i1 %366, label %367, label %381

367:                                              ; preds = %364
  %368 = load i32, i32* %37, align 4
  %369 = sext i32 %368 to i64
  %370 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t to [16 x [16 x float]]*), i64 0, i64 %369
  %371 = load i32, i32* %36, align 4
  %372 = sext i32 %371 to i64
  %373 = getelementptr inbounds [16 x float], [16 x float]* %370, i64 0, i64 %372
  %374 = load float, float* %373, align 4
  %375 = load i32, i32* %37, align 4
  %376 = sext i32 %375 to i64
  %377 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE12temp_on_cuda to [16 x [16 x float]]*), i64 0, i64 %376
  %378 = load i32, i32* %36, align 4
  %379 = sext i32 %378 to i64
  %380 = getelementptr inbounds [16 x float], [16 x float]* %377, i64 0, i64 %379
  store float %374, float* %380, align 4
  br label %381

381:                                              ; preds = %367, %364
  call void @llvm.nvvm.barrier0()
  br label %382

382:                                              ; preds = %381
  %383 = load i32, i32* %58, align 4
  %384 = add nsw i32 %383, 1
  store i32 %384, i32* %58, align 4
  br label %215

385:                                              ; preds = %363, %215
  %386 = load i8, i8* %57, align 1
  %387 = trunc i8 %386 to i1
  br i1 %387, label %388, label %400

388:                                              ; preds = %385
  %389 = load i32, i32* %37, align 4
  %390 = sext i32 %389 to i64
  %391 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]]* addrspacecast ([16 x [16 x float]] addrspace(3)* @_ZZ24calculate_temp_uncheckediPfS_S_iiiiffffffE6temp_t to [16 x [16 x float]]*), i64 0, i64 %390
  %392 = load i32, i32* %36, align 4
  %393 = sext i32 %392 to i64
  %394 = getelementptr inbounds [16 x float], [16 x float]* %391, i64 0, i64 %393
  %395 = load float, float* %394, align 4
  %396 = load float*, float** %18, align 8
  %397 = load i32, i32* %48, align 4
  %398 = sext i32 %397 to i64
  %399 = getelementptr inbounds float, float* %396, i64 %398
  store float %395, float* %399, align 4
  br label %400

400:                                              ; preds = %388, %385
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

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
