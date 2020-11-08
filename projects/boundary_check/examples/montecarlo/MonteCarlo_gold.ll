; ModuleID = 'MonteCarlo_gold.cpp'
source_filename = "MonteCarlo_gold.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.TOptionData = type { float, float, float, float, float }
%struct.TOptionValue = type { float, float }
%struct.curandGenerator_st = type opaque

$_Z5checkI12curandStatusEvT_PKcS3_i = comdat any

@.str = private unnamed_addr constant [59 x i8] c"curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT)\00", align 1
@.str.1 = private unnamed_addr constant [20 x i8] c"MonteCarlo_gold.cpp\00", align 1
@.str.2 = private unnamed_addr constant [46 x i8] c"curandSetPseudoRandomGeneratorSeed(gen, seed)\00", align 1
@.str.3 = private unnamed_addr constant [52 x i8] c"curandGenerateNormal(gen, samples, pathN, 0.0, 1.0)\00", align 1
@.str.4 = private unnamed_addr constant [28 x i8] c"curandDestroyGenerator(gen)\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.5 = private unnamed_addr constant [39 x i8] c"CUDA error at %s:%d code=%d(%s) \22%s\22 \0A\00", align 1
@.str.6 = private unnamed_addr constant [22 x i8] c"CURAND_STATUS_SUCCESS\00", align 1
@.str.7 = private unnamed_addr constant [31 x i8] c"CURAND_STATUS_VERSION_MISMATCH\00", align 1
@.str.8 = private unnamed_addr constant [30 x i8] c"CURAND_STATUS_NOT_INITIALIZED\00", align 1
@.str.9 = private unnamed_addr constant [32 x i8] c"CURAND_STATUS_ALLOCATION_FAILED\00", align 1
@.str.10 = private unnamed_addr constant [25 x i8] c"CURAND_STATUS_TYPE_ERROR\00", align 1
@.str.11 = private unnamed_addr constant [27 x i8] c"CURAND_STATUS_OUT_OF_RANGE\00", align 1
@.str.12 = private unnamed_addr constant [34 x i8] c"CURAND_STATUS_LENGTH_NOT_MULTIPLE\00", align 1
@.str.13 = private unnamed_addr constant [40 x i8] c"CURAND_STATUS_DOUBLE_PRECISION_REQUIRED\00", align 1
@.str.14 = private unnamed_addr constant [29 x i8] c"CURAND_STATUS_LAUNCH_FAILURE\00", align 1
@.str.15 = private unnamed_addr constant [34 x i8] c"CURAND_STATUS_PREEXISTING_FAILURE\00", align 1
@.str.16 = private unnamed_addr constant [36 x i8] c"CURAND_STATUS_INITIALIZATION_FAILED\00", align 1
@.str.17 = private unnamed_addr constant [28 x i8] c"CURAND_STATUS_ARCH_MISMATCH\00", align 1
@.str.18 = private unnamed_addr constant [29 x i8] c"CURAND_STATUS_INTERNAL_ERROR\00", align 1
@.str.19 = private unnamed_addr constant [10 x i8] c"<unknown>\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local double @_Z3CNDd(double) local_unnamed_addr #0 {
  %2 = tail call double @llvm.fabs.f64(double %0)
  %3 = fmul double %2, 0x3FCDA6711871100E
  %4 = fadd double %3, 1.000000e+00
  %5 = fdiv double 1.000000e+00, %4
  %6 = fmul double %0, -5.000000e-01
  %7 = fmul double %6, %0
  %8 = tail call double @exp(double %7) #9
  %9 = fmul double %8, 0x3FD9884533D43651
  %10 = fmul double %5, 0x3FF548CDD6F42943
  %11 = fadd double %10, 0xBFFD23DD4EF278D0
  %12 = fmul double %5, %11
  %13 = fadd double %12, 0x3FFC80EF025F5E68
  %14 = fmul double %5, %13
  %15 = fadd double %14, 0xBFD6D1F0E5A8325B
  %16 = fmul double %5, %15
  %17 = fadd double %16, 0x3FD470BF3A92F8EC
  %18 = fmul double %5, %17
  %19 = fmul double %18, %9
  %20 = fcmp ogt double %0, 0.000000e+00
  %21 = fsub double 1.000000e+00, %19
  %22 = select i1 %20, double %21, double %19
  ret double %22
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #2

; Function Attrs: nofree nounwind
declare dso_local double @exp(double) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @BlackScholesCall(float* nocapture dereferenceable(4), %struct.TOptionData* nocapture readonly byval(%struct.TOptionData) align 8) local_unnamed_addr #0 {
  %3 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 0
  %4 = load float, float* %3, align 8, !tbaa !2
  %5 = fpext float %4 to double
  %6 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 1
  %7 = load float, float* %6, align 4, !tbaa !7
  %8 = fpext float %7 to double
  %9 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 2
  %10 = load float, float* %9, align 8, !tbaa !8
  %11 = fpext float %10 to double
  %12 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 3
  %13 = load float, float* %12, align 4, !tbaa !9
  %14 = fpext float %13 to double
  %15 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 4
  %16 = load float, float* %15, align 8, !tbaa !10
  %17 = fpext float %16 to double
  %18 = tail call double @sqrt(double %11) #9
  %19 = fdiv double %5, %8
  %20 = tail call double @log(double %19) #9
  %21 = fmul double %17, 5.000000e-01
  %22 = fmul double %21, %17
  %23 = fadd double %22, %14
  %24 = fmul double %23, %11
  %25 = fadd double %20, %24
  %26 = fmul double %18, %17
  %27 = fdiv double %25, %26
  %28 = fsub double %27, %26
  %29 = tail call double @_Z3CNDd(double %27)
  %30 = tail call double @_Z3CNDd(double %28)
  %31 = fmul double %11, %14
  %32 = fsub double -0.000000e+00, %31
  %33 = tail call double @exp(double %32) #9
  %34 = fmul double %29, %5
  %35 = fmul double %33, %8
  %36 = fmul double %30, %35
  %37 = fsub double %34, %36
  %38 = fptrunc double %37 to float
  store float %38, float* %0, align 4, !tbaa !11
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local double @sqrt(double) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare dso_local double @log(double) local_unnamed_addr #3

; Function Attrs: uwtable
define dso_local void @MonteCarloCPU(%struct.TOptionValue* nocapture dereferenceable(8), %struct.TOptionData* nocapture readonly byval(%struct.TOptionData) align 8, float*, i32) local_unnamed_addr #4 {
  %5 = alloca %struct.curandGenerator_st*, align 8
  %6 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 0
  %7 = load float, float* %6, align 8, !tbaa !2
  %8 = fpext float %7 to double
  %9 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 1
  %10 = load float, float* %9, align 4, !tbaa !7
  %11 = fpext float %10 to double
  %12 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 2
  %13 = load float, float* %12, align 8, !tbaa !8
  %14 = fpext float %13 to double
  %15 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 3
  %16 = load float, float* %15, align 4, !tbaa !9
  %17 = fpext float %16 to double
  %18 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %1, i64 0, i32 4
  %19 = load float, float* %18, align 8, !tbaa !10
  %20 = fpext float %19 to double
  %21 = fmul double %20, 5.000000e-01
  %22 = fmul double %21, %20
  %23 = fsub double %17, %22
  %24 = fmul double %23, %14
  %25 = tail call double @sqrt(double %14) #9
  %26 = fmul double %25, %20
  %27 = bitcast %struct.curandGenerator_st** %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %27) #9
  %28 = call i32 @curandCreateGeneratorHost(%struct.curandGenerator_st** nonnull %5, i32 100)
  call void @_Z5checkI12curandStatusEvT_PKcS3_i(i32 %28, i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.1, i64 0, i64 0), i32 96)
  %29 = load %struct.curandGenerator_st*, %struct.curandGenerator_st** %5, align 8, !tbaa !12
  %30 = call i32 @curandSetPseudoRandomGeneratorSeed(%struct.curandGenerator_st* %29, i64 1234)
  call void @_Z5checkI12curandStatusEvT_PKcS3_i(i32 %30, i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.1, i64 0, i64 0), i32 98)
  %31 = icmp eq float* %2, null
  br i1 %31, label %32, label %39

32:                                               ; preds = %4
  %33 = sext i32 %3 to i64
  %34 = shl nsw i64 %33, 2
  %35 = call noalias i8* @malloc(i64 %34) #9
  %36 = bitcast i8* %35 to float*
  %37 = load %struct.curandGenerator_st*, %struct.curandGenerator_st** %5, align 8, !tbaa !12
  %38 = call i32 @curandGenerateNormal(%struct.curandGenerator_st* %37, float* %36, i64 %33, float 0.000000e+00, float 1.000000e+00)
  call void @_Z5checkI12curandStatusEvT_PKcS3_i(i32 %38, i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.1, i64 0, i64 0), i32 104)
  br label %39

39:                                               ; preds = %4, %32
  %40 = phi float* [ %36, %32 ], [ %2, %4 ]
  %41 = icmp sgt i32 %3, 0
  br i1 %41, label %42, label %44

42:                                               ; preds = %39
  %43 = zext i32 %3 to i64
  br label %47

44:                                               ; preds = %47, %39
  %45 = phi double [ 0.000000e+00, %39 ], [ %57, %47 ]
  %46 = phi double [ 0.000000e+00, %39 ], [ %55, %47 ]
  br i1 %31, label %60, label %62

47:                                               ; preds = %47, %42
  %48 = phi i64 [ 0, %42 ], [ %58, %47 ]
  %49 = phi double [ 0.000000e+00, %42 ], [ %55, %47 ]
  %50 = phi double [ 0.000000e+00, %42 ], [ %57, %47 ]
  %51 = getelementptr inbounds float, float* %40, i64 %48
  %52 = load float, float* %51, align 4, !tbaa !11
  %53 = fpext float %52 to double
  %54 = call fastcc double @_ZL12endCallValueddddd(double %8, double %11, double %53, double %24, double %26)
  %55 = fadd double %49, %54
  %56 = fmul double %54, %54
  %57 = fadd double %50, %56
  %58 = add nuw nsw i64 %48, 1
  %59 = icmp eq i64 %58, %43
  br i1 %59, label %44, label %47

60:                                               ; preds = %44
  %61 = bitcast float* %40 to i8*
  call void @free(i8* %61) #9
  br label %62

62:                                               ; preds = %60, %44
  %63 = load %struct.curandGenerator_st*, %struct.curandGenerator_st** %5, align 8, !tbaa !12
  %64 = call i32 @curandDestroyGenerator(%struct.curandGenerator_st* %63)
  call void @_Z5checkI12curandStatusEvT_PKcS3_i(i32 %64, i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.1, i64 0, i64 0), i32 122)
  %65 = fmul double %14, %17
  %66 = fsub double -0.000000e+00, %65
  %67 = call double @exp(double %66) #9
  %68 = fmul double %46, %67
  %69 = sitofp i32 %3 to double
  %70 = fdiv double %68, %69
  %71 = fptrunc double %70 to float
  %72 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %0, i64 0, i32 0
  store float %71, float* %72, align 4, !tbaa !14
  %73 = fmul double %45, %69
  %74 = fmul double %46, %46
  %75 = fsub double %73, %74
  %76 = add nsw i32 %3, -1
  %77 = sitofp i32 %76 to double
  %78 = fmul double %69, %77
  %79 = fdiv double %75, %78
  %80 = call double @sqrt(double %79) #9
  %81 = call double @exp(double %66) #9
  %82 = fmul double %81, 1.960000e+00
  %83 = fmul double %80, %82
  %84 = call double @sqrt(double %69) #9
  %85 = fdiv double %83, %84
  %86 = fptrunc double %85 to float
  %87 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %0, i64 0, i32 1
  store float %86, float* %87, align 4, !tbaa !16
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %27) #9
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_Z5checkI12curandStatusEvT_PKcS3_i(i32, i8*, i8*, i32) local_unnamed_addr #4 comdat {
  %5 = icmp eq i32 %0, 0
  br i1 %5, label %11, label %6

6:                                                ; preds = %4
  %7 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !12
  %8 = tail call fastcc i8* @_ZL17_cudaGetErrorEnum12curandStatus(i32 %0)
  %9 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %7, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.5, i64 0, i64 0), i8* %2, i32 %3, i32 %0, i8* %8, i8* %1) #10
  %10 = tail call i32 @cudaDeviceReset()
  tail call void @exit(i32 1) #11
  unreachable

11:                                               ; preds = %4
  ret void
}

declare dso_local i32 @curandCreateGeneratorHost(%struct.curandGenerator_st**, i32) local_unnamed_addr #5

declare dso_local i32 @curandSetPseudoRandomGeneratorSeed(%struct.curandGenerator_st*, i64) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #3

declare dso_local i32 @curandGenerateNormal(%struct.curandGenerator_st*, float*, i64, float, float) local_unnamed_addr #5

; Function Attrs: nofree nounwind uwtable
define internal fastcc double @_ZL12endCallValueddddd(double, double, double, double, double) unnamed_addr #0 {
  %6 = fmul double %2, %4
  %7 = fadd double %6, %3
  %8 = tail call double @exp(double %7) #9
  %9 = fmul double %8, %0
  %10 = fsub double %9, %1
  %11 = fcmp ogt double %10, 0.000000e+00
  %12 = select i1 %11, double %10, double 0.000000e+00
  ret double %12
}

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #6

declare dso_local i32 @curandDestroyGenerator(%struct.curandGenerator_st*) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: norecurse nounwind readnone uwtable
define internal fastcc i8* @_ZL17_cudaGetErrorEnum12curandStatus(i32) unnamed_addr #7 {
  switch i32 %0, label %14 [
    i32 0, label %15
    i32 100, label %2
    i32 101, label %3
    i32 102, label %4
    i32 103, label %5
    i32 104, label %6
    i32 105, label %7
    i32 106, label %8
    i32 201, label %9
    i32 202, label %10
    i32 203, label %11
    i32 204, label %12
    i32 999, label %13
  ]

2:                                                ; preds = %1
  br label %15

3:                                                ; preds = %1
  br label %15

4:                                                ; preds = %1
  br label %15

5:                                                ; preds = %1
  br label %15

6:                                                ; preds = %1
  br label %15

7:                                                ; preds = %1
  br label %15

8:                                                ; preds = %1
  br label %15

9:                                                ; preds = %1
  br label %15

10:                                               ; preds = %1
  br label %15

11:                                               ; preds = %1
  br label %15

12:                                               ; preds = %1
  br label %15

13:                                               ; preds = %1
  br label %15

14:                                               ; preds = %1
  br label %15

15:                                               ; preds = %1, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2
  %16 = phi i8* [ getelementptr inbounds ([10 x i8], [10 x i8]* @.str.19, i64 0, i64 0), %14 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.18, i64 0, i64 0), %13 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.17, i64 0, i64 0), %12 ], [ getelementptr inbounds ([36 x i8], [36 x i8]* @.str.16, i64 0, i64 0), %11 ], [ getelementptr inbounds ([34 x i8], [34 x i8]* @.str.15, i64 0, i64 0), %10 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.14, i64 0, i64 0), %9 ], [ getelementptr inbounds ([40 x i8], [40 x i8]* @.str.13, i64 0, i64 0), %8 ], [ getelementptr inbounds ([34 x i8], [34 x i8]* @.str.12, i64 0, i64 0), %7 ], [ getelementptr inbounds ([27 x i8], [27 x i8]* @.str.11, i64 0, i64 0), %6 ], [ getelementptr inbounds ([25 x i8], [25 x i8]* @.str.10, i64 0, i64 0), %5 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.9, i64 0, i64 0), %4 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.8, i64 0, i64 0), %3 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.7, i64 0, i64 0), %2 ], [ getelementptr inbounds ([22 x i8], [22 x i8]* @.str.6, i64 0, i64 0), %1 ]
  ret i8* %16
}

declare dso_local i32 @cudaDeviceReset() local_unnamed_addr #5

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #8

attributes #0 = { nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind }
attributes #10 = { cold }
attributes #11 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTS11TOptionData", !4, i64 0, !4, i64 4, !4, i64 8, !4, i64 12, !4, i64 16}
!4 = !{!"float", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!3, !4, i64 4}
!8 = !{!3, !4, i64 8}
!9 = !{!3, !4, i64 12}
!10 = !{!3, !4, i64 16}
!11 = !{!4, !4, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"any pointer", !5, i64 0}
!14 = !{!15, !4, i64 0}
!15 = !{!"_ZTS12TOptionValue", !4, i64 0, !4, i64 4}
!16 = !{!15, !4, i64 4}
