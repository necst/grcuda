; ModuleID = 'hotspot.cu'
source_filename = "hotspot.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

$_Z20check_array_equalityIfEiPT_S1_ifb = comdat any

$_ZNSt11char_traitsIcE6lengthEPKc = comdat any

$_ZStorSt12_Ios_IostateS_ = comdat any

$_ZSt13__check_facetISt5ctypeIcEERKT_PS3_ = comdat any

$_ZNKSt5ctypeIcE5widenEc = comdat any

$_ZSt3absf = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@t_chip = dso_local local_unnamed_addr global float 0x3F40624DE0000000, align 4
@chip_height = dso_local local_unnamed_addr global float 0x3F90624DE0000000, align 4
@chip_width = dso_local local_unnamed_addr global float 0x3F90624DE0000000, align 4
@amb_temp = dso_local local_unnamed_addr global float 8.000000e+01, align 4
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [11 x i8] c"error: %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.3 = private unnamed_addr constant [7 x i8] c"%d\09%g\0A\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.5 = private unnamed_addr constant [25 x i8] c"not enough lines in file\00", align 1
@.str.6 = private unnamed_addr constant [3 x i8] c"%f\00", align 1
@.str.7 = private unnamed_addr constant [20 x i8] c"invalid file format\00", align 1
@.str.8 = private unnamed_addr constant [100 x i8] c"Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\0A\00", align 1
@.str.9 = private unnamed_addr constant [78 x i8] c"\09<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\0A\00", align 1
@.str.10 = private unnamed_addr constant [53 x i8] c"\09<pyramid_height> - pyramid heigh(positive integer)\0A\00", align 1
@.str.11 = private unnamed_addr constant [38 x i8] c"\09<sim_time>   - number of iterations\0A\00", align 1
@.str.12 = private unnamed_addr constant [89 x i8] c"\09<temp_file>  - name of the file containing the initial temperature values of each cell\0A\00", align 1
@.str.13 = private unnamed_addr constant [86 x i8] c"\09<power_file> - name of the file containing the dissipated power values of each cell\0A\00", align 1
@.str.14 = private unnamed_addr constant [42 x i8] c"\09<output_file> - name of the output file\0A\00", align 1
@.str.15 = private unnamed_addr constant [29 x i8] c"WG size of kernel = %d X %d\0A\00", align 1
@_ZSt4cout = external dso_local global %"class.std::basic_ostream", align 8
@.str.16 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.17 = private unnamed_addr constant [26 x i8] c"unable to allocate memory\00", align 1
@.str.18 = private unnamed_addr constant [94 x i8] c"pyramidHeight: %d\0AgridSize: [%d, %d]\0Aborder:[%d, %d]\0AblockGrid:[%d, %d]\0AtargetBlock:[%d, %d]\0A\00", align 1
@_ZSt4cerr = external dso_local global %"class.std::basic_ostream", align 8
@.str.21 = private unnamed_addr constant [15 x i8] c"Cuda failure: \00", align 1
@.str.22 = private unnamed_addr constant [6 x i8] c" at: \00", align 1
@.str.23 = private unnamed_addr constant [11 x i8] c"hotspot.cu\00", align 1
@.str.25 = private unnamed_addr constant [19 x i8] c"number of errors: \00", align 1
@.str.26 = private unnamed_addr constant [6 x i8] c") X: \00", align 1
@.str.27 = private unnamed_addr constant [6 x i8] c", Y: \00", align 1
@.str.28 = private unnamed_addr constant [9 x i8] c", diff: \00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_hotspot.cu, i8* null }]
@str = private unnamed_addr constant [24 x i8] c"The file was not opened\00", align 1
@str.29 = private unnamed_addr constant [24 x i8] c"The file was not opened\00", align 1
@str.30 = private unnamed_addr constant [42 x i8] c"Start computing the transient temperature\00", align 1
@str.31 = private unnamed_addr constant [28 x i8] c"Ending simulation - Checked\00", align 1
@str.32 = private unnamed_addr constant [30 x i8] c"Ending simulation - Unchecked\00", align 1

; Function Attrs: uwtable
define internal fastcc void @__cxx_global_var_init() unnamed_addr #0 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #17
  ret void
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @_Z5fatalPKc(i8*) local_unnamed_addr #4 {
  %2 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %3 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %2, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* %0) #18
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: nounwind uwtable
define dso_local void @_Z11writeoutputPfiiPc(float* nocapture readonly, i32, i32, i8* nocapture readonly) local_unnamed_addr #6 {
  %5 = alloca [256 x i8], align 16
  %6 = getelementptr inbounds [256 x i8], [256 x i8]* %5, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 256, i8* nonnull %6) #17
  %7 = tail call %struct._IO_FILE* @fopen(i8* %3, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
  %8 = icmp eq %struct._IO_FILE* %7, null
  br i1 %8, label %9, label %11

9:                                                ; preds = %4
  %10 = tail call i32 @puts(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @str, i64 0, i64 0))
  br label %11

11:                                               ; preds = %9, %4
  %12 = icmp sgt i32 %1, 0
  br i1 %12, label %13, label %39

13:                                               ; preds = %11
  %14 = icmp sgt i32 %2, 0
  %15 = sext i32 %2 to i64
  %16 = zext i32 %1 to i64
  %17 = zext i32 %2 to i64
  br label %18

18:                                               ; preds = %35, %13
  %19 = phi i64 [ 0, %13 ], [ %37, %35 ]
  %20 = phi i32 [ 0, %13 ], [ %36, %35 ]
  br i1 %14, label %21, label %35

21:                                               ; preds = %18
  %22 = mul nsw i64 %19, %15
  br label %23

23:                                               ; preds = %23, %21
  %24 = phi i64 [ 0, %21 ], [ %33, %23 ]
  %25 = phi i32 [ %20, %21 ], [ %32, %23 ]
  %26 = add nsw i64 %24, %22
  %27 = getelementptr inbounds float, float* %0, i64 %26
  %28 = load float, float* %27, align 4, !tbaa !7
  %29 = fpext float %28 to double
  %30 = call i32 (i8*, i8*, ...) @sprintf(i8* nonnull %6, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.3, i64 0, i64 0), i32 %25, double %29) #17
  %31 = call i32 @fputs(i8* nonnull %6, %struct._IO_FILE* %7)
  %32 = add nsw i32 %25, 1
  %33 = add nuw nsw i64 %24, 1
  %34 = icmp eq i64 %33, %17
  br i1 %34, label %35, label %23

35:                                               ; preds = %23, %18
  %36 = phi i32 [ %20, %18 ], [ %32, %23 ]
  %37 = add nuw nsw i64 %19, 1
  %38 = icmp eq i64 %37, %16
  br i1 %38, label %39, label %18

39:                                               ; preds = %35, %11
  %40 = tail call i32 @fclose(%struct._IO_FILE* %7)
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %6) #17
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #7

; Function Attrs: nofree nounwind
declare dso_local noalias %struct._IO_FILE* @fopen(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local i32 @sprintf(i8* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local i32 @fputs(i8* nocapture readonly, %struct._IO_FILE* nocapture) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local i32 @fclose(%struct._IO_FILE* nocapture) local_unnamed_addr #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #7

; Function Attrs: nounwind uwtable
define dso_local void @_Z9readinputPfiiPc(float* nocapture, i32, i32, i8* nocapture readonly) local_unnamed_addr #6 {
  %5 = alloca [256 x i8], align 16
  %6 = alloca float, align 4
  %7 = getelementptr inbounds [256 x i8], [256 x i8]* %5, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 256, i8* nonnull %7) #17
  %8 = bitcast float* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %8) #17
  %9 = tail call %struct._IO_FILE* @fopen(i8* %3, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.4, i64 0, i64 0))
  %10 = icmp eq %struct._IO_FILE* %9, null
  br i1 %10, label %11, label %13

11:                                               ; preds = %4
  %12 = tail call i32 @puts(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @str.29, i64 0, i64 0))
  br label %13

13:                                               ; preds = %11, %4
  %14 = icmp sgt i32 %1, 0
  br i1 %14, label %15, label %45

15:                                               ; preds = %13
  %16 = icmp sgt i32 %2, 0
  %17 = bitcast float* %6 to i32*
  %18 = sext i32 %2 to i64
  %19 = zext i32 %1 to i64
  %20 = zext i32 %2 to i64
  br label %21

21:                                               ; preds = %42, %15
  %22 = phi i64 [ 0, %15 ], [ %43, %42 ]
  br i1 %16, label %23, label %42

23:                                               ; preds = %21
  %24 = mul nsw i64 %22, %18
  br label %25

25:                                               ; preds = %35, %23
  %26 = phi i64 [ 0, %23 ], [ %40, %35 ]
  %27 = call i8* @fgets_unlocked(i8* nonnull %7, i32 256, %struct._IO_FILE* %9)
  %28 = call i32 @feof(%struct._IO_FILE* %9) #17
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %31, label %30

30:                                               ; preds = %25
  call void @_Z5fatalPKc(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.5, i64 0, i64 0))
  br label %31

31:                                               ; preds = %25, %30
  %32 = call i32 (i8*, i8*, ...) @sscanf(i8* nonnull %7, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.6, i64 0, i64 0), float* nonnull %6) #17
  %33 = icmp eq i32 %32, 1
  br i1 %33, label %35, label %34

34:                                               ; preds = %31
  call void @_Z5fatalPKc(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.7, i64 0, i64 0))
  br label %35

35:                                               ; preds = %31, %34
  %36 = load i32, i32* %17, align 4, !tbaa !7
  %37 = add nsw i64 %26, %24
  %38 = getelementptr inbounds float, float* %0, i64 %37
  %39 = bitcast float* %38 to i32*
  store i32 %36, i32* %39, align 4, !tbaa !7
  %40 = add nuw nsw i64 %26, 1
  %41 = icmp eq i64 %40, %20
  br i1 %41, label %42, label %25

42:                                               ; preds = %35, %21
  %43 = add nuw nsw i64 %22, 1
  %44 = icmp eq i64 %43, %19
  br i1 %44, label %45, label %21

45:                                               ; preds = %42, %13
  %46 = call i32 @fclose(%struct._IO_FILE* %9)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %8) #17
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %7) #17
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local i32 @feof(%struct._IO_FILE* nocapture) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local i32 @sscanf(i8* nocapture readonly, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: uwtable
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
  %29 = alloca %struct.dim3, align 8
  %30 = alloca %struct.dim3, align 8
  %31 = alloca i64, align 8
  %32 = alloca i8*, align 8
  store i32 %0, i32* %15, align 4, !tbaa !9
  store float* %1, float** %16, align 8, !tbaa !3
  store float* %2, float** %17, align 8, !tbaa !3
  store float* %3, float** %18, align 8, !tbaa !3
  store i32 %4, i32* %19, align 4, !tbaa !9
  store i32 %5, i32* %20, align 4, !tbaa !9
  store i32 %6, i32* %21, align 4, !tbaa !9
  store i32 %7, i32* %22, align 4, !tbaa !9
  store float %8, float* %23, align 4, !tbaa !7
  store float %9, float* %24, align 4, !tbaa !7
  store float %10, float* %25, align 4, !tbaa !7
  store float %11, float* %26, align 4, !tbaa !7
  store float %12, float* %27, align 4, !tbaa !7
  store float %13, float* %28, align 4, !tbaa !7
  %33 = alloca [14 x i8*], align 16
  %34 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 0
  %35 = bitcast [14 x i8*]* %33 to i32**
  store i32* %15, i32** %35, align 16
  %36 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 1
  %37 = bitcast i8** %36 to float***
  store float** %16, float*** %37, align 8
  %38 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 2
  %39 = bitcast i8** %38 to float***
  store float** %17, float*** %39, align 16
  %40 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 3
  %41 = bitcast i8** %40 to float***
  store float** %18, float*** %41, align 8
  %42 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 4
  %43 = bitcast i8** %42 to i32**
  store i32* %19, i32** %43, align 16
  %44 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 5
  %45 = bitcast i8** %44 to i32**
  store i32* %20, i32** %45, align 8
  %46 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 6
  %47 = bitcast i8** %46 to i32**
  store i32* %21, i32** %47, align 16
  %48 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 7
  %49 = bitcast i8** %48 to i32**
  store i32* %22, i32** %49, align 8
  %50 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 8
  %51 = bitcast i8** %50 to float**
  store float* %23, float** %51, align 16
  %52 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 9
  %53 = bitcast i8** %52 to float**
  store float* %24, float** %53, align 8
  %54 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 10
  %55 = bitcast i8** %54 to float**
  store float* %25, float** %55, align 16
  %56 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 11
  %57 = bitcast i8** %56 to float**
  store float* %26, float** %57, align 8
  %58 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 12
  %59 = bitcast i8** %58 to float**
  store float* %27, float** %59, align 16
  %60 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 13
  %61 = bitcast i8** %60 to float**
  store float* %28, float** %61, align 8
  %62 = call i32 @__cudaPopCallConfiguration(%struct.dim3* nonnull %29, %struct.dim3* nonnull %30, i64* nonnull %31, i8** nonnull %32)
  %63 = load i64, i64* %31, align 8
  %64 = bitcast i8** %32 to %struct.CUstream_st**
  %65 = load %struct.CUstream_st*, %struct.CUstream_st** %64, align 8
  %66 = bitcast %struct.dim3* %29 to i64*
  %67 = load i64, i64* %66, align 8
  %68 = getelementptr inbounds %struct.dim3, %struct.dim3* %29, i64 0, i32 2
  %69 = load i32, i32* %68, align 8
  %70 = bitcast %struct.dim3* %30 to i64*
  %71 = load i64, i64* %70, align 8
  %72 = getelementptr inbounds %struct.dim3, %struct.dim3* %30, i64 0, i32 2
  %73 = load i32, i32* %72, align 8
  %74 = call i32 @cudaLaunchKernel(i8* bitcast (void (i32, float*, float*, float*, i32, i32, i32, i32, float, float, float, float, float, float)* @_Z22calculate_temp_checkediPfS_S_iiiiffffff to i8*), i64 %67, i32 %69, i64 %71, i32 %73, i8** nonnull %34, i64 %63, %struct.CUstream_st* %65)
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**) local_unnamed_addr

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*) local_unnamed_addr

; Function Attrs: uwtable
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
  %29 = alloca %struct.dim3, align 8
  %30 = alloca %struct.dim3, align 8
  %31 = alloca i64, align 8
  %32 = alloca i8*, align 8
  store i32 %0, i32* %15, align 4, !tbaa !9
  store float* %1, float** %16, align 8, !tbaa !3
  store float* %2, float** %17, align 8, !tbaa !3
  store float* %3, float** %18, align 8, !tbaa !3
  store i32 %4, i32* %19, align 4, !tbaa !9
  store i32 %5, i32* %20, align 4, !tbaa !9
  store i32 %6, i32* %21, align 4, !tbaa !9
  store i32 %7, i32* %22, align 4, !tbaa !9
  store float %8, float* %23, align 4, !tbaa !7
  store float %9, float* %24, align 4, !tbaa !7
  store float %10, float* %25, align 4, !tbaa !7
  store float %11, float* %26, align 4, !tbaa !7
  store float %12, float* %27, align 4, !tbaa !7
  store float %13, float* %28, align 4, !tbaa !7
  %33 = alloca [14 x i8*], align 16
  %34 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 0
  %35 = bitcast [14 x i8*]* %33 to i32**
  store i32* %15, i32** %35, align 16
  %36 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 1
  %37 = bitcast i8** %36 to float***
  store float** %16, float*** %37, align 8
  %38 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 2
  %39 = bitcast i8** %38 to float***
  store float** %17, float*** %39, align 16
  %40 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 3
  %41 = bitcast i8** %40 to float***
  store float** %18, float*** %41, align 8
  %42 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 4
  %43 = bitcast i8** %42 to i32**
  store i32* %19, i32** %43, align 16
  %44 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 5
  %45 = bitcast i8** %44 to i32**
  store i32* %20, i32** %45, align 8
  %46 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 6
  %47 = bitcast i8** %46 to i32**
  store i32* %21, i32** %47, align 16
  %48 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 7
  %49 = bitcast i8** %48 to i32**
  store i32* %22, i32** %49, align 8
  %50 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 8
  %51 = bitcast i8** %50 to float**
  store float* %23, float** %51, align 16
  %52 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 9
  %53 = bitcast i8** %52 to float**
  store float* %24, float** %53, align 8
  %54 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 10
  %55 = bitcast i8** %54 to float**
  store float* %25, float** %55, align 16
  %56 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 11
  %57 = bitcast i8** %56 to float**
  store float* %26, float** %57, align 8
  %58 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 12
  %59 = bitcast i8** %58 to float**
  store float* %27, float** %59, align 16
  %60 = getelementptr inbounds [14 x i8*], [14 x i8*]* %33, i64 0, i64 13
  %61 = bitcast i8** %60 to float**
  store float* %28, float** %61, align 8
  %62 = call i32 @__cudaPopCallConfiguration(%struct.dim3* nonnull %29, %struct.dim3* nonnull %30, i64* nonnull %31, i8** nonnull %32)
  %63 = load i64, i64* %31, align 8
  %64 = bitcast i8** %32 to %struct.CUstream_st**
  %65 = load %struct.CUstream_st*, %struct.CUstream_st** %64, align 8
  %66 = bitcast %struct.dim3* %29 to i64*
  %67 = load i64, i64* %66, align 8
  %68 = getelementptr inbounds %struct.dim3, %struct.dim3* %29, i64 0, i32 2
  %69 = load i32, i32* %68, align 8
  %70 = bitcast %struct.dim3* %30 to i64*
  %71 = load i64, i64* %70, align 8
  %72 = getelementptr inbounds %struct.dim3, %struct.dim3* %30, i64 0, i32 2
  %73 = load i32, i32* %72, align 8
  %74 = call i32 @cudaLaunchKernel(i8* bitcast (void (i32, float*, float*, float*, i32, i32, i32, i32, float, float, float, float, float, float)* @_Z24calculate_temp_uncheckediPfS_S_iiiiffffff to i8*), i64 %67, i32 %69, i64 %71, i32 %73, i8** nonnull %34, i64 %63, %struct.CUstream_st* %65)
  ret void
}

; Function Attrs: uwtable
define dso_local i32 @_Z25compute_tran_temp_checkedPfPS_iiiiiiii(float*, float** nocapture readonly, i32, i32, i32, i32, i32, i32, i32, i32) local_unnamed_addr #0 {
  %11 = alloca %struct.dim3, align 8
  %12 = alloca %struct.dim3, align 8
  %13 = bitcast %struct.dim3* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %13) #17
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %11, i32 16, i32 16, i32 1)
  %14 = bitcast %struct.dim3* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %14) #17
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %12, i32 %6, i32 %7, i32 1)
  %15 = load float, float* @chip_height, align 4, !tbaa !7
  %16 = sitofp i32 %3 to float
  %17 = fdiv float %15, %16
  %18 = load float, float* @chip_width, align 4, !tbaa !7
  %19 = sitofp i32 %2 to float
  %20 = fdiv float %18, %19
  %21 = load float, float* @t_chip, align 4, !tbaa !7
  %22 = fpext float %21 to double
  %23 = fmul contract double %22, 8.750000e+05
  %24 = fpext float %20 to double
  %25 = fmul contract double %23, %24
  %26 = fpext float %17 to double
  %27 = fmul contract double %25, %26
  %28 = fptrunc double %27 to float
  %29 = fmul contract double %22, 2.000000e+02
  %30 = fmul contract double %29, %26
  %31 = fdiv double %24, %30
  %32 = fptrunc double %31 to float
  %33 = fmul contract double %29, %24
  %34 = fdiv double %26, %33
  %35 = fptrunc double %34 to float
  %36 = fmul contract float %17, 1.000000e+02
  %37 = fmul contract float %36, %20
  %38 = fdiv float %21, %37
  %39 = fmul contract double %22, 5.000000e-01
  %40 = fmul contract double %39, 1.750000e+06
  %41 = fdiv double 3.000000e+06, %40
  %42 = fptrunc double %41 to float
  %43 = fpext float %42 to double
  %44 = fdiv double 1.000000e-03, %43
  %45 = fptrunc double %44 to float
  %46 = sitofp i32 %4 to float
  %47 = icmp sgt i32 %4, 0
  br i1 %47, label %48, label %79

48:                                               ; preds = %10
  %49 = bitcast %struct.dim3* %12 to i64*
  %50 = getelementptr inbounds %struct.dim3, %struct.dim3* %12, i64 0, i32 2
  %51 = bitcast %struct.dim3* %11 to i64*
  %52 = getelementptr inbounds %struct.dim3, %struct.dim3* %11, i64 0, i32 2
  %53 = sitofp i32 %5 to float
  %54 = sitofp i32 %5 to float
  br label %55

55:                                               ; preds = %48, %76
  %56 = phi float [ 0.000000e+00, %48 ], [ %77, %76 ]
  %57 = phi i32 [ 1, %48 ], [ %58, %76 ]
  %58 = phi i32 [ 0, %48 ], [ %57, %76 ]
  %59 = load i64, i64* %49, align 8
  %60 = load i32, i32* %50, align 8
  %61 = load i64, i64* %51, align 8
  %62 = load i32, i32* %52, align 8
  %63 = call i32 @__cudaPushCallConfiguration(i64 %59, i32 %60, i64 %61, i32 %62, i64 0, i8* null)
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %76

65:                                               ; preds = %55
  %66 = fsub contract float %46, %56
  %67 = fcmp oge float %66, %53
  %68 = select i1 %67, float %53, float %66
  %69 = fptosi float %68 to i32
  %70 = sext i32 %58 to i64
  %71 = getelementptr inbounds float*, float** %1, i64 %70
  %72 = load float*, float** %71, align 8, !tbaa !3
  %73 = sext i32 %57 to i64
  %74 = getelementptr inbounds float*, float** %1, i64 %73
  %75 = load float*, float** %74, align 8, !tbaa !3
  call void @_Z22calculate_temp_checkediPfS_S_iiiiffffff(i32 %69, float* %0, float* %72, float* %75, i32 %2, i32 %3, i32 %8, i32 %9, float %28, float %32, float %35, float %38, float %45, float 0x3F50624DE0000000)
  br label %76

76:                                               ; preds = %55, %65
  %77 = fadd contract float %56, %54
  %78 = fcmp olt float %77, %46
  br i1 %78, label %55, label %79

79:                                               ; preds = %76, %10
  %80 = phi i32 [ 0, %10 ], [ %57, %76 ]
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %14) #17
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %13) #17
  ret i32 %80
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3*, i32, i32, i32) unnamed_addr #6 comdat align 2 {
  %5 = getelementptr inbounds %struct.dim3, %struct.dim3* %0, i64 0, i32 0
  store i32 %1, i32* %5, align 4, !tbaa !11
  %6 = getelementptr inbounds %struct.dim3, %struct.dim3* %0, i64 0, i32 1
  store i32 %2, i32* %6, align 4, !tbaa !13
  %7 = getelementptr inbounds %struct.dim3, %struct.dim3* %0, i64 0, i32 2
  store i32 %3, i32* %7, align 4, !tbaa !14
  ret void
}

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) local_unnamed_addr #1

; Function Attrs: uwtable
define dso_local i32 @_Z27compute_tran_temp_uncheckedPfPS_iiiiiiii(float*, float** nocapture readonly, i32, i32, i32, i32, i32, i32, i32, i32) local_unnamed_addr #0 {
  %11 = alloca %struct.dim3, align 8
  %12 = alloca %struct.dim3, align 8
  %13 = bitcast %struct.dim3* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %13) #17
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %11, i32 16, i32 16, i32 1)
  %14 = bitcast %struct.dim3* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %14) #17
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %12, i32 %6, i32 %7, i32 1)
  %15 = load float, float* @chip_height, align 4, !tbaa !7
  %16 = sitofp i32 %3 to float
  %17 = fdiv float %15, %16
  %18 = load float, float* @chip_width, align 4, !tbaa !7
  %19 = sitofp i32 %2 to float
  %20 = fdiv float %18, %19
  %21 = load float, float* @t_chip, align 4, !tbaa !7
  %22 = fpext float %21 to double
  %23 = fmul contract double %22, 8.750000e+05
  %24 = fpext float %20 to double
  %25 = fmul contract double %23, %24
  %26 = fpext float %17 to double
  %27 = fmul contract double %25, %26
  %28 = fptrunc double %27 to float
  %29 = fmul contract double %22, 2.000000e+02
  %30 = fmul contract double %29, %26
  %31 = fdiv double %24, %30
  %32 = fptrunc double %31 to float
  %33 = fmul contract double %29, %24
  %34 = fdiv double %26, %33
  %35 = fptrunc double %34 to float
  %36 = fmul contract float %17, 1.000000e+02
  %37 = fmul contract float %36, %20
  %38 = fdiv float %21, %37
  %39 = fmul contract double %22, 5.000000e-01
  %40 = fmul contract double %39, 1.750000e+06
  %41 = fdiv double 3.000000e+06, %40
  %42 = fptrunc double %41 to float
  %43 = fpext float %42 to double
  %44 = fdiv double 1.000000e-03, %43
  %45 = fptrunc double %44 to float
  %46 = sitofp i32 %4 to float
  %47 = icmp sgt i32 %4, 0
  br i1 %47, label %48, label %79

48:                                               ; preds = %10
  %49 = bitcast %struct.dim3* %12 to i64*
  %50 = getelementptr inbounds %struct.dim3, %struct.dim3* %12, i64 0, i32 2
  %51 = bitcast %struct.dim3* %11 to i64*
  %52 = getelementptr inbounds %struct.dim3, %struct.dim3* %11, i64 0, i32 2
  %53 = sitofp i32 %5 to float
  %54 = sitofp i32 %5 to float
  br label %55

55:                                               ; preds = %48, %76
  %56 = phi float [ 0.000000e+00, %48 ], [ %77, %76 ]
  %57 = phi i32 [ 1, %48 ], [ %58, %76 ]
  %58 = phi i32 [ 0, %48 ], [ %57, %76 ]
  %59 = load i64, i64* %49, align 8
  %60 = load i32, i32* %50, align 8
  %61 = load i64, i64* %51, align 8
  %62 = load i32, i32* %52, align 8
  %63 = call i32 @__cudaPushCallConfiguration(i64 %59, i32 %60, i64 %61, i32 %62, i64 0, i8* null)
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %76

65:                                               ; preds = %55
  %66 = fsub contract float %46, %56
  %67 = fcmp oge float %66, %53
  %68 = select i1 %67, float %53, float %66
  %69 = fptosi float %68 to i32
  %70 = sext i32 %58 to i64
  %71 = getelementptr inbounds float*, float** %1, i64 %70
  %72 = load float*, float** %71, align 8, !tbaa !3
  %73 = sext i32 %57 to i64
  %74 = getelementptr inbounds float*, float** %1, i64 %73
  %75 = load float*, float** %74, align 8, !tbaa !3
  call void @_Z24calculate_temp_uncheckediPfS_S_iiiiffffff(i32 %69, float* %0, float* %72, float* %75, i32 %2, i32 %3, i32 %8, i32 %9, float %28, float %32, float %35, float %38, float %45, float 0x3F50624DE0000000)
  br label %76

76:                                               ; preds = %55, %65
  %77 = fadd contract float %56, %54
  %78 = fcmp olt float %77, %46
  br i1 %78, label %55, label %79

79:                                               ; preds = %76, %10
  %80 = phi i32 [ 0, %10 ], [ %57, %76 ]
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %14) #17
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %13) #17
  ret i32 %80
}

; Function Attrs: noreturn nounwind uwtable
define dso_local void @_Z5usageiPPc(i32, i8** nocapture readonly) local_unnamed_addr #8 {
  %3 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %4 = load i8*, i8** %1, align 8, !tbaa !3
  %5 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %3, i8* getelementptr inbounds ([100 x i8], [100 x i8]* @.str.8, i64 0, i64 0), i8* %4) #18
  %6 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %7 = tail call i64 @fwrite(i8* getelementptr inbounds ([78 x i8], [78 x i8]* @.str.9, i64 0, i64 0), i64 77, i64 1, %struct._IO_FILE* %6) #18
  %8 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %9 = tail call i64 @fwrite(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.10, i64 0, i64 0), i64 52, i64 1, %struct._IO_FILE* %8) #18
  %10 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %11 = tail call i64 @fwrite(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.11, i64 0, i64 0), i64 37, i64 1, %struct._IO_FILE* %10) #18
  %12 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %13 = tail call i64 @fwrite(i8* getelementptr inbounds ([89 x i8], [89 x i8]* @.str.12, i64 0, i64 0), i64 88, i64 1, %struct._IO_FILE* %12) #18
  %14 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %15 = tail call i64 @fwrite(i8* getelementptr inbounds ([86 x i8], [86 x i8]* @.str.13, i64 0, i64 0), i64 85, i64 1, %struct._IO_FILE* %14) #18
  %16 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !3
  %17 = tail call i64 @fwrite(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.14, i64 0, i64 0), i64 41, i64 1, %struct._IO_FILE* %16) #18
  tail call void @exit(i32 1) #19
  unreachable
}

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #9

; Function Attrs: norecurse uwtable
define dso_local i32 @main(i32, i8** nocapture readonly) local_unnamed_addr #10 {
  %3 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.15, i64 0, i64 0), i32 16, i32 16)
  tail call void @_Z3runiPPc(i32 %0, i8** %1)
  ret i32 0
}

; Function Attrs: uwtable
define dso_local void @_Z3runiPPc(i32, i8** nocapture readonly) local_unnamed_addr #0 {
  %3 = alloca [2 x float*], align 16
  %4 = alloca float*, align 8
  %5 = alloca [2 x float*], align 16
  %6 = icmp eq i32 %0, 7
  br i1 %6, label %8, label %7

7:                                                ; preds = %2
  tail call void @_Z5usageiPPc(i32 undef, i8** %1)
  unreachable

8:                                                ; preds = %2
  %9 = getelementptr inbounds i8*, i8** %1, i64 1
  %10 = load i8*, i8** %9, align 8, !tbaa !3
  %11 = tail call i32 @atoi(i8* %10) #20
  %12 = icmp slt i32 %11, 1
  br i1 %12, label %23, label %13

13:                                               ; preds = %8
  %14 = getelementptr inbounds i8*, i8** %1, i64 2
  %15 = load i8*, i8** %14, align 8, !tbaa !3
  %16 = tail call i32 @atoi(i8* %15) #20
  %17 = icmp slt i32 %16, 1
  br i1 %17, label %23, label %18

18:                                               ; preds = %13
  %19 = getelementptr inbounds i8*, i8** %1, i64 3
  %20 = load i8*, i8** %19, align 8, !tbaa !3
  %21 = tail call i32 @atoi(i8* %20) #20
  %22 = icmp slt i32 %21, 1
  br i1 %22, label %23, label %24

23:                                               ; preds = %18, %13, %8
  tail call void @_Z5usageiPPc(i32 undef, i8** nonnull %1)
  unreachable

24:                                               ; preds = %18
  %25 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i32 %21)
  %26 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %25, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.16, i64 0, i64 0))
  %27 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull %26, i32 %16)
  %28 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %27, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %29 = getelementptr inbounds i8*, i8** %1, i64 4
  %30 = load i8*, i8** %29, align 8, !tbaa !3
  %31 = getelementptr inbounds i8*, i8** %1, i64 5
  %32 = load i8*, i8** %31, align 8, !tbaa !3
  %33 = getelementptr inbounds i8*, i8** %1, i64 6
  %34 = load i8*, i8** %33, align 8, !tbaa !3
  %35 = mul nsw i32 %11, %11
  %36 = shl nsw i32 %16, 1
  %37 = sub nsw i32 16, %36
  %38 = sdiv i32 %11, %37
  %39 = srem i32 %11, %37
  %40 = icmp ne i32 %39, 0
  %41 = zext i1 %40 to i32
  %42 = add nsw i32 %38, %41
  %43 = zext i32 %35 to i64
  %44 = shl nuw nsw i64 %43, 2
  %45 = tail call noalias i8* @malloc(i64 %44) #17
  %46 = bitcast i8* %45 to float*
  %47 = tail call noalias i8* @malloc(i64 %44) #17
  %48 = bitcast i8* %47 to float*
  %49 = tail call noalias i8* @calloc(i64 %43, i64 4) #17
  %50 = bitcast i8* %49 to float*
  %51 = icmp ne i8* %47, null
  %52 = icmp ne i8* %45, null
  %53 = and i1 %52, %51
  %54 = icmp ne i8* %49, null
  %55 = and i1 %53, %54
  br i1 %55, label %57, label %56

56:                                               ; preds = %24
  tail call void @_Z5fatalPKc(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.17, i64 0, i64 0))
  br label %57

57:                                               ; preds = %24, %56
  %58 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([94 x i8], [94 x i8]* @.str.18, i64 0, i64 0), i32 %16, i32 %11, i32 %11, i32 %16, i32 %16, i32 %42, i32 %42, i32 %37, i32 %37)
  tail call void @_Z9readinputPfiiPc(float* %46, i32 %11, i32 %11, i8* %30)
  tail call void @_Z9readinputPfiiPc(float* %48, i32 %11, i32 %11, i8* %32)
  %59 = bitcast [2 x float*]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %59) #17
  %60 = bitcast float** %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %60) #17
  %61 = getelementptr inbounds [2 x float*], [2 x float*]* %3, i64 0, i64 0
  %62 = bitcast [2 x float*]* %3 to i8**
  %63 = call i32 @cudaMalloc(i8** nonnull %62, i64 %44)
  %64 = getelementptr inbounds [2 x float*], [2 x float*]* %3, i64 0, i64 1
  %65 = bitcast float** %64 to i8**
  %66 = call i32 @cudaMalloc(i8** nonnull %65, i64 %44)
  %67 = load i8*, i8** %62, align 16, !tbaa !3
  %68 = call i32 @cudaMemcpy(i8* %67, i8* %45, i64 %44, i32 1)
  %69 = bitcast float** %4 to i8**
  %70 = call i32 @cudaMalloc(i8** nonnull %69, i64 %44)
  %71 = load i8*, i8** %69, align 8, !tbaa !3
  %72 = call i32 @cudaMemcpy(i8* %71, i8* %47, i64 %44, i32 1)
  %73 = call i32 @puts(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @str.30, i64 0, i64 0))
  %74 = load float*, float** %4, align 8, !tbaa !3
  %75 = call i32 @_Z25compute_tran_temp_checkedPfPS_iiiiiiii(float* %74, float** nonnull %61, i32 %11, i32 %11, i32 %21, i32 %16, i32 %42, i32 %42, i32 %16, i32 %16)
  %76 = call i32 @puts(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @str.31, i64 0, i64 0))
  %77 = sext i32 %75 to i64
  %78 = getelementptr inbounds [2 x float*], [2 x float*]* %3, i64 0, i64 %77
  %79 = bitcast float** %78 to i8**
  %80 = load i8*, i8** %79, align 8, !tbaa !3
  %81 = call i32 @cudaMemcpy(i8* %49, i8* %80, i64 %44, i32 2)
  %82 = call i32 @cudaGetLastError()
  %83 = icmp eq i32 %82, 0
  br i1 %83, label %93, label %84

84:                                               ; preds = %57
  %85 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cerr, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.21, i64 0, i64 0))
  %86 = call i8* @cudaGetErrorString(i32 %82)
  %87 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %85, i8* %86)
  %88 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %87, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.22, i64 0, i64 0))
  %89 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %88, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.23, i64 0, i64 0))
  %90 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* nonnull dereferenceable(272) %89, i8 signext 58)
  %91 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull %90, i32 489)
  %92 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %91, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %93

93:                                               ; preds = %57, %84
  %94 = bitcast [2 x float*]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %94) #17
  %95 = call noalias i8* @calloc(i64 %43, i64 4) #17
  %96 = bitcast i8* %95 to float*
  %97 = getelementptr inbounds [2 x float*], [2 x float*]* %5, i64 0, i64 0
  %98 = bitcast [2 x float*]* %5 to i8**
  %99 = call i32 @cudaMalloc(i8** nonnull %98, i64 %44)
  %100 = getelementptr inbounds [2 x float*], [2 x float*]* %5, i64 0, i64 1
  %101 = bitcast float** %100 to i8**
  %102 = call i32 @cudaMalloc(i8** nonnull %101, i64 %44)
  %103 = load i8*, i8** %98, align 16, !tbaa !3
  %104 = call i32 @cudaMemcpy(i8* %103, i8* %45, i64 %44, i32 1)
  %105 = load float*, float** %4, align 8, !tbaa !3
  %106 = call i32 @_Z27compute_tran_temp_uncheckedPfPS_iiiiiiii(float* %105, float** nonnull %97, i32 %11, i32 %11, i32 %21, i32 %16, i32 %42, i32 %42, i32 %16, i32 %16)
  %107 = call i32 @puts(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @str.32, i64 0, i64 0))
  %108 = sext i32 %106 to i64
  %109 = getelementptr inbounds [2 x float*], [2 x float*]* %5, i64 0, i64 %108
  %110 = bitcast float** %109 to i8**
  %111 = load i8*, i8** %110, align 8, !tbaa !3
  %112 = call i32 @cudaMemcpy(i8* %95, i8* %111, i64 %44, i32 2)
  %113 = call i32 @_Z20check_array_equalityIfEiPT_S1_ifb(float* %50, float* %96, i32 %35, float 0x3E7AD7F2A0000000, i1 zeroext true)
  %114 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.25, i64 0, i64 0))
  %115 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull %114, i32 %113)
  %116 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %115, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  call void @_Z11writeoutputPfiiPc(float* %50, i32 %11, i32 %11, i8* %34)
  %117 = load i8*, i8** %69, align 8, !tbaa !3
  %118 = call i32 @cudaFree(i8* %117)
  %119 = load i8*, i8** %62, align 16, !tbaa !3
  %120 = call i32 @cudaFree(i8* %119)
  %121 = load i8*, i8** %65, align 8, !tbaa !3
  %122 = call i32 @cudaFree(i8* %121)
  call void @free(i8* %49) #17
  %123 = load i8*, i8** %98, align 16, !tbaa !3
  %124 = call i32 @cudaFree(i8* %123)
  %125 = load i8*, i8** %101, align 8, !tbaa !3
  %126 = call i32 @cudaFree(i8* %125)
  call void @free(i8* %95) #17
  %127 = call i32 @cudaGetLastError()
  %128 = icmp eq i32 %127, 0
  br i1 %128, label %138, label %129

129:                                              ; preds = %93
  %130 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cerr, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.21, i64 0, i64 0))
  %131 = call i8* @cudaGetErrorString(i32 %127)
  %132 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %130, i8* %131)
  %133 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %132, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.22, i64 0, i64 0))
  %134 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %133, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.23, i64 0, i64 0))
  %135 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* nonnull dereferenceable(272) %134, i8 signext 58)
  %136 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull %135, i32 514)
  %137 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %136, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %138

138:                                              ; preds = %93, %129
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %94) #17
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %60) #17
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %59) #17
  ret void
}

; Function Attrs: inlinehint nounwind readonly uwtable
define available_externally dso_local i32 @atoi(i8* nonnull) local_unnamed_addr #11 {
  %2 = tail call i64 @strtol(i8* nocapture nonnull %0, i8** null, i32 10) #17
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272), i8*) local_unnamed_addr #12 {
  %3 = icmp eq i8* %1, null
  br i1 %3, label %4, label %13

4:                                                ; preds = %2
  %5 = bitcast %"class.std::basic_ostream"* %0 to i8**
  %6 = load i8*, i8** %5, align 8, !tbaa !15
  %7 = getelementptr i8, i8* %6, i64 -24
  %8 = bitcast i8* %7 to i64*
  %9 = load i64, i64* %8, align 8
  %10 = bitcast %"class.std::basic_ostream"* %0 to i8*
  %11 = getelementptr inbounds i8, i8* %10, i64 %9
  %12 = bitcast i8* %11 to %"class.std::basic_ios"*
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate(%"class.std::basic_ios"* nonnull %12, i32 1)
  br label %16

13:                                               ; preds = %2
  %14 = tail call i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8* nonnull %1)
  %15 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %0, i8* nonnull %1, i64 %14)
  br label %16

16:                                               ; preds = %13, %4
  ret %"class.std::basic_ostream"* %0
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"*, i32) local_unnamed_addr #1

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"*, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)*) local_unnamed_addr #0 align 2 {
  %3 = tail call dereferenceable(272) %"class.std::basic_ostream"* %1(%"class.std::basic_ostream"* dereferenceable(272) %0)
  ret %"class.std::basic_ostream"* %3
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) #12 {
  %2 = bitcast %"class.std::basic_ostream"* %0 to i8**
  %3 = load i8*, i8** %2, align 8, !tbaa !15
  %4 = getelementptr i8, i8* %3, i64 -24
  %5 = bitcast i8* %4 to i64*
  %6 = load i64, i64* %5, align 8
  %7 = bitcast %"class.std::basic_ostream"* %0 to i8*
  %8 = getelementptr inbounds i8, i8* %7, i64 %6
  %9 = bitcast i8* %8 to %"class.std::basic_ios"*
  %10 = tail call signext i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc(%"class.std::basic_ios"* nonnull %9, i8 signext 10)
  %11 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %0, i8 signext %10)
  %12 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt5flushIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* nonnull dereferenceable(272) %11)
  ret %"class.std::basic_ostream"* %12
}

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @calloc(i64, i64) local_unnamed_addr #5

declare dso_local i32 @cudaMalloc(i8**, i64) local_unnamed_addr #1

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) local_unnamed_addr #1

declare dso_local i32 @cudaGetLastError() local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* dereferenceable(272), i8 signext) local_unnamed_addr #12 {
  %3 = alloca i8, align 1
  store i8 %1, i8* %3, align 1, !tbaa !17
  %4 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %0, i8* nonnull %3, i64 1)
  ret %"class.std::basic_ostream"* %4
}

declare dso_local i8* @cudaGetErrorString(i32) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i32 @_Z20check_array_equalityIfEiPT_S1_ifb(float*, float*, i32, float, i1 zeroext) local_unnamed_addr #12 comdat {
  %6 = icmp sgt i32 %2, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %5
  %8 = zext i32 %2 to i64
  br label %11

9:                                                ; preds = %37, %5
  %10 = phi i32 [ 0, %5 ], [ %38, %37 ]
  ret i32 %10

11:                                               ; preds = %37, %7
  %12 = phi i64 [ 0, %7 ], [ %39, %37 ]
  %13 = phi i32 [ 0, %7 ], [ %38, %37 ]
  %14 = getelementptr inbounds float, float* %0, i64 %12
  %15 = load float, float* %14, align 4, !tbaa !7
  %16 = getelementptr inbounds float, float* %1, i64 %12
  %17 = load float, float* %16, align 4, !tbaa !7
  %18 = fsub contract float %15, %17
  %19 = tail call float @_ZSt3absf(float %18)
  %20 = fcmp ogt float %19, %3
  br i1 %20, label %21, label %37

21:                                               ; preds = %11
  %22 = add nsw i32 %13, 1
  %23 = icmp slt i32 %22, 20
  %24 = and i1 %23, %4
  br i1 %24, label %25, label %37

25:                                               ; preds = %21
  %26 = trunc i64 %12 to i32
  %27 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i32 %26)
  %28 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %27, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.26, i64 0, i64 0))
  %29 = load float, float* %14, align 4, !tbaa !7
  %30 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %28, float %29)
  %31 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %30, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.27, i64 0, i64 0))
  %32 = load float, float* %16, align 4, !tbaa !7
  %33 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %31, float %32)
  %34 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %33, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.28, i64 0, i64 0))
  %35 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %34, float %19)
  %36 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %35, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %37

37:                                               ; preds = %21, %25, %11
  %38 = phi i32 [ %22, %25 ], [ %22, %21 ], [ %13, %11 ]
  %39 = add nuw nsw i64 %12, 1
  %40 = icmp eq i64 %39, %8
  br i1 %40, label %9, label %11
}

declare dso_local i32 @cudaFree(i8*) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #5

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate(%"class.std::basic_ios"*, i32) local_unnamed_addr #0 align 2 {
  %3 = tail call i32 @_ZNKSt9basic_iosIcSt11char_traitsIcEE7rdstateEv(%"class.std::basic_ios"* %0)
  %4 = tail call i32 @_ZStorSt12_Ios_IostateS_(i32 %3, i32 %1)
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"class.std::basic_ios"* %0, i32 %4)
  ret void
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8*) local_unnamed_addr #6 comdat align 2 {
  %2 = tail call i64 @strlen(i8* %0) #17
  ret i64 %2
}

declare dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"class.std::basic_ios"*, i32) local_unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i32 @_ZStorSt12_Ios_IostateS_(i32, i32) local_unnamed_addr #13 comdat {
  %3 = or i32 %1, %0
  ret i32 %3
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i32 @_ZNKSt9basic_iosIcSt11char_traitsIcEE7rdstateEv(%"class.std::basic_ios"*) local_unnamed_addr #6 align 2 {
  %2 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %0, i64 0, i32 0, i32 5
  %3 = load i32, i32* %2, align 8, !tbaa !18
  ret i32 %3
}

; Function Attrs: argmemonly nofree nounwind readonly
declare dso_local i64 @strlen(i8* nocapture) local_unnamed_addr #14

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt5flushIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) local_unnamed_addr #12 {
  %2 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %0)
  ret %"class.std::basic_ostream"* %2
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #1

; Function Attrs: uwtable
define available_externally dso_local signext i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc(%"class.std::basic_ios"*, i8 signext) local_unnamed_addr #0 align 2 {
  %3 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %0, i64 0, i32 5
  %4 = load %"class.std::ctype"*, %"class.std::ctype"** %3, align 8, !tbaa !25
  %5 = tail call dereferenceable(576) %"class.std::ctype"* @_ZSt13__check_facetISt5ctypeIcEERKT_PS3_(%"class.std::ctype"* %4)
  %6 = tail call signext i8 @_ZNKSt5ctypeIcE5widenEc(%"class.std::ctype"* nonnull %5, i8 signext %1)
  ret i8 %6
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local dereferenceable(576) %"class.std::ctype"* @_ZSt13__check_facetISt5ctypeIcEERKT_PS3_(%"class.std::ctype"*) local_unnamed_addr #12 comdat {
  %2 = icmp eq %"class.std::ctype"* %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void @_ZSt16__throw_bad_castv() #21
  unreachable

4:                                                ; preds = %1
  ret %"class.std::ctype"* %0
}

; Function Attrs: uwtable
define linkonce_odr dso_local signext i8 @_ZNKSt5ctypeIcE5widenEc(%"class.std::ctype"*, i8 signext) local_unnamed_addr #0 comdat align 2 {
  %3 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %0, i64 0, i32 8
  %4 = load i8, i8* %3, align 8, !tbaa !28
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %10, label %6

6:                                                ; preds = %2
  %7 = zext i8 %1 to i64
  %8 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %0, i64 0, i32 9, i64 %7
  %9 = load i8, i8* %8, align 1, !tbaa !17
  br label %16

10:                                               ; preds = %2
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %0)
  %11 = bitcast %"class.std::ctype"* %0 to i8 (%"class.std::ctype"*, i8)***
  %12 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %11, align 8, !tbaa !15
  %13 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %12, i64 6
  %14 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %13, align 8
  %15 = tail call signext i8 %14(%"class.std::ctype"* nonnull %0, i8 signext %1)
  br label %16

16:                                               ; preds = %10, %6
  %17 = phi i8 [ %9, %6 ], [ %15, %10 ]
  ret i8 %17
}

; Function Attrs: noreturn
declare dso_local void @_ZSt16__throw_bad_castv() local_unnamed_addr #15

declare dso_local void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZSt3absf(float) local_unnamed_addr #13 comdat {
  %2 = tail call float @llvm.fabs.f32(float %0)
  ret float %2
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"*, float) local_unnamed_addr #0 align 2 {
  %3 = fpext float %1 to double
  %4 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* %0, double %3)
  ret %"class.std::basic_ostream"* %4
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #16

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #1

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_hotspot.cu() #0 section ".text.startup" {
  tail call fastcc void @__cxx_global_var_init()
  ret void
}

; Function Attrs: nofree nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare i8* @fgets_unlocked(i8*, i32, %struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #3

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind }
attributes #4 = { nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { argmemonly nounwind }
attributes #8 = { noreturn nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #14 = { argmemonly nofree nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #15 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #16 = { nounwind readnone speculatable }
attributes #17 = { nounwind }
attributes #18 = { cold }
attributes #19 = { noreturn nounwind }
attributes #20 = { nounwind readonly }
attributes #21 = { noreturn }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !5, i64 0}
!11 = !{!12, !10, i64 0}
!12 = !{!"_ZTS4dim3", !10, i64 0, !10, i64 4, !10, i64 8}
!13 = !{!12, !10, i64 4}
!14 = !{!12, !10, i64 8}
!15 = !{!16, !16, i64 0}
!16 = !{!"vtable pointer", !6, i64 0}
!17 = !{!5, !5, i64 0}
!18 = !{!19, !22, i64 32}
!19 = !{!"_ZTSSt8ios_base", !20, i64 8, !20, i64 16, !21, i64 24, !22, i64 28, !22, i64 32, !4, i64 40, !23, i64 48, !5, i64 64, !10, i64 192, !4, i64 200, !24, i64 208}
!20 = !{!"long", !5, i64 0}
!21 = !{!"_ZTSSt13_Ios_Fmtflags", !5, i64 0}
!22 = !{!"_ZTSSt12_Ios_Iostate", !5, i64 0}
!23 = !{!"_ZTSNSt8ios_base6_WordsE", !4, i64 0, !20, i64 8}
!24 = !{!"_ZTSSt6locale", !4, i64 0}
!25 = !{!26, !4, i64 240}
!26 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !4, i64 216, !5, i64 224, !27, i64 225, !4, i64 232, !4, i64 240, !4, i64 248, !4, i64 256}
!27 = !{!"bool", !5, i64 0}
!28 = !{!29, !5, i64 56}
!29 = !{!"_ZTSSt5ctypeIcE", !4, i64 16, !27, i64 24, !4, i64 32, !4, i64 40, !4, i64 48, !5, i64 56, !5, i64 57, !5, i64 313, !5, i64 569}
