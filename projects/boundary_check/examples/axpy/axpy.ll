; ModuleID = 'axpy.cu'
source_filename = "axpy.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%struct.option = type { i8*, i32, i32*, i32 }
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
%"struct.std::chrono::time_point" = type { %"struct.std::chrono::duration" }
%"struct.std::chrono::duration" = type { i64 }
%"struct.std::chrono::duration.0" = type { i64 }
%"class.std::random_device" = type { %union.anon }
%union.anon = type { %"class.std::mersenne_twister_engine" }
%"class.std::mersenne_twister_engine" = type { [624 x i64], i64 }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon.1 }
%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%union.anon.1 = type { i64, [8 x i8] }
%"class.std::allocator" = type { i8 }
%"class.std::uniform_real_distribution" = type { %"struct.std::uniform_real_distribution<float>::param_type" }
%"struct.std::uniform_real_distribution<float>::param_type" = type { float, float }
%"struct.std::__detail::_Adaptor" = type { %"class.std::mersenne_twister_engine"* }

$_Z20create_sample_vectorPfibb = comdat any

$_Z19print_array_indexedIfEvPT_i = comdat any

$_ZSt3minIiERKT_S2_S2_ = comdat any

$_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE = comdat any

$_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE = comdat any

$_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv = comdat any

$_ZN4dim3C2Ejjj = comdat any

$_Z20check_array_equalityIfEiPT_S1_ifb = comdat any

$_ZNSt13random_deviceC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZNSt13random_deviceclEv = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em = comdat any

$_ZNSt25uniform_real_distributionIfEC2Eff = comdat any

$_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_ = comdat any

$_ZNSt13random_deviceD2Ev = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm = comdat any

$_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_ = comdat any

$_ZNSt8__detail5__modImLm624ELm1ELm0EEET_S1_ = comdat any

$_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm = comdat any

$_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm = comdat any

$_ZNSt25uniform_real_distributionIfE10param_typeC2Eff = comdat any

$_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_RKNS0_10param_typeE = comdat any

$_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEC2ERS2_ = comdat any

$_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEclEv = comdat any

$_ZNKSt25uniform_real_distributionIfE10param_type1bEv = comdat any

$_ZNKSt25uniform_real_distributionIfE10param_type1aEv = comdat any

$_ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_ = comdat any

$_ZSt3minImERKT_S2_S2_ = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv = comdat any

$_ZSt3loge = comdat any

$_ZSt3maxImERKT_S2_S2_ = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

$_ZSt9nextafterff = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv = comdat any

$__clang_call_terminate = comdat any

$_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000EEEES2_ILl1ELl1000000EElLb1ELb0EE6__castIlS2_ILl1ELl1000000000EEEES4_RKNS1_IT_T0_EE = comdat any

$_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv = comdat any

$_ZNSt6chrono8durationIlSt5ratioILl1ELl1000EEEC2IlvEERKT_ = comdat any

$_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_ = comdat any

$_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv = comdat any

$_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC2IlvEERKT_ = comdat any

$_ZSt3absf = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZZ4mainE12long_options = internal global [3 x %struct.option] [%struct.option { i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i32 0, i32 0), i32 0, i32* null, i32 104 }, %struct.option { i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.1, i32 0, i32 0), i32 1, i32* null, i32 115 }, %struct.option zeroinitializer], align 16
@.str = private unnamed_addr constant [15 x i8] c"human_readable\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"size\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"hs:c\00", align 1
@optarg = external dso_local global i8*, align 8
@_ZSt4cout = external dso_local global %"class.std::basic_ostream", align 8
@.str.3 = private unnamed_addr constant [4 x i8] c"a: \00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"\0AX: \00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"Y: \00", align 1
@.str.6 = private unnamed_addr constant [48 x i8] c"----------------------------------------------\0A\00", align 1
@.str.7 = private unnamed_addr constant [12 x i8] c"CPU Result:\00", align 1
@.str.8 = private unnamed_addr constant [11 x i8] c"Duration: \00", align 1
@.str.9 = private unnamed_addr constant [4 x i8] c" ms\00", align 1
@_ZSt4cerr = external dso_local global %"class.std::basic_ostream", align 8
@.str.10 = private unnamed_addr constant [15 x i8] c"Cuda failure: \00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c" at: \00", align 1
@.str.12 = private unnamed_addr constant [8 x i8] c"axpy.cu\00", align 1
@.str.13 = private unnamed_addr constant [24 x i8] c"GPU Result - Unchecked:\00", align 1
@.str.14 = private unnamed_addr constant [19 x i8] c"Number of errors: \00", align 1
@.str.15 = private unnamed_addr constant [22 x i8] c"GPU Result - Checked:\00", align 1
@.str.16 = private unnamed_addr constant [8 x i8] c"default\00", align 1
@.str.17 = private unnamed_addr constant [3 x i8] c") \00", align 1
@.str.18 = private unnamed_addr constant [6 x i8] c") X: \00", align 1
@.str.19 = private unnamed_addr constant [6 x i8] c", Y: \00", align 1
@.str.20 = private unnamed_addr constant [9 x i8] c", diff: \00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_axpy.cu, i8* null }]

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #0 section ".text.startup" {
  call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit)
  %1 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i32 0, i32 0), i8* @__dso_handle) #3
  ret void
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z8axpy_cpuPfS_fiS_(float*, float*, float, i32, float*) #4 {
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
  store i32 0, i32* %11, align 4
  br label %12

12:                                               ; preds = %34, %5
  %13 = load i32, i32* %11, align 4
  %14 = load i32, i32* %9, align 4
  %15 = icmp slt i32 %13, %14
  br i1 %15, label %16, label %37

16:                                               ; preds = %12
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
  br label %34

34:                                               ; preds = %16
  %35 = load i32, i32* %11, align 4
  %36 = add nsw i32 %35, 1
  store i32 %36, i32* %11, align 4
  br label %12

37:                                               ; preds = %12
  ret void
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z4axpyPfS_fiS_(float*, float*, float, i32, float*) #5 {
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float, align 4
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca %struct.dim3, align 8
  %12 = alloca %struct.dim3, align 8
  %13 = alloca i64, align 8
  %14 = alloca i8*, align 8
  %15 = alloca { i64, i32 }, align 8
  %16 = alloca { i64, i32 }, align 8
  store float* %0, float** %6, align 8
  store float* %1, float** %7, align 8
  store float %2, float* %8, align 4
  store i32 %3, i32* %9, align 4
  store float* %4, float** %10, align 8
  %17 = alloca i8*, i64 5, align 16
  %18 = bitcast float** %6 to i8*
  %19 = getelementptr i8*, i8** %17, i32 0
  store i8* %18, i8** %19
  %20 = bitcast float** %7 to i8*
  %21 = getelementptr i8*, i8** %17, i32 1
  store i8* %20, i8** %21
  %22 = bitcast float* %8 to i8*
  %23 = getelementptr i8*, i8** %17, i32 2
  store i8* %22, i8** %23
  %24 = bitcast i32* %9 to i8*
  %25 = getelementptr i8*, i8** %17, i32 3
  store i8* %24, i8** %25
  %26 = bitcast float** %10 to i8*
  %27 = getelementptr i8*, i8** %17, i32 4
  store i8* %26, i8** %27
  %28 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %11, %struct.dim3* %12, i64* %13, i8** %14)
  %29 = load i64, i64* %13, align 8
  %30 = load i8*, i8** %14, align 8
  %31 = bitcast { i64, i32 }* %15 to i8*
  %32 = bitcast %struct.dim3* %11 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %31, i8* align 8 %32, i64 12, i1 false)
  %33 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %15, i32 0, i32 0
  %34 = load i64, i64* %33, align 8
  %35 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %15, i32 0, i32 1
  %36 = load i32, i32* %35, align 8
  %37 = bitcast { i64, i32 }* %16 to i8*
  %38 = bitcast %struct.dim3* %12 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %37, i8* align 8 %38, i64 12, i1 false)
  %39 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 0
  %40 = load i64, i64* %39, align 8
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 1
  %42 = load i32, i32* %41, align 8
  %43 = bitcast i8* %30 to %struct.CUstream_st*
  %44 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, float, i32, float*)* @_Z4axpyPfS_fiS_ to i8*), i64 %34, i32 %36, i64 %40, i32 %42, i8** %17, i64 %29, %struct.CUstream_st* %43)
  br label %45

45:                                               ; preds = %5
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**)

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #6

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z12axpy_checkedPfS_fiS_(float*, float*, float, i32, float*) #5 {
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca float, align 4
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca %struct.dim3, align 8
  %12 = alloca %struct.dim3, align 8
  %13 = alloca i64, align 8
  %14 = alloca i8*, align 8
  %15 = alloca { i64, i32 }, align 8
  %16 = alloca { i64, i32 }, align 8
  store float* %0, float** %6, align 8
  store float* %1, float** %7, align 8
  store float %2, float* %8, align 4
  store i32 %3, i32* %9, align 4
  store float* %4, float** %10, align 8
  %17 = alloca i8*, i64 5, align 16
  %18 = bitcast float** %6 to i8*
  %19 = getelementptr i8*, i8** %17, i32 0
  store i8* %18, i8** %19
  %20 = bitcast float** %7 to i8*
  %21 = getelementptr i8*, i8** %17, i32 1
  store i8* %20, i8** %21
  %22 = bitcast float* %8 to i8*
  %23 = getelementptr i8*, i8** %17, i32 2
  store i8* %22, i8** %23
  %24 = bitcast i32* %9 to i8*
  %25 = getelementptr i8*, i8** %17, i32 3
  store i8* %24, i8** %25
  %26 = bitcast float** %10 to i8*
  %27 = getelementptr i8*, i8** %17, i32 4
  store i8* %26, i8** %27
  %28 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %11, %struct.dim3* %12, i64* %13, i8** %14)
  %29 = load i64, i64* %13, align 8
  %30 = load i8*, i8** %14, align 8
  %31 = bitcast { i64, i32 }* %15 to i8*
  %32 = bitcast %struct.dim3* %11 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %31, i8* align 8 %32, i64 12, i1 false)
  %33 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %15, i32 0, i32 0
  %34 = load i64, i64* %33, align 8
  %35 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %15, i32 0, i32 1
  %36 = load i32, i32* %35, align 8
  %37 = bitcast { i64, i32 }* %16 to i8*
  %38 = bitcast %struct.dim3* %12 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %37, i8* align 8 %38, i64 12, i1 false)
  %39 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 0
  %40 = load i64, i64* %39, align 8
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 1
  %42 = load i32, i32* %41, align 8
  %43 = bitcast i8* %30 to %struct.CUstream_st*
  %44 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, float, i32, float*)* @_Z12axpy_checkedPfS_fiS_ to i8*), i64 %34, i32 %36, i64 %40, i32 %42, i8** %17, i64 %29, %struct.CUstream_st* %43)
  br label %45

45:                                               ; preds = %5
  ret void
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z14axpy_with_argsPfiS_ifS_i(float*, i32, float*, i32, float, float*, i32) #5 {
  %8 = alloca float*, align 8
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca i32, align 4
  %12 = alloca float, align 4
  %13 = alloca float*, align 8
  %14 = alloca i32, align 4
  %15 = alloca %struct.dim3, align 8
  %16 = alloca %struct.dim3, align 8
  %17 = alloca i64, align 8
  %18 = alloca i8*, align 8
  %19 = alloca { i64, i32 }, align 8
  %20 = alloca { i64, i32 }, align 8
  store float* %0, float** %8, align 8
  store i32 %1, i32* %9, align 4
  store float* %2, float** %10, align 8
  store i32 %3, i32* %11, align 4
  store float %4, float* %12, align 4
  store float* %5, float** %13, align 8
  store i32 %6, i32* %14, align 4
  %21 = alloca i8*, i64 7, align 16
  %22 = bitcast float** %8 to i8*
  %23 = getelementptr i8*, i8** %21, i32 0
  store i8* %22, i8** %23
  %24 = bitcast i32* %9 to i8*
  %25 = getelementptr i8*, i8** %21, i32 1
  store i8* %24, i8** %25
  %26 = bitcast float** %10 to i8*
  %27 = getelementptr i8*, i8** %21, i32 2
  store i8* %26, i8** %27
  %28 = bitcast i32* %11 to i8*
  %29 = getelementptr i8*, i8** %21, i32 3
  store i8* %28, i8** %29
  %30 = bitcast float* %12 to i8*
  %31 = getelementptr i8*, i8** %21, i32 4
  store i8* %30, i8** %31
  %32 = bitcast float** %13 to i8*
  %33 = getelementptr i8*, i8** %21, i32 5
  store i8* %32, i8** %33
  %34 = bitcast i32* %14 to i8*
  %35 = getelementptr i8*, i8** %21, i32 6
  store i8* %34, i8** %35
  %36 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %15, %struct.dim3* %16, i64* %17, i8** %18)
  %37 = load i64, i64* %17, align 8
  %38 = load i8*, i8** %18, align 8
  %39 = bitcast { i64, i32 }* %19 to i8*
  %40 = bitcast %struct.dim3* %15 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %39, i8* align 8 %40, i64 12, i1 false)
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %19, i32 0, i32 0
  %42 = load i64, i64* %41, align 8
  %43 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %19, i32 0, i32 1
  %44 = load i32, i32* %43, align 8
  %45 = bitcast { i64, i32 }* %20 to i8*
  %46 = bitcast %struct.dim3* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %45, i8* align 8 %46, i64 12, i1 false)
  %47 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 0
  %48 = load i64, i64* %47, align 8
  %49 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 1
  %50 = load i32, i32* %49, align 8
  %51 = bitcast i8* %38 to %struct.CUstream_st*
  %52 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, i32, float*, i32, float, float*, i32)* @_Z14axpy_with_argsPfiS_ifS_i to i8*), i64 %42, i32 %44, i64 %48, i32 %50, i8** %21, i64 %37, %struct.CUstream_st* %51)
  br label %53

53:                                               ; preds = %7
  ret void
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z22axpy_with_args_checkedPfiS_ifS_i(float*, i32, float*, i32, float, float*, i32) #5 {
  %8 = alloca float*, align 8
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca i32, align 4
  %12 = alloca float, align 4
  %13 = alloca float*, align 8
  %14 = alloca i32, align 4
  %15 = alloca %struct.dim3, align 8
  %16 = alloca %struct.dim3, align 8
  %17 = alloca i64, align 8
  %18 = alloca i8*, align 8
  %19 = alloca { i64, i32 }, align 8
  %20 = alloca { i64, i32 }, align 8
  store float* %0, float** %8, align 8
  store i32 %1, i32* %9, align 4
  store float* %2, float** %10, align 8
  store i32 %3, i32* %11, align 4
  store float %4, float* %12, align 4
  store float* %5, float** %13, align 8
  store i32 %6, i32* %14, align 4
  %21 = alloca i8*, i64 7, align 16
  %22 = bitcast float** %8 to i8*
  %23 = getelementptr i8*, i8** %21, i32 0
  store i8* %22, i8** %23
  %24 = bitcast i32* %9 to i8*
  %25 = getelementptr i8*, i8** %21, i32 1
  store i8* %24, i8** %25
  %26 = bitcast float** %10 to i8*
  %27 = getelementptr i8*, i8** %21, i32 2
  store i8* %26, i8** %27
  %28 = bitcast i32* %11 to i8*
  %29 = getelementptr i8*, i8** %21, i32 3
  store i8* %28, i8** %29
  %30 = bitcast float* %12 to i8*
  %31 = getelementptr i8*, i8** %21, i32 4
  store i8* %30, i8** %31
  %32 = bitcast float** %13 to i8*
  %33 = getelementptr i8*, i8** %21, i32 5
  store i8* %32, i8** %33
  %34 = bitcast i32* %14 to i8*
  %35 = getelementptr i8*, i8** %21, i32 6
  store i8* %34, i8** %35
  %36 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %15, %struct.dim3* %16, i64* %17, i8** %18)
  %37 = load i64, i64* %17, align 8
  %38 = load i8*, i8** %18, align 8
  %39 = bitcast { i64, i32 }* %19 to i8*
  %40 = bitcast %struct.dim3* %15 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %39, i8* align 8 %40, i64 12, i1 false)
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %19, i32 0, i32 0
  %42 = load i64, i64* %41, align 8
  %43 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %19, i32 0, i32 1
  %44 = load i32, i32* %43, align 8
  %45 = bitcast { i64, i32 }* %20 to i8*
  %46 = bitcast %struct.dim3* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %45, i8* align 8 %46, i64 12, i1 false)
  %47 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 0
  %48 = load i64, i64* %47, align 8
  %49 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 1
  %50 = load i32, i32* %49, align 8
  %51 = bitcast i8* %38 to %struct.CUstream_st*
  %52 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, i32, float*, i32, float, float*, i32)* @_Z22axpy_with_args_checkedPfiS_ifS_i to i8*), i64 %42, i32 %44, i64 %48, i32 %50, i8** %21, i64 %37, %struct.CUstream_st* %51)
  br label %53

53:                                               ; preds = %7
  ret void
}

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main(i32, i8**) #7 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i8, align 1
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca float, align 4
  %12 = alloca float*, align 8
  %13 = alloca float*, align 8
  %14 = alloca float*, align 8
  %15 = alloca float*, align 8
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca float*, align 8
  %19 = alloca float*, align 8
  %20 = alloca float*, align 8
  %21 = alloca %"struct.std::chrono::time_point", align 8
  %22 = alloca %"struct.std::chrono::time_point", align 8
  %23 = alloca i64, align 8
  %24 = alloca %"struct.std::chrono::duration.0", align 8
  %25 = alloca %"struct.std::chrono::duration", align 8
  %26 = alloca i32, align 4
  %27 = alloca %"struct.std::chrono::time_point", align 8
  %28 = alloca %struct.dim3, align 4
  %29 = alloca %struct.dim3, align 4
  %30 = alloca { i64, i32 }, align 4
  %31 = alloca { i64, i32 }, align 4
  %32 = alloca %"struct.std::chrono::time_point", align 8
  %33 = alloca %"struct.std::chrono::duration.0", align 8
  %34 = alloca %"struct.std::chrono::duration", align 8
  %35 = alloca i32, align 4
  %36 = alloca i32, align 4
  %37 = alloca i32, align 4
  %38 = alloca %"struct.std::chrono::time_point", align 8
  %39 = alloca %struct.dim3, align 4
  %40 = alloca %struct.dim3, align 4
  %41 = alloca { i64, i32 }, align 4
  %42 = alloca { i64, i32 }, align 4
  %43 = alloca %"struct.std::chrono::time_point", align 8
  %44 = alloca %"struct.std::chrono::duration.0", align 8
  %45 = alloca %"struct.std::chrono::duration", align 8
  %46 = alloca i32, align 4
  %47 = alloca i32, align 4
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  store i8 0, i8* %6, align 1
  store i32 10, i32* %7, align 4
  store i32 0, i32* %9, align 4
  br label %48

48:                                               ; preds = %60, %2
  %49 = load i32, i32* %4, align 4
  %50 = load i8**, i8*** %5, align 8
  %51 = call i32 @getopt_long(i32 %49, i8** %50, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.2, i64 0, i64 0), %struct.option* getelementptr inbounds ([3 x %struct.option], [3 x %struct.option]* @_ZZ4mainE12long_options, i64 0, i64 0), i32* %9) #3
  store i32 %51, i32* %8, align 4
  %52 = icmp ne i32 %51, -1
  br i1 %52, label %53, label %61

53:                                               ; preds = %48
  %54 = load i32, i32* %8, align 4
  switch i32 %54, label %59 [
    i32 104, label %55
    i32 115, label %56
  ]

55:                                               ; preds = %53
  store i8 1, i8* %6, align 1
  br label %60

56:                                               ; preds = %53
  %57 = load i8*, i8** @optarg, align 8
  %58 = call i32 @atoi(i8* %57) #11
  store i32 %58, i32* %7, align 4
  br label %60

59:                                               ; preds = %53
  store i32 0, i32* %3, align 4
  br label %353

60:                                               ; preds = %56, %55
  br label %48

61:                                               ; preds = %48
  %62 = load i32, i32* %7, align 4
  %63 = add nsw i32 %62, 128
  %64 = sub nsw i32 %63, 1
  %65 = sdiv i32 %64, 128
  store i32 %65, i32* %10, align 4
  store float 0.000000e+00, float* %11, align 4
  %66 = load i32, i32* %7, align 4
  %67 = sext i32 %66 to i64
  %68 = mul i64 %67, 4
  %69 = call noalias i8* @malloc(i64 %68) #3
  %70 = bitcast i8* %69 to float*
  store float* %70, float** %12, align 8
  %71 = load i32, i32* %7, align 4
  %72 = sext i32 %71 to i64
  %73 = mul i64 %72, 4
  %74 = call noalias i8* @malloc(i64 %73) #3
  %75 = bitcast i8* %74 to float*
  store float* %75, float** %13, align 8
  %76 = load i32, i32* %7, align 4
  %77 = sext i32 %76 to i64
  %78 = call noalias i8* @calloc(i64 %77, i64 4) #3
  %79 = bitcast i8* %78 to float*
  store float* %79, float** %14, align 8
  %80 = load i32, i32* %7, align 4
  %81 = sext i32 %80 to i64
  %82 = call noalias i8* @calloc(i64 %81, i64 4) #3
  %83 = bitcast i8* %82 to float*
  store float* %83, float** %15, align 8
  call void @_Z20create_sample_vectorPfibb(float* %11, i32 1, i1 zeroext true, i1 zeroext false)
  %84 = load float*, float** %12, align 8
  %85 = load i32, i32* %7, align 4
  call void @_Z20create_sample_vectorPfibb(float* %84, i32 %85, i1 zeroext true, i1 zeroext true)
  %86 = load float*, float** %13, align 8
  %87 = load i32, i32* %7, align 4
  call void @_Z20create_sample_vectorPfibb(float* %86, i32 %87, i1 zeroext true, i1 zeroext true)
  %88 = load i8, i8* %6, align 1
  %89 = trunc i8 %88 to i1
  br i1 %89, label %90, label %107

90:                                               ; preds = %61
  %91 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i64 0, i64 0))
  %92 = load float, float* %11, align 4
  %93 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* %91, float %92)
  %94 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %93, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %95 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i64 0, i64 0))
  %96 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %95, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %97 = load float*, float** %12, align 8
  store i32 20, i32* %16, align 4
  %98 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %16, i32* dereferenceable(4) %7)
  %99 = load i32, i32* %98, align 4
  call void @_Z19print_array_indexedIfEvPT_i(float* %97, i32 %99)
  %100 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.5, i64 0, i64 0))
  %101 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %100, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %102 = load float*, float** %13, align 8
  store i32 20, i32* %17, align 4
  %103 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %17, i32* dereferenceable(4) %7)
  %104 = load i32, i32* %103, align 4
  call void @_Z19print_array_indexedIfEvPT_i(float* %102, i32 %104)
  %105 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.6, i64 0, i64 0))
  %106 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %105, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %107

107:                                              ; preds = %90, %61
  %108 = load i32, i32* %7, align 4
  %109 = sext i32 %108 to i64
  %110 = mul i64 4, %109
  %111 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %18, i64 %110)
  %112 = load i32, i32* %7, align 4
  %113 = sext i32 %112 to i64
  %114 = mul i64 4, %113
  %115 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %19, i64 %114)
  %116 = load i32, i32* %7, align 4
  %117 = sext i32 %116 to i64
  %118 = mul i64 4, %117
  %119 = call i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** %20, i64 %118)
  %120 = load float*, float** %18, align 8
  %121 = bitcast float* %120 to i8*
  %122 = load float*, float** %12, align 8
  %123 = bitcast float* %122 to i8*
  %124 = load i32, i32* %7, align 4
  %125 = sext i32 %124 to i64
  %126 = mul i64 %125, 4
  %127 = call i32 @cudaMemcpy(i8* %121, i8* %123, i64 %126, i32 1)
  %128 = load float*, float** %19, align 8
  %129 = bitcast float* %128 to i8*
  %130 = load float*, float** %13, align 8
  %131 = bitcast float* %130 to i8*
  %132 = load i32, i32* %7, align 4
  %133 = sext i32 %132 to i64
  %134 = mul i64 %133, 4
  %135 = call i32 @cudaMemcpy(i8* %129, i8* %131, i64 %134, i32 1)
  %136 = load float*, float** %20, align 8
  %137 = bitcast float* %136 to i8*
  %138 = load float*, float** %15, align 8
  %139 = bitcast float* %138 to i8*
  %140 = load i32, i32* %7, align 4
  %141 = sext i32 %140 to i64
  %142 = mul i64 %141, 4
  %143 = call i32 @cudaMemcpy(i8* %137, i8* %139, i64 %142, i32 1)
  %144 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #3
  %145 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %21, i32 0, i32 0
  %146 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %145, i32 0, i32 0
  store i64 %144, i64* %146, align 8
  %147 = load float*, float** %12, align 8
  %148 = load float*, float** %13, align 8
  %149 = load float, float* %11, align 4
  %150 = load i32, i32* %7, align 4
  %151 = load float*, float** %14, align 8
  call void @_Z8axpy_cpuPfS_fiS_(float* %147, float* %148, float %149, i32 %150, float* %151)
  %152 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #3
  %153 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %22, i32 0, i32 0
  %154 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %153, i32 0, i32 0
  store i64 %152, i64* %154, align 8
  %155 = call i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* dereferenceable(8) %22, %"struct.std::chrono::time_point"* dereferenceable(8) %21)
  %156 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %25, i32 0, i32 0
  store i64 %155, i64* %156, align 8
  %157 = call i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* dereferenceable(8) %25)
  %158 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %24, i32 0, i32 0
  store i64 %157, i64* %158, align 8
  %159 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"* %24)
  store i64 %159, i64* %23, align 8
  %160 = load i8, i8* %6, align 1
  %161 = trunc i8 %160 to i1
  br i1 %161, label %162, label %175

162:                                              ; preds = %107
  %163 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.7, i64 0, i64 0))
  %164 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %163, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %165 = load float*, float** %14, align 8
  store i32 20, i32* %26, align 4
  %166 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %26, i32* dereferenceable(4) %7)
  %167 = load i32, i32* %166, align 4
  call void @_Z19print_array_indexedIfEvPT_i(float* %165, i32 %167)
  %168 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.8, i64 0, i64 0))
  %169 = load i64, i64* %23, align 8
  %170 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"* %168, i64 %169)
  %171 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %170, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0))
  %172 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %171, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %173 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.6, i64 0, i64 0))
  %174 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %173, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %175

175:                                              ; preds = %162, %107
  %176 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #3
  %177 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %27, i32 0, i32 0
  %178 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %177, i32 0, i32 0
  store i64 %176, i64* %178, align 8
  %179 = bitcast %"struct.std::chrono::time_point"* %21 to i8*
  %180 = bitcast %"struct.std::chrono::time_point"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %179, i8* align 8 %180, i64 8, i1 false)
  %181 = load i32, i32* %10, align 4
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %28, i32 %181, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %29, i32 128, i32 1, i32 1)
  %182 = bitcast { i64, i32 }* %30 to i8*
  %183 = bitcast %struct.dim3* %28 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %182, i8* align 4 %183, i64 12, i1 false)
  %184 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %30, i32 0, i32 0
  %185 = load i64, i64* %184, align 4
  %186 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %30, i32 0, i32 1
  %187 = load i32, i32* %186, align 4
  %188 = bitcast { i64, i32 }* %31 to i8*
  %189 = bitcast %struct.dim3* %29 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %188, i8* align 4 %189, i64 12, i1 false)
  %190 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %31, i32 0, i32 0
  %191 = load i64, i64* %190, align 4
  %192 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %31, i32 0, i32 1
  %193 = load i32, i32* %192, align 4
  %194 = call i32 @__cudaPushCallConfiguration(i64 %185, i32 %187, i64 %191, i32 %193, i64 0, i8* null)
  %195 = icmp ne i32 %194, 0
  br i1 %195, label %202, label %196

196:                                              ; preds = %175
  %197 = load float*, float** %18, align 8
  %198 = load float*, float** %19, align 8
  %199 = load float, float* %11, align 4
  %200 = load i32, i32* %7, align 4
  %201 = load float*, float** %20, align 8
  call void @_Z12axpy_checkedPfS_fiS_(float* %197, float* %198, float %199, i32 %200, float* %201)
  br label %202

202:                                              ; preds = %196, %175
  %203 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #3
  %204 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %32, i32 0, i32 0
  %205 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %204, i32 0, i32 0
  store i64 %203, i64* %205, align 8
  %206 = bitcast %"struct.std::chrono::time_point"* %22 to i8*
  %207 = bitcast %"struct.std::chrono::time_point"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %206, i8* align 8 %207, i64 8, i1 false)
  %208 = call i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* dereferenceable(8) %22, %"struct.std::chrono::time_point"* dereferenceable(8) %21)
  %209 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %34, i32 0, i32 0
  store i64 %208, i64* %209, align 8
  %210 = call i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* dereferenceable(8) %34)
  %211 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %33, i32 0, i32 0
  store i64 %210, i64* %211, align 8
  %212 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"* %33)
  store i64 %212, i64* %23, align 8
  %213 = load float*, float** %15, align 8
  %214 = bitcast float* %213 to i8*
  %215 = load float*, float** %20, align 8
  %216 = bitcast float* %215 to i8*
  %217 = load i32, i32* %7, align 4
  %218 = sext i32 %217 to i64
  %219 = mul i64 %218, 4
  %220 = call i32 @cudaMemcpy(i8* %214, i8* %216, i64 %219, i32 2)
  %221 = call i32 @cudaGetLastError()
  store i32 %221, i32* %35, align 4
  %222 = load i32, i32* %35, align 4
  %223 = icmp ne i32 %222, 0
  br i1 %223, label %224, label %234

224:                                              ; preds = %202
  %225 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cerr, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.10, i64 0, i64 0))
  %226 = load i32, i32* %35, align 4
  %227 = call i8* @cudaGetErrorString(i32 %226)
  %228 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %225, i8* %227)
  %229 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %228, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0))
  %230 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %229, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.12, i64 0, i64 0))
  %231 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* dereferenceable(272) %230, i8 signext 58)
  %232 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* %231, i32 140)
  %233 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %232, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %234

234:                                              ; preds = %224, %202
  %235 = load i8, i8* %6, align 1
  %236 = trunc i8 %235 to i1
  br i1 %236, label %237, label %248

237:                                              ; preds = %234
  %238 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.13, i64 0, i64 0))
  %239 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %238, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %240 = load float*, float** %15, align 8
  store i32 20, i32* %36, align 4
  %241 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %36, i32* dereferenceable(4) %7)
  %242 = load i32, i32* %241, align 4
  call void @_Z19print_array_indexedIfEvPT_i(float* %240, i32 %242)
  %243 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.8, i64 0, i64 0))
  %244 = load i64, i64* %23, align 8
  %245 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"* %243, i64 %244)
  %246 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %245, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0))
  %247 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %246, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %248

248:                                              ; preds = %237, %234
  %249 = load float*, float** %14, align 8
  %250 = load float*, float** %15, align 8
  %251 = load i32, i32* %7, align 4
  %252 = load i8, i8* %6, align 1
  %253 = trunc i8 %252 to i1
  %254 = call i32 @_Z20check_array_equalityIfEiPT_S1_ifb(float* %249, float* %250, i32 %251, float 0x3E45798EE0000000, i1 zeroext %253)
  store i32 %254, i32* %37, align 4
  %255 = load i8, i8* %6, align 1
  %256 = trunc i8 %255 to i1
  br i1 %256, label %257, label %264

257:                                              ; preds = %248
  %258 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.14, i64 0, i64 0))
  %259 = load i32, i32* %37, align 4
  %260 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* %258, i32 %259)
  %261 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %260, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %262 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.6, i64 0, i64 0))
  %263 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %262, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %264

264:                                              ; preds = %257, %248
  %265 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #3
  %266 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %38, i32 0, i32 0
  %267 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %266, i32 0, i32 0
  store i64 %265, i64* %267, align 8
  %268 = bitcast %"struct.std::chrono::time_point"* %21 to i8*
  %269 = bitcast %"struct.std::chrono::time_point"* %38 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %268, i8* align 8 %269, i64 8, i1 false)
  %270 = load i32, i32* %10, align 4
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %39, i32 %270, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %40, i32 128, i32 1, i32 1)
  %271 = bitcast { i64, i32 }* %41 to i8*
  %272 = bitcast %struct.dim3* %39 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %271, i8* align 4 %272, i64 12, i1 false)
  %273 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %41, i32 0, i32 0
  %274 = load i64, i64* %273, align 4
  %275 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %41, i32 0, i32 1
  %276 = load i32, i32* %275, align 4
  %277 = bitcast { i64, i32 }* %42 to i8*
  %278 = bitcast %struct.dim3* %40 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %277, i8* align 4 %278, i64 12, i1 false)
  %279 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %42, i32 0, i32 0
  %280 = load i64, i64* %279, align 4
  %281 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %42, i32 0, i32 1
  %282 = load i32, i32* %281, align 4
  %283 = call i32 @__cudaPushCallConfiguration(i64 %274, i32 %276, i64 %280, i32 %282, i64 0, i8* null)
  %284 = icmp ne i32 %283, 0
  br i1 %284, label %291, label %285

285:                                              ; preds = %264
  %286 = load float*, float** %18, align 8
  %287 = load float*, float** %19, align 8
  %288 = load float, float* %11, align 4
  %289 = load i32, i32* %7, align 4
  %290 = load float*, float** %20, align 8
  call void @_Z4axpyPfS_fiS_(float* %286, float* %287, float %288, i32 %289, float* %290)
  br label %291

291:                                              ; preds = %285, %264
  %292 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #3
  %293 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %43, i32 0, i32 0
  %294 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %293, i32 0, i32 0
  store i64 %292, i64* %294, align 8
  %295 = bitcast %"struct.std::chrono::time_point"* %22 to i8*
  %296 = bitcast %"struct.std::chrono::time_point"* %43 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %295, i8* align 8 %296, i64 8, i1 false)
  %297 = call i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* dereferenceable(8) %22, %"struct.std::chrono::time_point"* dereferenceable(8) %21)
  %298 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %45, i32 0, i32 0
  store i64 %297, i64* %298, align 8
  %299 = call i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* dereferenceable(8) %45)
  %300 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %44, i32 0, i32 0
  store i64 %299, i64* %300, align 8
  %301 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"* %44)
  store i64 %301, i64* %23, align 8
  %302 = load float*, float** %15, align 8
  %303 = bitcast float* %302 to i8*
  %304 = load float*, float** %20, align 8
  %305 = bitcast float* %304 to i8*
  %306 = load i32, i32* %7, align 4
  %307 = sext i32 %306 to i64
  %308 = mul i64 %307, 4
  %309 = call i32 @cudaMemcpy(i8* %303, i8* %305, i64 %308, i32 2)
  %310 = call i32 @cudaGetLastError()
  store i32 %310, i32* %46, align 4
  %311 = load i32, i32* %46, align 4
  %312 = icmp ne i32 %311, 0
  br i1 %312, label %313, label %323

313:                                              ; preds = %291
  %314 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cerr, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.10, i64 0, i64 0))
  %315 = load i32, i32* %46, align 4
  %316 = call i8* @cudaGetErrorString(i32 %315)
  %317 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %314, i8* %316)
  %318 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %317, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0))
  %319 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %318, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.12, i64 0, i64 0))
  %320 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* dereferenceable(272) %319, i8 signext 58)
  %321 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* %320, i32 159)
  %322 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %321, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %323

323:                                              ; preds = %313, %291
  %324 = load i8, i8* %6, align 1
  %325 = trunc i8 %324 to i1
  br i1 %325, label %326, label %337

326:                                              ; preds = %323
  %327 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.15, i64 0, i64 0))
  %328 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %327, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %329 = load float*, float** %15, align 8
  store i32 20, i32* %47, align 4
  %330 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4) %47, i32* dereferenceable(4) %7)
  %331 = load i32, i32* %330, align 4
  call void @_Z19print_array_indexedIfEvPT_i(float* %329, i32 %331)
  %332 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.8, i64 0, i64 0))
  %333 = load i64, i64* %23, align 8
  %334 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"* %332, i64 %333)
  %335 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %334, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0))
  %336 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %335, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %337

337:                                              ; preds = %326, %323
  %338 = load float*, float** %14, align 8
  %339 = load float*, float** %15, align 8
  %340 = load i32, i32* %7, align 4
  %341 = load i8, i8* %6, align 1
  %342 = trunc i8 %341 to i1
  %343 = call i32 @_Z20check_array_equalityIfEiPT_S1_ifb(float* %338, float* %339, i32 %340, float 0x3E45798EE0000000, i1 zeroext %342)
  store i32 %343, i32* %37, align 4
  %344 = load i8, i8* %6, align 1
  %345 = trunc i8 %344 to i1
  br i1 %345, label %346, label %353

346:                                              ; preds = %337
  %347 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.14, i64 0, i64 0))
  %348 = load i32, i32* %37, align 4
  %349 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* %347, i32 %348)
  %350 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %349, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %351 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.6, i64 0, i64 0))
  %352 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %351, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %353

353:                                              ; preds = %59, %346, %337
  %354 = load i32, i32* %3, align 4
  ret i32 %354
}

; Function Attrs: nounwind
declare dso_local i32 @getopt_long(i32, i8**, i8*, %struct.option*, i32*) #2

; Function Attrs: nounwind readonly
declare dso_local i32 @atoi(i8*) #8

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #2

; Function Attrs: nounwind
declare dso_local noalias i8* @calloc(i64, i64) #2

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_Z20create_sample_vectorPfibb(float*, i32, i1 zeroext, i1 zeroext) #5 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %5 = alloca float*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i8, align 1
  %8 = alloca i8, align 1
  %9 = alloca %"class.std::random_device", align 8
  %10 = alloca %"class.std::__cxx11::basic_string", align 8
  %11 = alloca %"class.std::allocator", align 1
  %12 = alloca i8*
  %13 = alloca i32
  %14 = alloca %"class.std::mersenne_twister_engine", align 8
  %15 = alloca %"class.std::uniform_real_distribution", align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca float, align 4
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  store float* %0, float** %5, align 8
  store i32 %1, i32* %6, align 4
  %21 = zext i1 %2 to i8
  store i8 %21, i8* %7, align 1
  %22 = zext i1 %3 to i8
  store i8 %22, i8* %8, align 1
  %23 = load i8, i8* %7, align 1
  %24 = trunc i8 %23 to i1
  br i1 %24, label %25, label %61

25:                                               ; preds = %4
  call void @_ZNSaIcEC1Ev(%"class.std::allocator"* %11) #3
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EPKcRKS3_(%"class.std::__cxx11::basic_string"* %10, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.16, i64 0, i64 0), %"class.std::allocator"* dereferenceable(1) %11)
          to label %26 unwind label %47

26:                                               ; preds = %25
  invoke void @_ZNSt13random_deviceC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"* %9, %"class.std::__cxx11::basic_string"* dereferenceable(32) %10)
          to label %27 unwind label %51

27:                                               ; preds = %26
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_string"* %10) #3
  call void @_ZNSaIcED1Ev(%"class.std::allocator"* %11) #3
  %28 = invoke i32 @_ZNSt13random_deviceclEv(%"class.std::random_device"* %9)
          to label %29 unwind label %56

29:                                               ; preds = %27
  %30 = zext i32 %28 to i64
  invoke void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em(%"class.std::mersenne_twister_engine"* %14, i64 %30)
          to label %31 unwind label %56

31:                                               ; preds = %29
  invoke void @_ZNSt25uniform_real_distributionIfEC2Eff(%"class.std::uniform_real_distribution"* %15, float 0.000000e+00, float 1.000000e+00)
          to label %32 unwind label %56

32:                                               ; preds = %31
  store i32 0, i32* %16, align 4
  br label %33

33:                                               ; preds = %44, %32
  %34 = load i32, i32* %16, align 4
  %35 = load i32, i32* %6, align 4
  %36 = icmp slt i32 %34, %35
  br i1 %36, label %37, label %60

37:                                               ; preds = %33
  %38 = invoke float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_(%"class.std::uniform_real_distribution"* %15, %"class.std::mersenne_twister_engine"* dereferenceable(5000) %14)
          to label %39 unwind label %56

39:                                               ; preds = %37
  %40 = load float*, float** %5, align 8
  %41 = load i32, i32* %16, align 4
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds float, float* %40, i64 %42
  store float %38, float* %43, align 4
  br label %44

44:                                               ; preds = %39
  %45 = load i32, i32* %16, align 4
  %46 = add nsw i32 %45, 1
  store i32 %46, i32* %16, align 4
  br label %33

47:                                               ; preds = %25
  %48 = landingpad { i8*, i32 }
          cleanup
  %49 = extractvalue { i8*, i32 } %48, 0
  store i8* %49, i8** %12, align 8
  %50 = extractvalue { i8*, i32 } %48, 1
  store i32 %50, i32* %13, align 4
  br label %55

51:                                               ; preds = %26
  %52 = landingpad { i8*, i32 }
          cleanup
  %53 = extractvalue { i8*, i32 } %52, 0
  store i8* %53, i8** %12, align 8
  %54 = extractvalue { i8*, i32 } %52, 1
  store i32 %54, i32* %13, align 4
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_string"* %10) #3
  br label %55

55:                                               ; preds = %51, %47
  call void @_ZNSaIcED1Ev(%"class.std::allocator"* %11) #3
  br label %112

56:                                               ; preds = %37, %31, %29, %27
  %57 = landingpad { i8*, i32 }
          cleanup
  %58 = extractvalue { i8*, i32 } %57, 0
  store i8* %58, i8** %12, align 8
  %59 = extractvalue { i8*, i32 } %57, 1
  store i32 %59, i32* %13, align 4
  call void @_ZNSt13random_deviceD2Ev(%"class.std::random_device"* %9) #3
  br label %112

60:                                               ; preds = %33
  call void @_ZNSt13random_deviceD2Ev(%"class.std::random_device"* %9) #3
  br label %75

61:                                               ; preds = %4
  store i32 0, i32* %17, align 4
  br label %62

62:                                               ; preds = %71, %61
  %63 = load i32, i32* %17, align 4
  %64 = load i32, i32* %6, align 4
  %65 = icmp slt i32 %63, %64
  br i1 %65, label %66, label %74

66:                                               ; preds = %62
  %67 = load float*, float** %5, align 8
  %68 = load i32, i32* %17, align 4
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds float, float* %67, i64 %69
  store float 1.000000e+00, float* %70, align 4
  br label %71

71:                                               ; preds = %66
  %72 = load i32, i32* %17, align 4
  %73 = add nsw i32 %72, 1
  store i32 %73, i32* %17, align 4
  br label %62

74:                                               ; preds = %62
  br label %75

75:                                               ; preds = %74, %60
  %76 = load i8, i8* %8, align 1
  %77 = trunc i8 %76 to i1
  br i1 %77, label %78, label %111

78:                                               ; preds = %75
  store float 0.000000e+00, float* %18, align 4
  store i32 0, i32* %19, align 4
  br label %79

79:                                               ; preds = %91, %78
  %80 = load i32, i32* %19, align 4
  %81 = load i32, i32* %6, align 4
  %82 = icmp slt i32 %80, %81
  br i1 %82, label %83, label %94

83:                                               ; preds = %79
  %84 = load float*, float** %5, align 8
  %85 = load i32, i32* %19, align 4
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds float, float* %84, i64 %86
  %88 = load float, float* %87, align 4
  %89 = load float, float* %18, align 4
  %90 = fadd contract float %89, %88
  store float %90, float* %18, align 4
  br label %91

91:                                               ; preds = %83
  %92 = load i32, i32* %19, align 4
  %93 = add nsw i32 %92, 1
  store i32 %93, i32* %19, align 4
  br label %79

94:                                               ; preds = %79
  store i32 0, i32* %20, align 4
  br label %95

95:                                               ; preds = %107, %94
  %96 = load i32, i32* %20, align 4
  %97 = load i32, i32* %6, align 4
  %98 = icmp slt i32 %96, %97
  br i1 %98, label %99, label %110

99:                                               ; preds = %95
  %100 = load float, float* %18, align 4
  %101 = load float*, float** %5, align 8
  %102 = load i32, i32* %20, align 4
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds float, float* %101, i64 %103
  %105 = load float, float* %104, align 4
  %106 = fdiv float %105, %100
  store float %106, float* %104, align 4
  br label %107

107:                                              ; preds = %99
  %108 = load i32, i32* %20, align 4
  %109 = add nsw i32 %108, 1
  store i32 %109, i32* %20, align 4
  br label %95

110:                                              ; preds = %95
  br label %111

111:                                              ; preds = %110, %75
  ret void

112:                                              ; preds = %56, %55
  %113 = load i8*, i8** %12, align 8
  %114 = load i32, i32* %13, align 4
  %115 = insertvalue { i8*, i32 } undef, i8* %113, 0
  %116 = insertvalue { i8*, i32 } %115, i32 %114, 1
  resume { i8*, i32 } %116
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272), i8*) #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"*, float) #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"*, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)*) #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) #1

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_Z19print_array_indexedIfEvPT_i(float*, i32) #5 comdat {
  %3 = alloca float*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store float* %0, float** %3, align 8
  store i32 %1, i32* %4, align 4
  store i32 0, i32* %5, align 4
  br label %6

6:                                                ; preds = %21, %2
  %7 = load i32, i32* %5, align 4
  %8 = load i32, i32* %4, align 4
  %9 = icmp slt i32 %7, %8
  br i1 %9, label %10, label %24

10:                                               ; preds = %6
  %11 = load i32, i32* %5, align 4
  %12 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* @_ZSt4cout, i32 %11)
  %13 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %12, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.17, i64 0, i64 0))
  %14 = load float*, float** %3, align 8
  %15 = load i32, i32* %5, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds float, float* %14, i64 %16
  %18 = load float, float* %17, align 4
  %19 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* %13, float %18)
  %20 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %19, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %21

21:                                               ; preds = %10
  %22 = load i32, i32* %5, align 4
  %23 = add nsw i32 %22, 1
  store i32 %23, i32* %5, align 4
  br label %6

24:                                               ; preds = %6
  %25 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* @_ZSt4cout, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4), i32* dereferenceable(4)) #4 comdat {
  %3 = alloca i32*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  store i32* %0, i32** %4, align 8
  store i32* %1, i32** %5, align 8
  %6 = load i32*, i32** %5, align 8
  %7 = load i32, i32* %6, align 4
  %8 = load i32*, i32** %4, align 8
  %9 = load i32, i32* %8, align 4
  %10 = icmp slt i32 %7, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = load i32*, i32** %5, align 8
  store i32* %12, i32** %3, align 8
  br label %15

13:                                               ; preds = %2
  %14 = load i32*, i32** %4, align 8
  store i32* %14, i32** %3, align 8
  br label %15

15:                                               ; preds = %13, %11
  %16 = load i32*, i32** %3, align 8
  ret i32* %16
}

; Function Attrs: noinline optnone uwtable
define internal i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(float**, i64) #5 {
  %3 = alloca float**, align 8
  %4 = alloca i64, align 8
  store float** %0, float*** %3, align 8
  store i64 %1, i64* %4, align 8
  %5 = load float**, float*** %3, align 8
  %6 = bitcast float** %5 to i8*
  %7 = bitcast i8* %6 to i8**
  %8 = load i64, i64* %4, align 8
  %9 = call i32 @cudaMalloc(i8** %7, i64 %8)
  ret i32 %9
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #1

; Function Attrs: nounwind
declare dso_local i64 @_ZNSt6chrono3_V212system_clock3nowEv() #2

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* dereferenceable(8)) #5 comdat {
  %2 = alloca %"struct.std::chrono::duration.0", align 8
  %3 = alloca %"struct.std::chrono::duration"*, align 8
  store %"struct.std::chrono::duration"* %0, %"struct.std::chrono::duration"** %3, align 8
  %4 = load %"struct.std::chrono::duration"*, %"struct.std::chrono::duration"** %3, align 8
  %5 = call i64 @_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000EEEES2_ILl1ELl1000000EElLb1ELb0EE6__castIlS2_ILl1ELl1000000000EEEES4_RKNS1_IT_T0_EE(%"struct.std::chrono::duration"* dereferenceable(8) %4)
  %6 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %2, i32 0, i32 0
  store i64 %5, i64* %6, align 8
  %7 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %2, i32 0, i32 0
  %8 = load i64, i64* %7, align 8
  ret i64 %8
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* dereferenceable(8), %"struct.std::chrono::time_point"* dereferenceable(8)) #5 comdat {
  %3 = alloca %"struct.std::chrono::duration", align 8
  %4 = alloca %"struct.std::chrono::time_point"*, align 8
  %5 = alloca %"struct.std::chrono::time_point"*, align 8
  %6 = alloca %"struct.std::chrono::duration", align 8
  %7 = alloca %"struct.std::chrono::duration", align 8
  store %"struct.std::chrono::time_point"* %0, %"struct.std::chrono::time_point"** %4, align 8
  store %"struct.std::chrono::time_point"* %1, %"struct.std::chrono::time_point"** %5, align 8
  %8 = load %"struct.std::chrono::time_point"*, %"struct.std::chrono::time_point"** %4, align 8
  %9 = call i64 @_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv(%"struct.std::chrono::time_point"* %8)
  %10 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %6, i32 0, i32 0
  store i64 %9, i64* %10, align 8
  %11 = load %"struct.std::chrono::time_point"*, %"struct.std::chrono::time_point"** %5, align 8
  %12 = call i64 @_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv(%"struct.std::chrono::time_point"* %11)
  %13 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %7, i32 0, i32 0
  store i64 %12, i64* %13, align 8
  %14 = call i64 @_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_(%"struct.std::chrono::duration"* dereferenceable(8) %6, %"struct.std::chrono::duration"* dereferenceable(8) %7)
  %15 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %3, i32 0, i32 0
  store i64 %14, i64* %15, align 8
  %16 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %3, i32 0, i32 0
  %17 = load i64, i64* %16, align 8
  ret i64 %17
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"*) #4 comdat align 2 {
  %2 = alloca %"struct.std::chrono::duration.0"*, align 8
  store %"struct.std::chrono::duration.0"* %0, %"struct.std::chrono::duration.0"** %2, align 8
  %3 = load %"struct.std::chrono::duration.0"*, %"struct.std::chrono::duration.0"** %2, align 8
  %4 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %3, i32 0, i32 0
  %5 = load i64, i64* %4, align 8
  ret i64 %5
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"*, i64) #1

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3*, i32, i32, i32) unnamed_addr #4 comdat align 2 {
  %5 = alloca %struct.dim3*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store %struct.dim3* %0, %struct.dim3** %5, align 8
  store i32 %1, i32* %6, align 4
  store i32 %2, i32* %7, align 4
  store i32 %3, i32* %8, align 4
  %9 = load %struct.dim3*, %struct.dim3** %5, align 8
  %10 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 0
  %11 = load i32, i32* %6, align 4
  store i32 %11, i32* %10, align 4
  %12 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 1
  %13 = load i32, i32* %7, align 4
  store i32 %13, i32* %12, align 4
  %14 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 2
  %15 = load i32, i32* %8, align 4
  store i32 %15, i32* %14, align 4
  ret void
}

declare dso_local i32 @cudaGetLastError() #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* dereferenceable(272), i8 signext) #1

declare dso_local i8* @cudaGetErrorString(i32) #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"*, i32) #1

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i32 @_Z20check_array_equalityIfEiPT_S1_ifb(float*, float*, i32, float, i1 zeroext) #5 comdat {
  %6 = alloca float*, align 8
  %7 = alloca float*, align 8
  %8 = alloca i32, align 4
  %9 = alloca float, align 4
  %10 = alloca i8, align 1
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca float, align 4
  store float* %0, float** %6, align 8
  store float* %1, float** %7, align 8
  store i32 %2, i32* %8, align 4
  store float %3, float* %9, align 4
  %14 = zext i1 %4 to i8
  store i8 %14, i8* %10, align 1
  store i32 0, i32* %11, align 4
  store i32 0, i32* %12, align 4
  br label %15

15:                                               ; preds = %66, %5
  %16 = load i32, i32* %12, align 4
  %17 = load i32, i32* %8, align 4
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %69

19:                                               ; preds = %15
  %20 = load float*, float** %6, align 8
  %21 = load i32, i32* %12, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds float, float* %20, i64 %22
  %24 = load float, float* %23, align 4
  %25 = load float*, float** %7, align 8
  %26 = load i32, i32* %12, align 4
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds float, float* %25, i64 %27
  %29 = load float, float* %28, align 4
  %30 = fsub contract float %24, %29
  %31 = call float @_ZSt3absf(float %30)
  store float %31, float* %13, align 4
  %32 = load float, float* %13, align 4
  %33 = load float, float* %9, align 4
  %34 = fcmp ogt float %32, %33
  br i1 %34, label %35, label %65

35:                                               ; preds = %19
  %36 = load i32, i32* %11, align 4
  %37 = add nsw i32 %36, 1
  store i32 %37, i32* %11, align 4
  %38 = load i8, i8* %10, align 1
  %39 = trunc i8 %38 to i1
  br i1 %39, label %40, label %64

40:                                               ; preds = %35
  %41 = load i32, i32* %11, align 4
  %42 = icmp slt i32 %41, 20
  br i1 %42, label %43, label %64

43:                                               ; preds = %40
  %44 = load i32, i32* %12, align 4
  %45 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* @_ZSt4cout, i32 %44)
  %46 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %45, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.18, i64 0, i64 0))
  %47 = load float*, float** %6, align 8
  %48 = load i32, i32* %12, align 4
  %49 = sext i32 %48 to i64
  %50 = getelementptr inbounds float, float* %47, i64 %49
  %51 = load float, float* %50, align 4
  %52 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* %46, float %51)
  %53 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %52, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.19, i64 0, i64 0))
  %54 = load float*, float** %7, align 8
  %55 = load i32, i32* %12, align 4
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds float, float* %54, i64 %56
  %58 = load float, float* %57, align 4
  %59 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* %53, float %58)
  %60 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %59, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.20, i64 0, i64 0))
  %61 = load float, float* %13, align 4
  %62 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* %60, float %61)
  %63 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %62, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %64

64:                                               ; preds = %43, %40, %35
  br label %65

65:                                               ; preds = %64, %19
  br label %66

66:                                               ; preds = %65
  %67 = load i32, i32* %12, align 4
  %68 = add nsw i32 %67, 1
  store i32 %68, i32* %12, align 4
  br label %15

69:                                               ; preds = %15
  %70 = load i32, i32* %11, align 4
  ret i32 %70
}

; Function Attrs: nounwind
declare dso_local void @_ZNSaIcEC1Ev(%"class.std::allocator"*) unnamed_addr #2

declare dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EPKcRKS3_(%"class.std::__cxx11::basic_string"*, i8*, %"class.std::allocator"* dereferenceable(1)) unnamed_addr #1

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt13random_deviceC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"*, %"class.std::__cxx11::basic_string"* dereferenceable(32)) unnamed_addr #5 comdat align 2 {
  %3 = alloca %"class.std::random_device"*, align 8
  %4 = alloca %"class.std::__cxx11::basic_string"*, align 8
  store %"class.std::random_device"* %0, %"class.std::random_device"** %3, align 8
  store %"class.std::__cxx11::basic_string"* %1, %"class.std::__cxx11::basic_string"** %4, align 8
  %5 = load %"class.std::random_device"*, %"class.std::random_device"** %3, align 8
  %6 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %4, align 8
  call void @_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"* %5, %"class.std::__cxx11::basic_string"* dereferenceable(32) %6)
  ret void
}

; Function Attrs: nounwind
declare dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_string"*) unnamed_addr #2

; Function Attrs: nounwind
declare dso_local void @_ZNSaIcED1Ev(%"class.std::allocator"*) unnamed_addr #2

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i32 @_ZNSt13random_deviceclEv(%"class.std::random_device"*) #5 comdat align 2 {
  %2 = alloca %"class.std::random_device"*, align 8
  store %"class.std::random_device"* %0, %"class.std::random_device"** %2, align 8
  %3 = load %"class.std::random_device"*, %"class.std::random_device"** %2, align 8
  %4 = call i32 @_ZNSt13random_device9_M_getvalEv(%"class.std::random_device"* %3)
  ret i32 %4
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em(%"class.std::mersenne_twister_engine"*, i64) unnamed_addr #5 comdat align 2 {
  %3 = alloca %"class.std::mersenne_twister_engine"*, align 8
  %4 = alloca i64, align 8
  store %"class.std::mersenne_twister_engine"* %0, %"class.std::mersenne_twister_engine"** %3, align 8
  store i64 %1, i64* %4, align 8
  %5 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %3, align 8
  %6 = load i64, i64* %4, align 8
  call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm(%"class.std::mersenne_twister_engine"* %5, i64 %6)
  ret void
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt25uniform_real_distributionIfEC2Eff(%"class.std::uniform_real_distribution"*, float, float) unnamed_addr #5 comdat align 2 {
  %4 = alloca %"class.std::uniform_real_distribution"*, align 8
  %5 = alloca float, align 4
  %6 = alloca float, align 4
  store %"class.std::uniform_real_distribution"* %0, %"class.std::uniform_real_distribution"** %4, align 8
  store float %1, float* %5, align 4
  store float %2, float* %6, align 4
  %7 = load %"class.std::uniform_real_distribution"*, %"class.std::uniform_real_distribution"** %4, align 8
  %8 = getelementptr inbounds %"class.std::uniform_real_distribution", %"class.std::uniform_real_distribution"* %7, i32 0, i32 0
  %9 = load float, float* %5, align 4
  %10 = load float, float* %6, align 4
  call void @_ZNSt25uniform_real_distributionIfE10param_typeC2Eff(%"struct.std::uniform_real_distribution<float>::param_type"* %8, float %9, float %10)
  ret void
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_(%"class.std::uniform_real_distribution"*, %"class.std::mersenne_twister_engine"* dereferenceable(5000)) #5 comdat align 2 {
  %3 = alloca %"class.std::uniform_real_distribution"*, align 8
  %4 = alloca %"class.std::mersenne_twister_engine"*, align 8
  store %"class.std::uniform_real_distribution"* %0, %"class.std::uniform_real_distribution"** %3, align 8
  store %"class.std::mersenne_twister_engine"* %1, %"class.std::mersenne_twister_engine"** %4, align 8
  %5 = load %"class.std::uniform_real_distribution"*, %"class.std::uniform_real_distribution"** %3, align 8
  %6 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %4, align 8
  %7 = getelementptr inbounds %"class.std::uniform_real_distribution", %"class.std::uniform_real_distribution"* %5, i32 0, i32 0
  %8 = call float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_RKNS0_10param_typeE(%"class.std::uniform_real_distribution"* %5, %"class.std::mersenne_twister_engine"* dereferenceable(5000) %6, %"struct.std::uniform_real_distribution<float>::param_type"* dereferenceable(8) %7)
  ret float %8
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt13random_deviceD2Ev(%"class.std::random_device"*) unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca %"class.std::random_device"*, align 8
  store %"class.std::random_device"* %0, %"class.std::random_device"** %2, align 8
  %3 = load %"class.std::random_device"*, %"class.std::random_device"** %2, align 8
  invoke void @_ZNSt13random_device7_M_finiEv(%"class.std::random_device"* %3)
          to label %4 unwind label %5

4:                                                ; preds = %1
  ret void

5:                                                ; preds = %1
  %6 = landingpad { i8*, i32 }
          catch i8* null
  %7 = extractvalue { i8*, i32 } %6, 0
  call void @__clang_call_terminate(i8* %7) #12
  unreachable
}

declare dso_local void @_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"*, %"class.std::__cxx11::basic_string"* dereferenceable(32)) #1

declare dso_local i32 @_ZNSt13random_device9_M_getvalEv(%"class.std::random_device"*) #1

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm(%"class.std::mersenne_twister_engine"*, i64) #5 comdat align 2 {
  %3 = alloca %"class.std::mersenne_twister_engine"*, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  store %"class.std::mersenne_twister_engine"* %0, %"class.std::mersenne_twister_engine"** %3, align 8
  store i64 %1, i64* %4, align 8
  %7 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %3, align 8
  %8 = load i64, i64* %4, align 8
  %9 = call i64 @_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_(i64 %8)
  %10 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %7, i32 0, i32 0
  %11 = getelementptr inbounds [624 x i64], [624 x i64]* %10, i64 0, i64 0
  store i64 %9, i64* %11, align 8
  store i64 1, i64* %5, align 8
  br label %12

12:                                               ; preds = %36, %2
  %13 = load i64, i64* %5, align 8
  %14 = icmp ult i64 %13, 624
  br i1 %14, label %15, label %39

15:                                               ; preds = %12
  %16 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %7, i32 0, i32 0
  %17 = load i64, i64* %5, align 8
  %18 = sub i64 %17, 1
  %19 = getelementptr inbounds [624 x i64], [624 x i64]* %16, i64 0, i64 %18
  %20 = load i64, i64* %19, align 8
  store i64 %20, i64* %6, align 8
  %21 = load i64, i64* %6, align 8
  %22 = lshr i64 %21, 30
  %23 = load i64, i64* %6, align 8
  %24 = xor i64 %23, %22
  store i64 %24, i64* %6, align 8
  %25 = load i64, i64* %6, align 8
  %26 = mul i64 %25, 1812433253
  store i64 %26, i64* %6, align 8
  %27 = load i64, i64* %5, align 8
  %28 = call i64 @_ZNSt8__detail5__modImLm624ELm1ELm0EEET_S1_(i64 %27)
  %29 = load i64, i64* %6, align 8
  %30 = add i64 %29, %28
  store i64 %30, i64* %6, align 8
  %31 = load i64, i64* %6, align 8
  %32 = call i64 @_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_(i64 %31)
  %33 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %7, i32 0, i32 0
  %34 = load i64, i64* %5, align 8
  %35 = getelementptr inbounds [624 x i64], [624 x i64]* %33, i64 0, i64 %34
  store i64 %32, i64* %35, align 8
  br label %36

36:                                               ; preds = %15
  %37 = load i64, i64* %5, align 8
  %38 = add i64 %37, 1
  store i64 %38, i64* %5, align 8
  br label %12

39:                                               ; preds = %12
  %40 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %7, i32 0, i32 1
  store i64 624, i64* %40, align 8
  ret void
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_(i64) #5 comdat {
  %2 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  %3 = load i64, i64* %2, align 8
  %4 = call i64 @_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm(i64 %3)
  ret i64 %4
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail5__modImLm624ELm1ELm0EEET_S1_(i64) #5 comdat {
  %2 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  %3 = load i64, i64* %2, align 8
  %4 = call i64 @_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm(i64 %3)
  ret i64 %4
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm(i64) #4 comdat align 2 {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  %4 = load i64, i64* %2, align 8
  %5 = mul i64 1, %4
  %6 = add i64 %5, 0
  store i64 %6, i64* %3, align 8
  %7 = load i64, i64* %3, align 8
  %8 = urem i64 %7, 4294967296
  store i64 %8, i64* %3, align 8
  %9 = load i64, i64* %3, align 8
  ret i64 %9
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm(i64) #4 comdat align 2 {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  %4 = load i64, i64* %2, align 8
  %5 = mul i64 1, %4
  %6 = add i64 %5, 0
  store i64 %6, i64* %3, align 8
  %7 = load i64, i64* %3, align 8
  %8 = urem i64 %7, 624
  store i64 %8, i64* %3, align 8
  %9 = load i64, i64* %3, align 8
  ret i64 %9
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt25uniform_real_distributionIfE10param_typeC2Eff(%"struct.std::uniform_real_distribution<float>::param_type"*, float, float) unnamed_addr #4 comdat align 2 {
  %4 = alloca %"struct.std::uniform_real_distribution<float>::param_type"*, align 8
  %5 = alloca float, align 4
  %6 = alloca float, align 4
  store %"struct.std::uniform_real_distribution<float>::param_type"* %0, %"struct.std::uniform_real_distribution<float>::param_type"** %4, align 8
  store float %1, float* %5, align 4
  store float %2, float* %6, align 4
  %7 = load %"struct.std::uniform_real_distribution<float>::param_type"*, %"struct.std::uniform_real_distribution<float>::param_type"** %4, align 8
  %8 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %7, i32 0, i32 0
  %9 = load float, float* %5, align 4
  store float %9, float* %8, align 4
  %10 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %7, i32 0, i32 1
  %11 = load float, float* %6, align 4
  store float %11, float* %10, align 4
  ret void
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_RKNS0_10param_typeE(%"class.std::uniform_real_distribution"*, %"class.std::mersenne_twister_engine"* dereferenceable(5000), %"struct.std::uniform_real_distribution<float>::param_type"* dereferenceable(8)) #5 comdat align 2 {
  %4 = alloca %"class.std::uniform_real_distribution"*, align 8
  %5 = alloca %"class.std::mersenne_twister_engine"*, align 8
  %6 = alloca %"struct.std::uniform_real_distribution<float>::param_type"*, align 8
  %7 = alloca %"struct.std::__detail::_Adaptor", align 8
  store %"class.std::uniform_real_distribution"* %0, %"class.std::uniform_real_distribution"** %4, align 8
  store %"class.std::mersenne_twister_engine"* %1, %"class.std::mersenne_twister_engine"** %5, align 8
  store %"struct.std::uniform_real_distribution<float>::param_type"* %2, %"struct.std::uniform_real_distribution<float>::param_type"** %6, align 8
  %8 = load %"class.std::uniform_real_distribution"*, %"class.std::uniform_real_distribution"** %4, align 8
  %9 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %5, align 8
  call void @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEC2ERS2_(%"struct.std::__detail::_Adaptor"* %7, %"class.std::mersenne_twister_engine"* dereferenceable(5000) %9)
  %10 = call float @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEclEv(%"struct.std::__detail::_Adaptor"* %7)
  %11 = load %"struct.std::uniform_real_distribution<float>::param_type"*, %"struct.std::uniform_real_distribution<float>::param_type"** %6, align 8
  %12 = call float @_ZNKSt25uniform_real_distributionIfE10param_type1bEv(%"struct.std::uniform_real_distribution<float>::param_type"* %11)
  %13 = load %"struct.std::uniform_real_distribution<float>::param_type"*, %"struct.std::uniform_real_distribution<float>::param_type"** %6, align 8
  %14 = call float @_ZNKSt25uniform_real_distributionIfE10param_type1aEv(%"struct.std::uniform_real_distribution<float>::param_type"* %13)
  %15 = fsub contract float %12, %14
  %16 = fmul contract float %10, %15
  %17 = load %"struct.std::uniform_real_distribution<float>::param_type"*, %"struct.std::uniform_real_distribution<float>::param_type"** %6, align 8
  %18 = call float @_ZNKSt25uniform_real_distributionIfE10param_type1aEv(%"struct.std::uniform_real_distribution<float>::param_type"* %17)
  %19 = fadd contract float %16, %18
  ret float %19
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEC2ERS2_(%"struct.std::__detail::_Adaptor"*, %"class.std::mersenne_twister_engine"* dereferenceable(5000)) unnamed_addr #4 comdat align 2 {
  %3 = alloca %"struct.std::__detail::_Adaptor"*, align 8
  %4 = alloca %"class.std::mersenne_twister_engine"*, align 8
  store %"struct.std::__detail::_Adaptor"* %0, %"struct.std::__detail::_Adaptor"** %3, align 8
  store %"class.std::mersenne_twister_engine"* %1, %"class.std::mersenne_twister_engine"** %4, align 8
  %5 = load %"struct.std::__detail::_Adaptor"*, %"struct.std::__detail::_Adaptor"** %3, align 8
  %6 = getelementptr inbounds %"struct.std::__detail::_Adaptor", %"struct.std::__detail::_Adaptor"* %5, i32 0, i32 0
  %7 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %4, align 8
  store %"class.std::mersenne_twister_engine"* %7, %"class.std::mersenne_twister_engine"** %6, align 8
  ret void
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local float @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEclEv(%"struct.std::__detail::_Adaptor"*) #5 comdat align 2 {
  %2 = alloca %"struct.std::__detail::_Adaptor"*, align 8
  store %"struct.std::__detail::_Adaptor"* %0, %"struct.std::__detail::_Adaptor"** %2, align 8
  %3 = load %"struct.std::__detail::_Adaptor"*, %"struct.std::__detail::_Adaptor"** %2, align 8
  %4 = getelementptr inbounds %"struct.std::__detail::_Adaptor", %"struct.std::__detail::_Adaptor"* %3, i32 0, i32 0
  %5 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %4, align 8
  %6 = call float @_ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_(%"class.std::mersenne_twister_engine"* dereferenceable(5000) %5)
  ret float %6
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local float @_ZNKSt25uniform_real_distributionIfE10param_type1bEv(%"struct.std::uniform_real_distribution<float>::param_type"*) #4 comdat align 2 {
  %2 = alloca %"struct.std::uniform_real_distribution<float>::param_type"*, align 8
  store %"struct.std::uniform_real_distribution<float>::param_type"* %0, %"struct.std::uniform_real_distribution<float>::param_type"** %2, align 8
  %3 = load %"struct.std::uniform_real_distribution<float>::param_type"*, %"struct.std::uniform_real_distribution<float>::param_type"** %2, align 8
  %4 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %3, i32 0, i32 1
  %5 = load float, float* %4, align 4
  ret float %5
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local float @_ZNKSt25uniform_real_distributionIfE10param_type1aEv(%"struct.std::uniform_real_distribution<float>::param_type"*) #4 comdat align 2 {
  %2 = alloca %"struct.std::uniform_real_distribution<float>::param_type"*, align 8
  store %"struct.std::uniform_real_distribution<float>::param_type"* %0, %"struct.std::uniform_real_distribution<float>::param_type"** %2, align 8
  %3 = load %"struct.std::uniform_real_distribution<float>::param_type"*, %"struct.std::uniform_real_distribution<float>::param_type"** %2, align 8
  %4 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %3, i32 0, i32 0
  %5 = load float, float* %4, align 4
  ret float %5
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local float @_ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_(%"class.std::mersenne_twister_engine"* dereferenceable(5000)) #5 comdat {
  %2 = alloca %"class.std::mersenne_twister_engine"*, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca x86_fp80, align 16
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca float, align 4
  %12 = alloca float, align 4
  %13 = alloca float, align 4
  %14 = alloca i64, align 8
  store %"class.std::mersenne_twister_engine"* %0, %"class.std::mersenne_twister_engine"** %2, align 8
  store i64 24, i64* %4, align 8
  store i64 24, i64* %5, align 8
  %15 = call dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* dereferenceable(8) %4, i64* dereferenceable(8) %5)
  %16 = load i64, i64* %15, align 8
  store i64 %16, i64* %3, align 8
  %17 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8
  %18 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv()
  %19 = uitofp i64 %18 to x86_fp80
  %20 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8
  %21 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv()
  %22 = uitofp i64 %21 to x86_fp80
  %23 = fsub contract x86_fp80 %19, %22
  %24 = fadd contract x86_fp80 %23, 0xK3FFF8000000000000000
  store x86_fp80 %24, x86_fp80* %6, align 16
  %25 = load x86_fp80, x86_fp80* %6, align 16
  %26 = call x86_fp80 @_ZSt3loge(x86_fp80 %25)
  %27 = call x86_fp80 @_ZSt3loge(x86_fp80 0xK40008000000000000000)
  %28 = fdiv x86_fp80 %26, %27
  %29 = fptoui x86_fp80 %28 to i64
  store i64 %29, i64* %7, align 8
  store i64 1, i64* %9, align 8
  %30 = load i64, i64* %3, align 8
  %31 = load i64, i64* %7, align 8
  %32 = add i64 %30, %31
  %33 = sub i64 %32, 1
  %34 = load i64, i64* %7, align 8
  %35 = udiv i64 %33, %34
  store i64 %35, i64* %10, align 8
  %36 = call dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* dereferenceable(8) %9, i64* dereferenceable(8) %10)
  %37 = load i64, i64* %36, align 8
  store i64 %37, i64* %8, align 8
  store float 0.000000e+00, float* %12, align 4
  store float 1.000000e+00, float* %13, align 4
  %38 = load i64, i64* %8, align 8
  store i64 %38, i64* %14, align 8
  br label %39

39:                                               ; preds = %58, %1
  %40 = load i64, i64* %14, align 8
  %41 = icmp ne i64 %40, 0
  br i1 %41, label %42, label %61

42:                                               ; preds = %39
  %43 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8
  %44 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"* %43)
  %45 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8
  %46 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv()
  %47 = sub i64 %44, %46
  %48 = uitofp i64 %47 to float
  %49 = load float, float* %13, align 4
  %50 = fmul contract float %48, %49
  %51 = load float, float* %12, align 4
  %52 = fadd contract float %51, %50
  store float %52, float* %12, align 4
  %53 = load x86_fp80, x86_fp80* %6, align 16
  %54 = load float, float* %13, align 4
  %55 = fpext float %54 to x86_fp80
  %56 = fmul contract x86_fp80 %55, %53
  %57 = fptrunc x86_fp80 %56 to float
  store float %57, float* %13, align 4
  br label %58

58:                                               ; preds = %42
  %59 = load i64, i64* %14, align 8
  %60 = add i64 %59, -1
  store i64 %60, i64* %14, align 8
  br label %39

61:                                               ; preds = %39
  %62 = load float, float* %12, align 4
  %63 = load float, float* %13, align 4
  %64 = fdiv float %62, %63
  store float %64, float* %11, align 4
  %65 = load float, float* %11, align 4
  %66 = fcmp oge float %65, 1.000000e+00
  br i1 %66, label %67, label %69

67:                                               ; preds = %61
  %68 = call float @_ZSt9nextafterff(float 1.000000e+00, float 0.000000e+00)
  store float %68, float* %11, align 4
  br label %69

69:                                               ; preds = %67, %61
  %70 = load float, float* %11, align 4
  ret float %70
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* dereferenceable(8), i64* dereferenceable(8)) #4 comdat {
  %3 = alloca i64*, align 8
  %4 = alloca i64*, align 8
  %5 = alloca i64*, align 8
  store i64* %0, i64** %4, align 8
  store i64* %1, i64** %5, align 8
  %6 = load i64*, i64** %5, align 8
  %7 = load i64, i64* %6, align 8
  %8 = load i64*, i64** %4, align 8
  %9 = load i64, i64* %8, align 8
  %10 = icmp ult i64 %7, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = load i64*, i64** %5, align 8
  store i64* %12, i64** %3, align 8
  br label %15

13:                                               ; preds = %2
  %14 = load i64*, i64** %4, align 8
  store i64* %14, i64** %3, align 8
  br label %15

15:                                               ; preds = %13, %11
  %16 = load i64*, i64** %3, align 8
  ret i64* %16
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv() #4 comdat align 2 {
  ret i64 4294967295
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv() #4 comdat align 2 {
  ret i64 0
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local x86_fp80 @_ZSt3loge(x86_fp80) #4 comdat {
  %2 = alloca x86_fp80, align 16
  store x86_fp80 %0, x86_fp80* %2, align 16
  %3 = load x86_fp80, x86_fp80* %2, align 16
  %4 = call x86_fp80 @logl(x86_fp80 %3) #3
  ret x86_fp80 %4
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* dereferenceable(8), i64* dereferenceable(8)) #4 comdat {
  %3 = alloca i64*, align 8
  %4 = alloca i64*, align 8
  %5 = alloca i64*, align 8
  store i64* %0, i64** %4, align 8
  store i64* %1, i64** %5, align 8
  %6 = load i64*, i64** %4, align 8
  %7 = load i64, i64* %6, align 8
  %8 = load i64*, i64** %5, align 8
  %9 = load i64, i64* %8, align 8
  %10 = icmp ult i64 %7, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %2
  %12 = load i64*, i64** %5, align 8
  store i64* %12, i64** %3, align 8
  br label %15

13:                                               ; preds = %2
  %14 = load i64*, i64** %4, align 8
  store i64* %14, i64** %3, align 8
  br label %15

15:                                               ; preds = %13, %11
  %16 = load i64*, i64** %3, align 8
  ret i64* %16
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"*) #5 comdat align 2 {
  %2 = alloca %"class.std::mersenne_twister_engine"*, align 8
  %3 = alloca i64, align 8
  store %"class.std::mersenne_twister_engine"* %0, %"class.std::mersenne_twister_engine"** %2, align 8
  %4 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8
  %5 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %4, i32 0, i32 1
  %6 = load i64, i64* %5, align 8
  %7 = icmp uge i64 %6, 624
  br i1 %7, label %8, label %9

8:                                                ; preds = %1
  call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv(%"class.std::mersenne_twister_engine"* %4)
  br label %9

9:                                                ; preds = %8, %1
  %10 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %4, i32 0, i32 0
  %11 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %4, i32 0, i32 1
  %12 = load i64, i64* %11, align 8
  %13 = add i64 %12, 1
  store i64 %13, i64* %11, align 8
  %14 = getelementptr inbounds [624 x i64], [624 x i64]* %10, i64 0, i64 %12
  %15 = load i64, i64* %14, align 8
  store i64 %15, i64* %3, align 8
  %16 = load i64, i64* %3, align 8
  %17 = lshr i64 %16, 11
  %18 = and i64 %17, 4294967295
  %19 = load i64, i64* %3, align 8
  %20 = xor i64 %19, %18
  store i64 %20, i64* %3, align 8
  %21 = load i64, i64* %3, align 8
  %22 = shl i64 %21, 7
  %23 = and i64 %22, 2636928640
  %24 = load i64, i64* %3, align 8
  %25 = xor i64 %24, %23
  store i64 %25, i64* %3, align 8
  %26 = load i64, i64* %3, align 8
  %27 = shl i64 %26, 15
  %28 = and i64 %27, 4022730752
  %29 = load i64, i64* %3, align 8
  %30 = xor i64 %29, %28
  store i64 %30, i64* %3, align 8
  %31 = load i64, i64* %3, align 8
  %32 = lshr i64 %31, 18
  %33 = load i64, i64* %3, align 8
  %34 = xor i64 %33, %32
  store i64 %34, i64* %3, align 8
  %35 = load i64, i64* %3, align 8
  ret i64 %35
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local float @_ZSt9nextafterff(float, float) #4 comdat {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, float* %3, align 4
  store float %1, float* %4, align 4
  %5 = load float, float* %3, align 4
  %6 = load float, float* %4, align 4
  %7 = call float @nextafterf(float %5, float %6) #3
  ret float %7
}

; Function Attrs: nounwind
declare dso_local x86_fp80 @logl(x86_fp80) #2

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv(%"class.std::mersenne_twister_engine"*) #4 comdat align 2 {
  %2 = alloca %"class.std::mersenne_twister_engine"*, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store %"class.std::mersenne_twister_engine"* %0, %"class.std::mersenne_twister_engine"** %2, align 8
  %10 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8
  store i64 -2147483648, i64* %3, align 8
  store i64 2147483647, i64* %4, align 8
  store i64 0, i64* %5, align 8
  br label %11

11:                                               ; preds = %44, %1
  %12 = load i64, i64* %5, align 8
  %13 = icmp ult i64 %12, 227
  br i1 %13, label %14, label %47

14:                                               ; preds = %11
  %15 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %16 = load i64, i64* %5, align 8
  %17 = getelementptr inbounds [624 x i64], [624 x i64]* %15, i64 0, i64 %16
  %18 = load i64, i64* %17, align 8
  %19 = and i64 %18, -2147483648
  %20 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %21 = load i64, i64* %5, align 8
  %22 = add i64 %21, 1
  %23 = getelementptr inbounds [624 x i64], [624 x i64]* %20, i64 0, i64 %22
  %24 = load i64, i64* %23, align 8
  %25 = and i64 %24, 2147483647
  %26 = or i64 %19, %25
  store i64 %26, i64* %6, align 8
  %27 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %28 = load i64, i64* %5, align 8
  %29 = add i64 %28, 397
  %30 = getelementptr inbounds [624 x i64], [624 x i64]* %27, i64 0, i64 %29
  %31 = load i64, i64* %30, align 8
  %32 = load i64, i64* %6, align 8
  %33 = lshr i64 %32, 1
  %34 = xor i64 %31, %33
  %35 = load i64, i64* %6, align 8
  %36 = and i64 %35, 1
  %37 = icmp ne i64 %36, 0
  %38 = zext i1 %37 to i64
  %39 = select i1 %37, i64 2567483615, i64 0
  %40 = xor i64 %34, %39
  %41 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %42 = load i64, i64* %5, align 8
  %43 = getelementptr inbounds [624 x i64], [624 x i64]* %41, i64 0, i64 %42
  store i64 %40, i64* %43, align 8
  br label %44

44:                                               ; preds = %14
  %45 = load i64, i64* %5, align 8
  %46 = add i64 %45, 1
  store i64 %46, i64* %5, align 8
  br label %11

47:                                               ; preds = %11
  store i64 227, i64* %7, align 8
  br label %48

48:                                               ; preds = %81, %47
  %49 = load i64, i64* %7, align 8
  %50 = icmp ult i64 %49, 623
  br i1 %50, label %51, label %84

51:                                               ; preds = %48
  %52 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %53 = load i64, i64* %7, align 8
  %54 = getelementptr inbounds [624 x i64], [624 x i64]* %52, i64 0, i64 %53
  %55 = load i64, i64* %54, align 8
  %56 = and i64 %55, -2147483648
  %57 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %58 = load i64, i64* %7, align 8
  %59 = add i64 %58, 1
  %60 = getelementptr inbounds [624 x i64], [624 x i64]* %57, i64 0, i64 %59
  %61 = load i64, i64* %60, align 8
  %62 = and i64 %61, 2147483647
  %63 = or i64 %56, %62
  store i64 %63, i64* %8, align 8
  %64 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %65 = load i64, i64* %7, align 8
  %66 = add i64 %65, -227
  %67 = getelementptr inbounds [624 x i64], [624 x i64]* %64, i64 0, i64 %66
  %68 = load i64, i64* %67, align 8
  %69 = load i64, i64* %8, align 8
  %70 = lshr i64 %69, 1
  %71 = xor i64 %68, %70
  %72 = load i64, i64* %8, align 8
  %73 = and i64 %72, 1
  %74 = icmp ne i64 %73, 0
  %75 = zext i1 %74 to i64
  %76 = select i1 %74, i64 2567483615, i64 0
  %77 = xor i64 %71, %76
  %78 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %79 = load i64, i64* %7, align 8
  %80 = getelementptr inbounds [624 x i64], [624 x i64]* %78, i64 0, i64 %79
  store i64 %77, i64* %80, align 8
  br label %81

81:                                               ; preds = %51
  %82 = load i64, i64* %7, align 8
  %83 = add i64 %82, 1
  store i64 %83, i64* %7, align 8
  br label %48

84:                                               ; preds = %48
  %85 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %86 = getelementptr inbounds [624 x i64], [624 x i64]* %85, i64 0, i64 623
  %87 = load i64, i64* %86, align 8
  %88 = and i64 %87, -2147483648
  %89 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %90 = getelementptr inbounds [624 x i64], [624 x i64]* %89, i64 0, i64 0
  %91 = load i64, i64* %90, align 8
  %92 = and i64 %91, 2147483647
  %93 = or i64 %88, %92
  store i64 %93, i64* %9, align 8
  %94 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %95 = getelementptr inbounds [624 x i64], [624 x i64]* %94, i64 0, i64 396
  %96 = load i64, i64* %95, align 8
  %97 = load i64, i64* %9, align 8
  %98 = lshr i64 %97, 1
  %99 = xor i64 %96, %98
  %100 = load i64, i64* %9, align 8
  %101 = and i64 %100, 1
  %102 = icmp ne i64 %101, 0
  %103 = zext i1 %102 to i64
  %104 = select i1 %102, i64 2567483615, i64 0
  %105 = xor i64 %99, %104
  %106 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 0
  %107 = getelementptr inbounds [624 x i64], [624 x i64]* %106, i64 0, i64 623
  store i64 %105, i64* %107, align 8
  %108 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %10, i32 0, i32 1
  store i64 0, i64* %108, align 8
  ret void
}

; Function Attrs: nounwind
declare dso_local float @nextafterf(float, float) #2

declare dso_local void @_ZNSt13random_device7_M_finiEv(%"class.std::random_device"*) #1

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) #9 comdat {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #3
  call void @_ZSt9terminatev() #12
  unreachable
}

declare dso_local i8* @__cxa_begin_catch(i8*)

declare dso_local void @_ZSt9terminatev()

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000EEEES2_ILl1ELl1000000EElLb1ELb0EE6__castIlS2_ILl1ELl1000000000EEEES4_RKNS1_IT_T0_EE(%"struct.std::chrono::duration"* dereferenceable(8)) #5 comdat align 2 {
  %2 = alloca %"struct.std::chrono::duration.0", align 8
  %3 = alloca %"struct.std::chrono::duration"*, align 8
  %4 = alloca i64, align 8
  store %"struct.std::chrono::duration"* %0, %"struct.std::chrono::duration"** %3, align 8
  %5 = load %"struct.std::chrono::duration"*, %"struct.std::chrono::duration"** %3, align 8
  %6 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"* %5)
  %7 = sdiv i64 %6, 1000000
  store i64 %7, i64* %4, align 8
  call void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000EEEC2IlvEERKT_(%"struct.std::chrono::duration.0"* %2, i64* dereferenceable(8) %4)
  %8 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %2, i32 0, i32 0
  %9 = load i64, i64* %8, align 8
  ret i64 %9
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"*) #4 comdat align 2 {
  %2 = alloca %"struct.std::chrono::duration"*, align 8
  store %"struct.std::chrono::duration"* %0, %"struct.std::chrono::duration"** %2, align 8
  %3 = load %"struct.std::chrono::duration"*, %"struct.std::chrono::duration"** %2, align 8
  %4 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %3, i32 0, i32 0
  %5 = load i64, i64* %4, align 8
  ret i64 %5
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000EEEC2IlvEERKT_(%"struct.std::chrono::duration.0"*, i64* dereferenceable(8)) unnamed_addr #4 comdat align 2 {
  %3 = alloca %"struct.std::chrono::duration.0"*, align 8
  %4 = alloca i64*, align 8
  store %"struct.std::chrono::duration.0"* %0, %"struct.std::chrono::duration.0"** %3, align 8
  store i64* %1, i64** %4, align 8
  %5 = load %"struct.std::chrono::duration.0"*, %"struct.std::chrono::duration.0"** %3, align 8
  %6 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %5, i32 0, i32 0
  %7 = load i64*, i64** %4, align 8
  %8 = load i64, i64* %7, align 8
  store i64 %8, i64* %6, align 8
  ret void
}

; Function Attrs: noinline optnone uwtable
define linkonce_odr dso_local i64 @_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_(%"struct.std::chrono::duration"* dereferenceable(8), %"struct.std::chrono::duration"* dereferenceable(8)) #5 comdat {
  %3 = alloca %"struct.std::chrono::duration", align 8
  %4 = alloca %"struct.std::chrono::duration"*, align 8
  %5 = alloca %"struct.std::chrono::duration"*, align 8
  %6 = alloca i64, align 8
  %7 = alloca %"struct.std::chrono::duration", align 8
  %8 = alloca %"struct.std::chrono::duration", align 8
  store %"struct.std::chrono::duration"* %0, %"struct.std::chrono::duration"** %4, align 8
  store %"struct.std::chrono::duration"* %1, %"struct.std::chrono::duration"** %5, align 8
  %9 = load %"struct.std::chrono::duration"*, %"struct.std::chrono::duration"** %4, align 8
  %10 = bitcast %"struct.std::chrono::duration"* %7 to i8*
  %11 = bitcast %"struct.std::chrono::duration"* %9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %10, i8* align 8 %11, i64 8, i1 false)
  %12 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"* %7)
  %13 = load %"struct.std::chrono::duration"*, %"struct.std::chrono::duration"** %5, align 8
  %14 = bitcast %"struct.std::chrono::duration"* %8 to i8*
  %15 = bitcast %"struct.std::chrono::duration"* %13 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %14, i8* align 8 %15, i64 8, i1 false)
  %16 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"* %8)
  %17 = sub nsw i64 %12, %16
  store i64 %17, i64* %6, align 8
  call void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC2IlvEERKT_(%"struct.std::chrono::duration"* %3, i64* dereferenceable(8) %6)
  %18 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %3, i32 0, i32 0
  %19 = load i64, i64* %18, align 8
  ret i64 %19
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i64 @_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv(%"struct.std::chrono::time_point"*) #4 comdat align 2 {
  %2 = alloca %"struct.std::chrono::duration", align 8
  %3 = alloca %"struct.std::chrono::time_point"*, align 8
  store %"struct.std::chrono::time_point"* %0, %"struct.std::chrono::time_point"** %3, align 8
  %4 = load %"struct.std::chrono::time_point"*, %"struct.std::chrono::time_point"** %3, align 8
  %5 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %4, i32 0, i32 0
  %6 = bitcast %"struct.std::chrono::duration"* %2 to i8*
  %7 = bitcast %"struct.std::chrono::duration"* %5 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %6, i8* align 8 %7, i64 8, i1 false)
  %8 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %2, i32 0, i32 0
  %9 = load i64, i64* %8, align 8
  ret i64 %9
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC2IlvEERKT_(%"struct.std::chrono::duration"*, i64* dereferenceable(8)) unnamed_addr #4 comdat align 2 {
  %3 = alloca %"struct.std::chrono::duration"*, align 8
  %4 = alloca i64*, align 8
  store %"struct.std::chrono::duration"* %0, %"struct.std::chrono::duration"** %3, align 8
  store i64* %1, i64** %4, align 8
  %5 = load %"struct.std::chrono::duration"*, %"struct.std::chrono::duration"** %3, align 8
  %6 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %5, i32 0, i32 0
  %7 = load i64*, i64** %4, align 8
  %8 = load i64, i64* %7, align 8
  store i64 %8, i64* %6, align 8
  ret void
}

declare dso_local i32 @cudaMalloc(i8**, i64) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local float @_ZSt3absf(float) #4 comdat {
  %2 = alloca float, align 4
  store float %0, float* %2, align 4
  %3 = load float, float* %2, align 4
  %4 = call float @llvm.fabs.f32(float %3)
  ret float %4
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #10

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_axpy.cu() #0 section ".text.startup" {
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { noinline noreturn nounwind }
attributes #10 = { nounwind readnone speculatable }
attributes #11 = { nounwind readonly }
attributes #12 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
