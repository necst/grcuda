; ModuleID = 'dot_product.cu'
source_filename = "dot_product.cu"
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
%"class.__gnu_cxx::new_allocator" = type { i8 }
%"struct.std::__detail::_Adaptor" = type { %"class.std::mersenne_twister_engine"* }

$_Z20create_sample_vectorPfibb = comdat any

$_Z19print_array_indexedIfEvPT_i = comdat any

$_ZSt3minIiERKT_S2_S2_ = comdat any

$_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE = comdat any

$_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE = comdat any

$_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv = comdat any

$_ZN4dim3C2Ejjj = comdat any

$_Z14check_equalityIfEbT_S0_fb = comdat any

$_ZNSt13random_deviceC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZNSt13random_deviceclEv = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em = comdat any

$_ZNSt25uniform_real_distributionIfEC2Eff = comdat any

$_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_ = comdat any

$_ZNSt13random_deviceD2Ev = comdat any

$_ZN9__gnu_cxx13new_allocatorIcEC2Ev = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_ = comdat any

$_ZNSt11char_traitsIcE6lengthEPKc = comdat any

$_ZNSt14pointer_traitsIPcE10pointer_toERc = comdat any

$_ZSt9addressofIcEPT_RS0_ = comdat any

$_ZSt11__addressofIcEPT_RS0_ = comdat any

$_ZN9__gnu_cxx13new_allocatorIcEC2ERKS1_ = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPKcEEvT_S8_St12__false_type = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag = comdat any

$_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_ = comdat any

$_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_ = comdat any

$__clang_call_terminate = comdat any

$_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag = comdat any

$_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_ = comdat any

$_ZNSt11char_traitsIcE6assignERcRKc = comdat any

$_ZNSt11char_traitsIcE4copyEPcPKcm = comdat any

$_ZNSt14pointer_traitsIPKcE10pointer_toERS0_ = comdat any

$_ZSt9addressofIKcEPT_RS1_ = comdat any

$_ZSt11__addressofIKcEPT_RS1_ = comdat any

$_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm = comdat any

$_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm = comdat any

$_ZN9__gnu_cxx13new_allocatorIcED2Ev = comdat any

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

$_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000EEEES2_ILl1ELl1000000EElLb1ELb0EE6__castIlS2_ILl1ELl1000000000EEEES4_RKNS1_IT_T0_EE = comdat any

$_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv = comdat any

$_ZNSt6chrono8durationIlSt5ratioILl1ELl1000EEEC2IlvEERKT_ = comdat any

$_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_ = comdat any

$_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv = comdat any

$_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC2IlvEERKT_ = comdat any

$_ZStorSt12_Ios_IostateS_ = comdat any

$_ZSt13__check_facetISt5ctypeIcEERKT_PS3_ = comdat any

$_ZNKSt5ctypeIcE5widenEc = comdat any

$_ZSt3absf = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZZ4mainE12long_options = internal global [3 x %struct.option] [%struct.option { i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i32 0, i32 0), i32 0, i32* null, i32 104 }, %struct.option { i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.1, i32 0, i32 0), i32 1, i32* null, i32 115 }, %struct.option zeroinitializer], align 16
@.str = private unnamed_addr constant [15 x i8] c"human_readable\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"size\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"hs:c\00", align 1
@optarg = external dso_local local_unnamed_addr global i8*, align 8
@_ZSt4cout = external dso_local global %"class.std::basic_ostream", align 8
@.str.3 = private unnamed_addr constant [5 x i8] c"\0AX: \00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"Y: \00", align 1
@.str.5 = private unnamed_addr constant [48 x i8] c"----------------------------------------------\0A\00", align 1
@.str.6 = private unnamed_addr constant [13 x i8] c"CPU Result: \00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c"Duration: \00", align 1
@.str.8 = private unnamed_addr constant [4 x i8] c" ms\00", align 1
@_ZSt4cerr = external dso_local global %"class.std::basic_ostream", align 8
@.str.9 = private unnamed_addr constant [15 x i8] c"Cuda failure: \00", align 1
@.str.10 = private unnamed_addr constant [6 x i8] c" at: \00", align 1
@.str.11 = private unnamed_addr constant [15 x i8] c"dot_product.cu\00", align 1
@.str.12 = private unnamed_addr constant [23 x i8] c"GPU Result - Checked: \00", align 1
@.str.13 = private unnamed_addr constant [10 x i8] c"Correct: \00", align 1
@.str.14 = private unnamed_addr constant [25 x i8] c"GPU Result - Unchecked: \00", align 1
@.str.15 = private unnamed_addr constant [8 x i8] c"default\00", align 1
@.str.16 = private unnamed_addr constant [42 x i8] c"basic_string::_M_construct null not valid\00", align 1
@.str.17 = private unnamed_addr constant [3 x i8] c") \00", align 1
@.str.18 = private unnamed_addr constant [4 x i8] c"x: \00", align 1
@.str.19 = private unnamed_addr constant [6 x i8] c", y: \00", align 1
@.str.20 = private unnamed_addr constant [9 x i8] c", diff: \00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_dot_product.cu, i8* null }]

; Function Attrs: uwtable
define internal fastcc void @__cxx_global_var_init() unnamed_addr #0 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #18
  ret void
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #3

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @_Z15dot_product_cpuPfS_iS_(float* nocapture readonly, float* nocapture readonly, i32, float* nocapture) local_unnamed_addr #4 {
  %5 = icmp sgt i32 %2, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = zext i32 %2 to i64
  br label %9

8:                                                ; preds = %9, %4
  ret void

9:                                                ; preds = %9, %6
  %10 = phi i64 [ 0, %6 ], [ %18, %9 ]
  %11 = getelementptr inbounds float, float* %0, i64 %10
  %12 = load float, float* %11, align 4, !tbaa !3
  %13 = getelementptr inbounds float, float* %1, i64 %10
  %14 = load float, float* %13, align 4, !tbaa !3
  %15 = fmul contract float %12, %14
  %16 = load float, float* %3, align 4, !tbaa !3
  %17 = fadd contract float %16, %15
  store float %17, float* %3, align 4, !tbaa !3
  %18 = add nuw nsw i64 %10, 1
  %19 = icmp eq i64 %18, %7
  br i1 %19, label %8, label %9
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: uwtable
define dso_local void @_Z11dot_productPfS_iS_(float*, float*, i32, float*) #0 {
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i32, align 4
  %8 = alloca float*, align 8
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca i8*, align 8
  store float* %0, float** %5, align 8, !tbaa !7
  store float* %1, float** %6, align 8, !tbaa !7
  store i32 %2, i32* %7, align 4, !tbaa !9
  store float* %3, float** %8, align 8, !tbaa !7
  %13 = alloca [4 x i8*], align 16
  %14 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 0
  %15 = bitcast [4 x i8*]* %13 to float***
  store float** %5, float*** %15, align 16
  %16 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 1
  %17 = bitcast i8** %16 to float***
  store float** %6, float*** %17, align 8
  %18 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 2
  %19 = bitcast i8** %18 to i32**
  store i32* %7, i32** %19, align 16
  %20 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 3
  %21 = bitcast i8** %20 to float***
  store float** %8, float*** %21, align 8
  %22 = call i32 @__cudaPopCallConfiguration(%struct.dim3* nonnull %9, %struct.dim3* nonnull %10, i64* nonnull %11, i8** nonnull %12)
  %23 = load i64, i64* %11, align 8
  %24 = bitcast i8** %12 to %struct.CUstream_st**
  %25 = load %struct.CUstream_st*, %struct.CUstream_st** %24, align 8
  %26 = bitcast %struct.dim3* %9 to i64*
  %27 = load i64, i64* %26, align 8
  %28 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i64 0, i32 2
  %29 = load i32, i32* %28, align 8
  %30 = bitcast %struct.dim3* %10 to i64*
  %31 = load i64, i64* %30, align 8
  %32 = getelementptr inbounds %struct.dim3, %struct.dim3* %10, i64 0, i32 2
  %33 = load i32, i32* %32, align 8
  %34 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, i32, float*)* @_Z11dot_productPfS_iS_ to i8*), i64 %27, i32 %29, i64 %31, i32 %33, i8** nonnull %14, i64 %23, %struct.CUstream_st* %25)
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**) local_unnamed_addr

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #5

; Function Attrs: uwtable
define dso_local void @_Z19dot_product_checkedPfS_iS_(float*, float*, i32, float*) #0 {
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i32, align 4
  %8 = alloca float*, align 8
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca i8*, align 8
  store float* %0, float** %5, align 8, !tbaa !7
  store float* %1, float** %6, align 8, !tbaa !7
  store i32 %2, i32* %7, align 4, !tbaa !9
  store float* %3, float** %8, align 8, !tbaa !7
  %13 = alloca [4 x i8*], align 16
  %14 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 0
  %15 = bitcast [4 x i8*]* %13 to float***
  store float** %5, float*** %15, align 16
  %16 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 1
  %17 = bitcast i8** %16 to float***
  store float** %6, float*** %17, align 8
  %18 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 2
  %19 = bitcast i8** %18 to i32**
  store i32* %7, i32** %19, align 16
  %20 = getelementptr inbounds [4 x i8*], [4 x i8*]* %13, i64 0, i64 3
  %21 = bitcast i8** %20 to float***
  store float** %8, float*** %21, align 8
  %22 = call i32 @__cudaPopCallConfiguration(%struct.dim3* nonnull %9, %struct.dim3* nonnull %10, i64* nonnull %11, i8** nonnull %12)
  %23 = load i64, i64* %11, align 8
  %24 = bitcast i8** %12 to %struct.CUstream_st**
  %25 = load %struct.CUstream_st*, %struct.CUstream_st** %24, align 8
  %26 = bitcast %struct.dim3* %9 to i64*
  %27 = load i64, i64* %26, align 8
  %28 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i64 0, i32 2
  %29 = load i32, i32* %28, align 8
  %30 = bitcast %struct.dim3* %10 to i64*
  %31 = load i64, i64* %30, align 8
  %32 = getelementptr inbounds %struct.dim3, %struct.dim3* %10, i64 0, i32 2
  %33 = load i32, i32* %32, align 8
  %34 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, i32, float*)* @_Z19dot_product_checkedPfS_iS_ to i8*), i64 %27, i32 %29, i64 %31, i32 %33, i8** nonnull %14, i64 %23, %struct.CUstream_st* %25)
  ret void
}

; Function Attrs: norecurse uwtable
define dso_local i32 @main(i32, i8**) local_unnamed_addr #6 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca float, align 4
  %6 = alloca float, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca float*, align 8
  %10 = alloca float*, align 8
  %11 = alloca float*, align 8
  %12 = alloca %"struct.std::chrono::time_point", align 8
  %13 = alloca %"struct.std::chrono::time_point", align 8
  %14 = alloca %"struct.std::chrono::duration.0", align 8
  %15 = alloca %"struct.std::chrono::duration", align 8
  %16 = alloca %struct.dim3, align 8
  %17 = alloca %struct.dim3, align 8
  %18 = alloca %"struct.std::chrono::duration.0", align 8
  %19 = alloca %"struct.std::chrono::duration", align 8
  %20 = alloca %struct.dim3, align 8
  %21 = alloca %struct.dim3, align 8
  %22 = alloca %"struct.std::chrono::duration.0", align 8
  %23 = alloca %"struct.std::chrono::duration", align 8
  %24 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %24) #18
  store i32 10, i32* %3, align 4, !tbaa !9
  %25 = bitcast i32* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %25) #18
  store i32 0, i32* %4, align 4, !tbaa !9
  br label %26

26:                                               ; preds = %28, %2
  %27 = phi i8 [ 0, %2 ], [ 1, %28 ]
  br label %28

28:                                               ; preds = %26, %30
  %29 = call i32 @getopt_long(i32 %0, i8** %1, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.2, i64 0, i64 0), %struct.option* getelementptr inbounds ([3 x %struct.option], [3 x %struct.option]* @_ZZ4mainE12long_options, i64 0, i64 0), i32* nonnull %4) #18
  switch i32 %29, label %234 [
    i32 -1, label %33
    i32 104, label %26
    i32 115, label %30
  ]

30:                                               ; preds = %28
  %31 = load i8*, i8** @optarg, align 8, !tbaa !7
  %32 = call i32 @atoi(i8* %31) #19
  store i32 %32, i32* %3, align 4, !tbaa !9
  br label %28

33:                                               ; preds = %28
  %34 = load i32, i32* %3, align 4, !tbaa !9
  %35 = add nsw i32 %34, 127
  %36 = sdiv i32 %35, 128
  %37 = sext i32 %34 to i64
  %38 = shl nsw i64 %37, 2
  %39 = call noalias i8* @malloc(i64 %38) #18
  %40 = bitcast i8* %39 to float*
  %41 = call noalias i8* @malloc(i64 %38) #18
  %42 = bitcast i8* %41 to float*
  %43 = bitcast float* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %43) #18
  store float 0.000000e+00, float* %5, align 4, !tbaa !3
  %44 = bitcast float* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %44) #18
  store float 0.000000e+00, float* %6, align 4, !tbaa !3
  call void @_Z20create_sample_vectorPfibb(float* %40, i32 %34, i1 zeroext true, i1 zeroext true)
  %45 = load i32, i32* %3, align 4, !tbaa !9
  call void @_Z20create_sample_vectorPfibb(float* %42, i32 %45, i1 zeroext true, i1 zeroext true)
  %46 = icmp ne i8 %27, 0
  br i1 %46, label %47, label %60

47:                                               ; preds = %33
  %48 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0))
  %49 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %48, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %50 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %50) #18
  store i32 20, i32* %7, align 4, !tbaa !9
  %51 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* nonnull dereferenceable(4) %7, i32* nonnull dereferenceable(4) %3)
  %52 = load i32, i32* %51, align 4, !tbaa !9
  call void @_Z19print_array_indexedIfEvPT_i(float* %40, i32 %52)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %50) #18
  %53 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.4, i64 0, i64 0))
  %54 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %53, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %55 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %55) #18
  store i32 20, i32* %8, align 4, !tbaa !9
  %56 = call dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* nonnull dereferenceable(4) %8, i32* nonnull dereferenceable(4) %3)
  %57 = load i32, i32* %56, align 4, !tbaa !9
  call void @_Z19print_array_indexedIfEvPT_i(float* %42, i32 %57)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %55) #18
  %58 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.5, i64 0, i64 0))
  %59 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %58, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %60

60:                                               ; preds = %47, %33
  %61 = bitcast float** %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %61) #18
  %62 = bitcast float** %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %62) #18
  %63 = bitcast float** %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %63) #18
  %64 = load i32, i32* %3, align 4, !tbaa !9
  %65 = sext i32 %64 to i64
  %66 = shl nsw i64 %65, 2
  call fastcc void @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** nonnull %9, i64 %66)
  %67 = load i32, i32* %3, align 4, !tbaa !9
  %68 = sext i32 %67 to i64
  %69 = shl nsw i64 %68, 2
  call fastcc void @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** nonnull %10, i64 %69)
  call fastcc void @_ZL10cudaMallocIfE9cudaErrorPPT_m(float** nonnull %11, i64 4)
  %70 = bitcast float** %9 to i8**
  %71 = load i8*, i8** %70, align 8, !tbaa !7
  %72 = load i32, i32* %3, align 4, !tbaa !9
  %73 = sext i32 %72 to i64
  %74 = shl nsw i64 %73, 2
  %75 = call i32 @cudaMemcpy(i8* %71, i8* %39, i64 %74, i32 1)
  %76 = bitcast float** %10 to i8**
  %77 = load i8*, i8** %76, align 8, !tbaa !7
  %78 = load i32, i32* %3, align 4, !tbaa !9
  %79 = sext i32 %78 to i64
  %80 = shl nsw i64 %79, 2
  %81 = call i32 @cudaMemcpy(i8* %77, i8* %41, i64 %80, i32 1)
  %82 = bitcast %"struct.std::chrono::time_point"* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %82) #18
  %83 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #18
  %84 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %12, i64 0, i32 0, i32 0
  store i64 %83, i64* %84, align 8
  %85 = load i32, i32* %3, align 4, !tbaa !9
  call void @_Z15dot_product_cpuPfS_iS_(float* %40, float* %42, i32 %85, float* nonnull %5)
  %86 = bitcast %"struct.std::chrono::time_point"* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %86) #18
  %87 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #18
  %88 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %13, i64 0, i32 0, i32 0
  store i64 %87, i64* %88, align 8
  %89 = bitcast %"struct.std::chrono::duration.0"* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %89) #18
  %90 = bitcast %"struct.std::chrono::duration"* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %90) #18
  %91 = call i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* nonnull dereferenceable(8) %13, %"struct.std::chrono::time_point"* nonnull dereferenceable(8) %12)
  %92 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %15, i64 0, i32 0
  store i64 %91, i64* %92, align 8
  %93 = call i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* nonnull dereferenceable(8) %15)
  %94 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %14, i64 0, i32 0
  store i64 %93, i64* %94, align 8
  %95 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"* nonnull %14)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %90) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %89) #18
  br i1 %46, label %96, label %107

96:                                               ; preds = %60
  %97 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.6, i64 0, i64 0))
  %98 = load float, float* %5, align 4, !tbaa !3
  %99 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %97, float %98)
  %100 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %99, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %101 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.7, i64 0, i64 0))
  %102 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"* nonnull %101, i64 %95)
  %103 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %102, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.8, i64 0, i64 0))
  %104 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %103, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %105 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.5, i64 0, i64 0))
  %106 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %105, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %107

107:                                              ; preds = %96, %60
  store float 0.000000e+00, float* %6, align 4, !tbaa !3
  %108 = bitcast float** %11 to i8**
  %109 = load i8*, i8** %108, align 8, !tbaa !7
  %110 = call i32 @cudaMemcpy(i8* %109, i8* nonnull %44, i64 4, i32 1)
  %111 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #18
  store i64 %111, i64* %84, align 8
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %16, i32 %36, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %17, i32 128, i32 1, i32 1)
  %112 = bitcast %struct.dim3* %16 to i64*
  %113 = load i64, i64* %112, align 8
  %114 = getelementptr inbounds %struct.dim3, %struct.dim3* %16, i64 0, i32 2
  %115 = load i32, i32* %114, align 8
  %116 = bitcast %struct.dim3* %17 to i64*
  %117 = load i64, i64* %116, align 8
  %118 = getelementptr inbounds %struct.dim3, %struct.dim3* %17, i64 0, i32 2
  %119 = load i32, i32* %118, align 8
  %120 = call i32 @__cudaPushCallConfiguration(i64 %113, i32 %115, i64 %117, i32 %119, i64 0, i8* null)
  %121 = icmp eq i32 %120, 0
  br i1 %121, label %122, label %127

122:                                              ; preds = %107
  %123 = load float*, float** %9, align 8, !tbaa !7
  %124 = load float*, float** %10, align 8, !tbaa !7
  %125 = load i32, i32* %3, align 4, !tbaa !9
  %126 = load float*, float** %11, align 8, !tbaa !7
  call void @_Z19dot_product_checkedPfS_iS_(float* %123, float* %124, i32 %125, float* %126)
  br label %127

127:                                              ; preds = %107, %122
  %128 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #18
  store i64 %128, i64* %88, align 8
  %129 = bitcast %"struct.std::chrono::duration.0"* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %129) #18
  %130 = bitcast %"struct.std::chrono::duration"* %19 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130) #18
  %131 = call i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* nonnull dereferenceable(8) %13, %"struct.std::chrono::time_point"* nonnull dereferenceable(8) %12)
  %132 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %19, i64 0, i32 0
  store i64 %131, i64* %132, align 8
  %133 = call i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* nonnull dereferenceable(8) %19)
  %134 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %18, i64 0, i32 0
  store i64 %133, i64* %134, align 8
  %135 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"* nonnull %18)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %129) #18
  %136 = load i8*, i8** %108, align 8, !tbaa !7
  %137 = call i32 @cudaMemcpy(i8* nonnull %44, i8* %136, i64 4, i32 2)
  %138 = call i32 @cudaGetLastError()
  %139 = icmp eq i32 %138, 0
  br i1 %139, label %149, label %140

140:                                              ; preds = %127
  %141 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cerr, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.9, i64 0, i64 0))
  %142 = call i8* @cudaGetErrorString(i32 %138)
  %143 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %141, i8* %142)
  %144 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %143, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.10, i64 0, i64 0))
  %145 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %144, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.11, i64 0, i64 0))
  %146 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* nonnull dereferenceable(272) %145, i8 signext 58)
  %147 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull %146, i32 161)
  %148 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %147, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %149

149:                                              ; preds = %127, %140
  br i1 %46, label %154, label %150

150:                                              ; preds = %149
  %151 = load float, float* %5, align 4, !tbaa !3
  %152 = load float, float* %6, align 4, !tbaa !3
  %153 = call zeroext i1 @_Z14check_equalityIfEbT_S0_fb(float %151, float %152, float 0x3E45798EE0000000, i1 zeroext %46)
  br label %171

154:                                              ; preds = %149
  %155 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.12, i64 0, i64 0))
  %156 = load float, float* %6, align 4, !tbaa !3
  %157 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %155, float %156)
  %158 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %157, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %159 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.7, i64 0, i64 0))
  %160 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"* nonnull %159, i64 %135)
  %161 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %160, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.8, i64 0, i64 0))
  %162 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %161, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %163 = load float, float* %5, align 4, !tbaa !3
  %164 = load float, float* %6, align 4, !tbaa !3
  %165 = call zeroext i1 @_Z14check_equalityIfEbT_S0_fb(float %163, float %164, float 0x3E45798EE0000000, i1 zeroext %46)
  %166 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.13, i64 0, i64 0))
  %167 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEb(%"class.std::basic_ostream"* nonnull %166, i1 zeroext %165)
  %168 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %167, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %169 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.5, i64 0, i64 0))
  %170 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %169, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %171

171:                                              ; preds = %150, %154
  %172 = phi float [ %163, %154 ], [ %151, %150 ]
  store float 0.000000e+00, float* %6, align 4, !tbaa !3
  %173 = load i8*, i8** %108, align 8, !tbaa !7
  %174 = call i32 @cudaMemcpy(i8* %173, i8* nonnull %44, i64 4, i32 1)
  %175 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #18
  store i64 %175, i64* %84, align 8
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %20, i32 %36, i32 1, i32 1)
  call void @_ZN4dim3C2Ejjj(%struct.dim3* nonnull %21, i32 128, i32 1, i32 1)
  %176 = bitcast %struct.dim3* %20 to i64*
  %177 = load i64, i64* %176, align 8
  %178 = getelementptr inbounds %struct.dim3, %struct.dim3* %20, i64 0, i32 2
  %179 = load i32, i32* %178, align 8
  %180 = bitcast %struct.dim3* %21 to i64*
  %181 = load i64, i64* %180, align 8
  %182 = getelementptr inbounds %struct.dim3, %struct.dim3* %21, i64 0, i32 2
  %183 = load i32, i32* %182, align 8
  %184 = call i32 @__cudaPushCallConfiguration(i64 %177, i32 %179, i64 %181, i32 %183, i64 0, i8* null)
  %185 = icmp eq i32 %184, 0
  br i1 %185, label %186, label %191

186:                                              ; preds = %171
  %187 = load float*, float** %9, align 8, !tbaa !7
  %188 = load float*, float** %10, align 8, !tbaa !7
  %189 = load i32, i32* %3, align 4, !tbaa !9
  %190 = load float*, float** %11, align 8, !tbaa !7
  call void @_Z11dot_productPfS_iS_(float* %187, float* %188, i32 %189, float* %190)
  br label %191

191:                                              ; preds = %171, %186
  %192 = call i64 @_ZNSt6chrono3_V212system_clock3nowEv() #18
  store i64 %192, i64* %88, align 8
  %193 = bitcast %"struct.std::chrono::duration.0"* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %193) #18
  %194 = bitcast %"struct.std::chrono::duration"* %23 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %194) #18
  %195 = call i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* nonnull dereferenceable(8) %13, %"struct.std::chrono::time_point"* nonnull dereferenceable(8) %12)
  %196 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %23, i64 0, i32 0
  store i64 %195, i64* %196, align 8
  %197 = call i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* nonnull dereferenceable(8) %23)
  %198 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %22, i64 0, i32 0
  store i64 %197, i64* %198, align 8
  %199 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"* nonnull %22)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %194) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %193) #18
  %200 = load i8*, i8** %108, align 8, !tbaa !7
  %201 = call i32 @cudaMemcpy(i8* nonnull %44, i8* %200, i64 4, i32 2)
  %202 = call i32 @cudaGetLastError()
  %203 = icmp eq i32 %202, 0
  br i1 %203, label %213, label %204

204:                                              ; preds = %191
  %205 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cerr, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.9, i64 0, i64 0))
  %206 = call i8* @cudaGetErrorString(i32 %202)
  %207 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %205, i8* %206)
  %208 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %207, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.10, i64 0, i64 0))
  %209 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %208, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.11, i64 0, i64 0))
  %210 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* nonnull dereferenceable(272) %209, i8 signext 58)
  %211 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull %210, i32 182)
  %212 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %211, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %213

213:                                              ; preds = %191, %204
  br i1 %46, label %217, label %214

214:                                              ; preds = %213
  %215 = load float, float* %6, align 4, !tbaa !3
  %216 = call zeroext i1 @_Z14check_equalityIfEbT_S0_fb(float %172, float %215, float 0x3E45798EE0000000, i1 zeroext %46)
  br label %233

217:                                              ; preds = %213
  %218 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.14, i64 0, i64 0))
  %219 = load float, float* %6, align 4, !tbaa !3
  %220 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %218, float %219)
  %221 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %220, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %222 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.7, i64 0, i64 0))
  %223 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"* nonnull %222, i64 %199)
  %224 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %223, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.8, i64 0, i64 0))
  %225 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %224, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %226 = load float, float* %6, align 4, !tbaa !3
  %227 = call zeroext i1 @_Z14check_equalityIfEbT_S0_fb(float %172, float %226, float 0x3E45798EE0000000, i1 zeroext %46)
  %228 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.13, i64 0, i64 0))
  %229 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEb(%"class.std::basic_ostream"* nonnull %228, i1 zeroext %227)
  %230 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %229, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %231 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.5, i64 0, i64 0))
  %232 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %231, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %233

233:                                              ; preds = %214, %217
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %86) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %82) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %63) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %62) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %61) #18
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %44) #18
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %43) #18
  br label %234

234:                                              ; preds = %28, %233
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %25) #18
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %24) #18
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @getopt_long(i32, i8**, i8*, %struct.option*, i32*) local_unnamed_addr #2

; Function Attrs: inlinehint nounwind readonly uwtable
define available_externally dso_local i32 @atoi(i8* nonnull) local_unnamed_addr #7 {
  %2 = tail call i64 @strtol(i8* nocapture nonnull %0, i8** null, i32 10) #18
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #8

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_Z20create_sample_vectorPfibb(float*, i32, i1 zeroext, i1 zeroext) local_unnamed_addr #9 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %5 = alloca %"class.std::random_device", align 8
  %6 = alloca %"class.std::__cxx11::basic_string", align 8
  %7 = alloca %"class.std::allocator", align 1
  %8 = alloca %"class.std::mersenne_twister_engine", align 8
  %9 = alloca %"class.std::uniform_real_distribution", align 4
  br i1 %2, label %14, label %10

10:                                               ; preds = %4
  %11 = icmp sgt i32 %1, 0
  br i1 %11, label %12, label %74

12:                                               ; preds = %10
  %13 = zext i32 %1 to i64
  br label %69

14:                                               ; preds = %4
  %15 = bitcast %"class.std::random_device"* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5000, i8* nonnull %15) #18
  %16 = bitcast %"class.std::__cxx11::basic_string"* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %16) #18
  %17 = getelementptr inbounds %"class.std::allocator", %"class.std::allocator"* %7, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %17) #18
  call void @_ZNSaIcEC2Ev(%"class.std::allocator"* nonnull %7) #18
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_(%"class.std::__cxx11::basic_string"* nonnull %6, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.15, i64 0, i64 0), %"class.std::allocator"* nonnull dereferenceable(1) %7)
          to label %18 unwind label %31

18:                                               ; preds = %14
  invoke void @_ZNSt13random_deviceC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"* nonnull %5, %"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %6)
          to label %19 unwind label %35

19:                                               ; preds = %18
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %6) #18
  call void @_ZNSaIcED2Ev(%"class.std::allocator"* nonnull %7) #18
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %17) #18
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %16) #18
  %20 = bitcast %"class.std::mersenne_twister_engine"* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5000, i8* nonnull %20) #18
  %21 = invoke i32 @_ZNSt13random_deviceclEv(%"class.std::random_device"* nonnull %5)
          to label %22 unwind label %42

22:                                               ; preds = %19
  %23 = zext i32 %21 to i64
  invoke void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em(%"class.std::mersenne_twister_engine"* nonnull %8, i64 %23)
          to label %24 unwind label %42

24:                                               ; preds = %22
  %25 = bitcast %"class.std::uniform_real_distribution"* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %25) #18
  invoke void @_ZNSt25uniform_real_distributionIfEC2Eff(%"class.std::uniform_real_distribution"* nonnull %9, float 0.000000e+00, float 1.000000e+00)
          to label %26 unwind label %46

26:                                               ; preds = %24
  %27 = icmp sgt i32 %1, 0
  br i1 %27, label %28, label %30

28:                                               ; preds = %26
  %29 = zext i32 %1 to i64
  br label %48

30:                                               ; preds = %51, %26
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %25) #18
  call void @llvm.lifetime.end.p0i8(i64 5000, i8* nonnull %20) #18
  call void @_ZNSt13random_deviceD2Ev(%"class.std::random_device"* nonnull %5) #18
  call void @llvm.lifetime.end.p0i8(i64 5000, i8* nonnull %15) #18
  br label %74

31:                                               ; preds = %14
  %32 = landingpad { i8*, i32 }
          cleanup
  %33 = extractvalue { i8*, i32 } %32, 0
  %34 = extractvalue { i8*, i32 } %32, 1
  br label %39

35:                                               ; preds = %18
  %36 = landingpad { i8*, i32 }
          cleanup
  %37 = extractvalue { i8*, i32 } %36, 0
  %38 = extractvalue { i8*, i32 } %36, 1
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %6) #18
  br label %39

39:                                               ; preds = %35, %31
  %40 = phi i8* [ %37, %35 ], [ %33, %31 ]
  %41 = phi i32 [ %38, %35 ], [ %34, %31 ]
  call void @_ZNSaIcED2Ev(%"class.std::allocator"* nonnull %7) #18
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %17) #18
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %16) #18
  br label %64

42:                                               ; preds = %22, %19
  %43 = landingpad { i8*, i32 }
          cleanup
  %44 = extractvalue { i8*, i32 } %43, 0
  %45 = extractvalue { i8*, i32 } %43, 1
  br label %61

46:                                               ; preds = %24
  %47 = landingpad { i8*, i32 }
          cleanup
  br label %57

48:                                               ; preds = %51, %28
  %49 = phi i64 [ 0, %28 ], [ %53, %51 ]
  %50 = invoke float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_(%"class.std::uniform_real_distribution"* nonnull %9, %"class.std::mersenne_twister_engine"* nonnull dereferenceable(5000) %8)
          to label %51 unwind label %55

51:                                               ; preds = %48
  %52 = getelementptr inbounds float, float* %0, i64 %49
  store float %50, float* %52, align 4, !tbaa !3
  %53 = add nuw nsw i64 %49, 1
  %54 = icmp eq i64 %53, %29
  br i1 %54, label %30, label %48

55:                                               ; preds = %48
  %56 = landingpad { i8*, i32 }
          cleanup
  br label %57

57:                                               ; preds = %55, %46
  %58 = phi { i8*, i32 } [ %56, %55 ], [ %47, %46 ]
  %59 = extractvalue { i8*, i32 } %58, 0
  %60 = extractvalue { i8*, i32 } %58, 1
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %25) #18
  br label %61

61:                                               ; preds = %57, %42
  %62 = phi i8* [ %59, %57 ], [ %44, %42 ]
  %63 = phi i32 [ %60, %57 ], [ %45, %42 ]
  call void @llvm.lifetime.end.p0i8(i64 5000, i8* nonnull %20) #18
  call void @_ZNSt13random_deviceD2Ev(%"class.std::random_device"* nonnull %5) #18
  br label %64

64:                                               ; preds = %61, %39
  %65 = phi i8* [ %62, %61 ], [ %40, %39 ]
  %66 = phi i32 [ %63, %61 ], [ %41, %39 ]
  call void @llvm.lifetime.end.p0i8(i64 5000, i8* nonnull %15) #18
  %67 = insertvalue { i8*, i32 } undef, i8* %65, 0
  %68 = insertvalue { i8*, i32 } %67, i32 %66, 1
  resume { i8*, i32 } %68

69:                                               ; preds = %69, %12
  %70 = phi i64 [ 0, %12 ], [ %72, %69 ]
  %71 = getelementptr inbounds float, float* %0, i64 %70
  store float 1.000000e+00, float* %71, align 4, !tbaa !3
  %72 = add nuw nsw i64 %70, 1
  %73 = icmp eq i64 %72, %13
  br i1 %73, label %74, label %69

74:                                               ; preds = %69, %10, %30
  %75 = icmp sgt i32 %1, 0
  %76 = and i1 %75, %3
  br i1 %76, label %77, label %98

77:                                               ; preds = %74
  %78 = zext i32 %1 to i64
  br label %83

79:                                               ; preds = %83
  %80 = icmp sgt i32 %1, 0
  br i1 %80, label %81, label %98

81:                                               ; preds = %79
  %82 = zext i32 %1 to i64
  br label %91

83:                                               ; preds = %83, %77
  %84 = phi i64 [ 0, %77 ], [ %89, %83 ]
  %85 = phi float [ 0.000000e+00, %77 ], [ %88, %83 ]
  %86 = getelementptr inbounds float, float* %0, i64 %84
  %87 = load float, float* %86, align 4, !tbaa !3
  %88 = fadd contract float %85, %87
  %89 = add nuw nsw i64 %84, 1
  %90 = icmp eq i64 %89, %78
  br i1 %90, label %79, label %83

91:                                               ; preds = %91, %81
  %92 = phi i64 [ 0, %81 ], [ %96, %91 ]
  %93 = getelementptr inbounds float, float* %0, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !3
  %95 = fdiv float %94, %88
  store float %95, float* %93, align 4, !tbaa !3
  %96 = add nuw nsw i64 %92, 1
  %97 = icmp eq i64 %96, %82
  br i1 %97, label %98, label %91

98:                                               ; preds = %91, %79, %74
  ret void
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272), i8*) local_unnamed_addr #9 {
  %3 = icmp eq i8* %1, null
  br i1 %3, label %4, label %13

4:                                                ; preds = %2
  %5 = bitcast %"class.std::basic_ostream"* %0 to i8**
  %6 = load i8*, i8** %5, align 8, !tbaa !11
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

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"*, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)*) local_unnamed_addr #0 align 2 {
  %3 = tail call dereferenceable(272) %"class.std::basic_ostream"* %1(%"class.std::basic_ostream"* dereferenceable(272) %0)
  ret %"class.std::basic_ostream"* %3
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) #9 {
  %2 = bitcast %"class.std::basic_ostream"* %0 to i8**
  %3 = load i8*, i8** %2, align 8, !tbaa !11
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

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_Z19print_array_indexedIfEvPT_i(float*, i32) local_unnamed_addr #9 comdat {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %2
  %5 = zext i32 %1 to i64
  br label %8

6:                                                ; preds = %8, %2
  %7 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull @_ZSt4cout, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  ret void

8:                                                ; preds = %8, %4
  %9 = phi i64 [ 0, %4 ], [ %17, %8 ]
  %10 = trunc i64 %9 to i32
  %11 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i32 %10)
  %12 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %11, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.17, i64 0, i64 0))
  %13 = getelementptr inbounds float, float* %0, i64 %9
  %14 = load float, float* %13, align 4, !tbaa !3
  %15 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %12, float %14)
  %16 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %15, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %17 = add nuw nsw i64 %9, 1
  %18 = icmp eq i64 %17, %5
  br i1 %18, label %6, label %8
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(4) i32* @_ZSt3minIiERKT_S2_S2_(i32* dereferenceable(4), i32* dereferenceable(4)) local_unnamed_addr #10 comdat {
  %3 = load i32, i32* %1, align 4, !tbaa !9
  %4 = load i32, i32* %0, align 4, !tbaa !9
  %5 = icmp slt i32 %3, %4
  %6 = select i1 %5, i32* %1, i32* %0
  ret i32* %6
}

; Function Attrs: inlinehint norecurse uwtable
define internal fastcc void @_ZL10cudaMallocIfE9cudaErrorPPT_m(float**, i64) unnamed_addr #11 {
  %3 = bitcast float** %0 to i8**
  %4 = tail call i32 @cudaMalloc(i8** %3, i64 %1)
  ret void
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local i64 @_ZNSt6chrono3_V212system_clock3nowEv() local_unnamed_addr #2

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsr13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE(%"struct.std::chrono::duration"* dereferenceable(8)) local_unnamed_addr #0 comdat {
  %2 = tail call i64 @_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000EEEES2_ILl1ELl1000000EElLb1ELb0EE6__castIlS2_ILl1ELl1000000000EEEES4_RKNS1_IT_T0_EE(%"struct.std::chrono::duration"* nonnull dereferenceable(8) %0)
  ret i64 %2
}

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE(%"struct.std::chrono::time_point"* dereferenceable(8), %"struct.std::chrono::time_point"* dereferenceable(8)) local_unnamed_addr #0 comdat {
  %3 = alloca %"struct.std::chrono::duration", align 8
  %4 = alloca %"struct.std::chrono::duration", align 8
  %5 = bitcast %"struct.std::chrono::duration"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5) #18
  %6 = tail call i64 @_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv(%"struct.std::chrono::time_point"* nonnull %0)
  %7 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %3, i64 0, i32 0
  store i64 %6, i64* %7, align 8
  %8 = bitcast %"struct.std::chrono::duration"* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %8) #18
  %9 = tail call i64 @_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv(%"struct.std::chrono::time_point"* nonnull %1)
  %10 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %4, i64 0, i32 0
  store i64 %9, i64* %10, align 8
  %11 = call i64 @_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_(%"struct.std::chrono::duration"* nonnull dereferenceable(8) %3, %"struct.std::chrono::duration"* nonnull dereferenceable(8) %4)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %8) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5) #18
  ret i64 %11
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000EEE5countEv(%"struct.std::chrono::duration.0"*) local_unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %0, i64 0, i32 0
  %3 = load i64, i64* %2, align 8, !tbaa !13
  ret i64 %3
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"*, float) local_unnamed_addr #0 align 2 {
  %3 = fpext float %1 to double
  %4 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* %0, double %3)
  ret %"class.std::basic_ostream"* %4
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEl(%"class.std::basic_ostream"*, i64) local_unnamed_addr #0 align 2 {
  %3 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIlEERSoT_(%"class.std::basic_ostream"* %0, i64 %1)
  ret %"class.std::basic_ostream"* %3
}

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3*, i32, i32, i32) unnamed_addr #12 comdat align 2 {
  %5 = getelementptr inbounds %struct.dim3, %struct.dim3* %0, i64 0, i32 0
  store i32 %1, i32* %5, align 4, !tbaa !16
  %6 = getelementptr inbounds %struct.dim3, %struct.dim3* %0, i64 0, i32 1
  store i32 %2, i32* %6, align 4, !tbaa !18
  %7 = getelementptr inbounds %struct.dim3, %struct.dim3* %0, i64 0, i32 2
  store i32 %3, i32* %7, align 4, !tbaa !19
  ret void
}

declare dso_local i32 @cudaGetLastError() local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%"class.std::basic_ostream"* dereferenceable(272), i8 signext) local_unnamed_addr #9 {
  %3 = alloca i8, align 1
  store i8 %1, i8* %3, align 1, !tbaa !20
  %4 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %0, i8* nonnull %3, i64 1)
  ret %"class.std::basic_ostream"* %4
}

declare dso_local i8* @cudaGetErrorString(i32) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"*, i32) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local zeroext i1 @_Z14check_equalityIfEbT_S0_fb(float, float, float, i1 zeroext) local_unnamed_addr #9 comdat {
  %5 = fsub contract float %0, %1
  %6 = tail call float @_ZSt3absf(float %5)
  %7 = fcmp ule float %6, %2
  %8 = xor i1 %3, true
  %9 = or i1 %7, %8
  br i1 %9, label %18, label %10

10:                                               ; preds = %4
  %11 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.18, i64 0, i64 0))
  %12 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %11, float %0)
  %13 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %12, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.19, i64 0, i64 0))
  %14 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %13, float %1)
  %15 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %14, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.20, i64 0, i64 0))
  %16 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %15, float %6)
  %17 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %16, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %18

18:                                               ; preds = %4, %10
  %19 = phi i1 [ false, %10 ], [ %7, %4 ]
  ret i1 %19
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEb(%"class.std::basic_ostream"*, i1 zeroext) local_unnamed_addr #0 align 2 {
  %3 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIbEERSoT_(%"class.std::basic_ostream"* %0, i1 zeroext %1)
  ret %"class.std::basic_ostream"* %3
}

; Function Attrs: nofree nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #8

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSaIcEC2Ev(%"class.std::allocator"*) unnamed_addr #12 align 2 {
  %2 = bitcast %"class.std::allocator"* %0 to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcEC2Ev(%"class.__gnu_cxx::new_allocator"* %2) #18
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_(%"class.std::__cxx11::basic_string"*, i8*, %"class.std::allocator"* dereferenceable(1)) unnamed_addr #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 0
  %5 = tail call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %0)
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcRKS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %4, i8* %5, %"class.std::allocator"* nonnull dereferenceable(1) %2)
  %6 = icmp eq i8* %1, null
  br i1 %6, label %10, label %7

7:                                                ; preds = %3
  %8 = tail call i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8* nonnull %1)
  %9 = getelementptr inbounds i8, i8* %1, i64 %8
  br label %10

10:                                               ; preds = %3, %7
  %11 = phi i8* [ %9, %7 ], [ inttoptr (i64 -1 to i8*), %3 ]
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_(%"class.std::__cxx11::basic_string"* %0, i8* %1, i8* %11)
          to label %12 unwind label %13

12:                                               ; preds = %10
  ret void

13:                                               ; preds = %10
  %14 = landingpad { i8*, i32 }
          cleanup
  %15 = bitcast %"class.std::__cxx11::basic_string"* %0 to %"class.std::allocator"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator"* %15) #18
  resume { i8*, i32 } %14
}

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt13random_deviceC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"*, %"class.std::__cxx11::basic_string"* dereferenceable(32)) unnamed_addr #0 comdat align 2 {
  tail call void @_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"* %0, %"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %1)
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"*) unnamed_addr #12 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv(%"class.std::__cxx11::basic_string"* %0)
          to label %2 unwind label %4

2:                                                ; preds = %1
  %3 = bitcast %"class.std::__cxx11::basic_string"* %0 to %"class.std::allocator"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator"* %3) #18
  ret void

4:                                                ; preds = %1
  %5 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %6 = extractvalue { i8*, i32 } %5, 0
  %7 = bitcast %"class.std::__cxx11::basic_string"* %0 to %"class.std::allocator"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator"* %7) #18
  tail call void @__cxa_call_unexpected(i8* %6) #20
  unreachable
}

; Function Attrs: uwtable
define linkonce_odr dso_local i32 @_ZNSt13random_deviceclEv(%"class.std::random_device"*) local_unnamed_addr #0 comdat align 2 {
  %2 = tail call i32 @_ZNSt13random_device9_M_getvalEv(%"class.std::random_device"* %0)
  ret i32 %2
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em(%"class.std::mersenne_twister_engine"*, i64) unnamed_addr #0 comdat align 2 {
  tail call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm(%"class.std::mersenne_twister_engine"* %0, i64 %1)
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt25uniform_real_distributionIfEC2Eff(%"class.std::uniform_real_distribution"*, float, float) unnamed_addr #0 comdat align 2 {
  %4 = getelementptr inbounds %"class.std::uniform_real_distribution", %"class.std::uniform_real_distribution"* %0, i64 0, i32 0
  tail call void @_ZNSt25uniform_real_distributionIfE10param_typeC2Eff(%"struct.std::uniform_real_distribution<float>::param_type"* %4, float %1, float %2)
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_(%"class.std::uniform_real_distribution"*, %"class.std::mersenne_twister_engine"* dereferenceable(5000)) local_unnamed_addr #0 comdat align 2 {
  %3 = getelementptr inbounds %"class.std::uniform_real_distribution", %"class.std::uniform_real_distribution"* %0, i64 0, i32 0
  %4 = tail call float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_RKNS0_10param_typeE(%"class.std::uniform_real_distribution"* %0, %"class.std::mersenne_twister_engine"* nonnull dereferenceable(5000) %1, %"struct.std::uniform_real_distribution<float>::param_type"* dereferenceable(8) %3)
  ret float %4
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt13random_deviceD2Ev(%"class.std::random_device"*) unnamed_addr #12 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void @_ZNSt13random_device7_M_finiEv(%"class.std::random_device"* %0)
          to label %2 unwind label %3

2:                                                ; preds = %1
  ret void

3:                                                ; preds = %1
  %4 = landingpad { i8*, i32 }
          catch i8* null
  %5 = extractvalue { i8*, i32 } %4, 0
  tail call void @__clang_call_terminate(i8* %5) #20
  unreachable
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcEC2Ev(%"class.__gnu_cxx::new_allocator"*) unnamed_addr #12 comdat align 2 {
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"*) local_unnamed_addr #12 align 2 {
  %2 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 2
  %3 = bitcast %union.anon.1* %2 to i8*
  %4 = tail call i8* @_ZNSt14pointer_traitsIPcE10pointer_toERc(i8* nonnull dereferenceable(1) %3) #18
  ret i8* %4
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcRKS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"*, i8*, %"class.std::allocator"* dereferenceable(1)) unnamed_addr #12 align 2 {
  %4 = bitcast %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %0 to %"class.std::allocator"*
  tail call void @_ZNSaIcEC2ERKS_(%"class.std::allocator"* %4, %"class.std::allocator"* nonnull dereferenceable(1) %2) #18
  %5 = getelementptr inbounds %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %0, i64 0, i32 0
  store i8* %1, i8** %5, align 8, !tbaa !21
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_(%"class.std::__cxx11::basic_string"*, i8*, i8*) local_unnamed_addr #0 comdat align 2 {
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPKcEEvT_S8_St12__false_type(%"class.std::__cxx11::basic_string"* %0, i8* %1, i8* %2)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8*) local_unnamed_addr #12 comdat align 2 {
  %2 = tail call i64 @strlen(i8* %0) #18
  ret i64 %2
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSaIcED2Ev(%"class.std::allocator"*) unnamed_addr #12 align 2 {
  %2 = bitcast %"class.std::allocator"* %0 to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcED2Ev(%"class.__gnu_cxx::new_allocator"* %2) #18
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i8* @_ZNSt14pointer_traitsIPcE10pointer_toERc(i8* dereferenceable(1)) local_unnamed_addr #12 comdat align 2 {
  %2 = tail call i8* @_ZSt9addressofIcEPT_RS0_(i8* nonnull dereferenceable(1) %0) #18
  ret i8* %2
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt9addressofIcEPT_RS0_(i8* dereferenceable(1)) local_unnamed_addr #10 comdat {
  %2 = tail call i8* @_ZSt11__addressofIcEPT_RS0_(i8* nonnull dereferenceable(1) %0) #18
  ret i8* %2
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt11__addressofIcEPT_RS0_(i8* dereferenceable(1)) local_unnamed_addr #10 comdat {
  ret i8* %0
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSaIcEC2ERKS_(%"class.std::allocator"*, %"class.std::allocator"* dereferenceable(1)) unnamed_addr #12 align 2 {
  %3 = bitcast %"class.std::allocator"* %0 to %"class.__gnu_cxx::new_allocator"*
  %4 = bitcast %"class.std::allocator"* %1 to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcEC2ERKS1_(%"class.__gnu_cxx::new_allocator"* %3, %"class.__gnu_cxx::new_allocator"* nonnull dereferenceable(1) %4) #18
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcEC2ERKS1_(%"class.__gnu_cxx::new_allocator"*, %"class.__gnu_cxx::new_allocator"* dereferenceable(1)) unnamed_addr #12 comdat align 2 {
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPKcEEvT_S8_St12__false_type(%"class.std::__cxx11::basic_string"*, i8*, i8*) local_unnamed_addr #0 comdat align 2 {
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag(%"class.std::__cxx11::basic_string"* %0, i8* %1, i8* %2)
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag(%"class.std::__cxx11::basic_string"*, i8*, i8*) local_unnamed_addr #0 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca i64, align 8
  %5 = tail call zeroext i1 @_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_(i8* %1)
  %6 = xor i1 %5, true
  %7 = icmp eq i8* %1, %2
  %8 = or i1 %7, %6
  br i1 %8, label %10, label %9

9:                                                ; preds = %3
  tail call void @_ZSt19__throw_logic_errorPKc(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.16, i64 0, i64 0)) #21
  unreachable

10:                                               ; preds = %3
  %11 = bitcast i64* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %11) #18
  %12 = tail call i64 @_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_(i8* %1, i8* %2)
  store i64 %12, i64* %4, align 8, !tbaa !23
  %13 = icmp ugt i64 %12, 15
  br i1 %13, label %14, label %17

14:                                               ; preds = %10
  %15 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* %0, i64* nonnull dereferenceable(8) %4, i64 0)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc(%"class.std::__cxx11::basic_string"* %0, i8* %15)
  %16 = load i64, i64* %4, align 8, !tbaa !23
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm(%"class.std::__cxx11::basic_string"* %0, i64 %16)
  br label %17

17:                                               ; preds = %14, %10
  %18 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %0)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_(i8* %18, i8* %1, i8* %2) #18
  %19 = load i64, i64* %4, align 8, !tbaa !23
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm(%"class.std::__cxx11::basic_string"* %0, i64 %19)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %11) #18
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_(i8*) local_unnamed_addr #10 comdat {
  %2 = icmp eq i8* %0, null
  ret i1 %2
}

; Function Attrs: noreturn
declare dso_local void @_ZSt19__throw_logic_errorPKc(i8*) local_unnamed_addr #13

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i64 @_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_(i8*, i8*) local_unnamed_addr #9 comdat {
  %3 = alloca i8*, align 8
  store i8* %0, i8** %3, align 8, !tbaa !7
  call void @_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_(i8** nonnull dereferenceable(8) %3)
  %4 = call i64 @_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag(i8* %0, i8* %1)
  ret i64 %4
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc(%"class.std::__cxx11::basic_string"*, i8*) local_unnamed_addr #12 align 2 {
  %3 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 0, i32 0
  store i8* %1, i8** %3, align 8, !tbaa !24
  ret void
}

declare dso_local i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"*, i64* dereferenceable(8), i64) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm(%"class.std::__cxx11::basic_string"*, i64) local_unnamed_addr #12 align 2 {
  %3 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 2, i32 0
  store i64 %1, i64* %3, align 8, !tbaa !20
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_(i8*, i8*, i8*) local_unnamed_addr #12 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = ptrtoint i8* %2 to i64
  %5 = ptrtoint i8* %1 to i64
  %6 = sub i64 %4, %5
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm(i8* %0, i8* %1, i64 %6)
          to label %7 unwind label %8

7:                                                ; preds = %3
  ret void

8:                                                ; preds = %3
  %9 = landingpad { i8*, i32 }
          catch i8* null
  %10 = extractvalue { i8*, i32 } %9, 0
  tail call void @__clang_call_terminate(i8* %10) #20
  unreachable
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"*) local_unnamed_addr #12 align 2 {
  %2 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 0, i32 0
  %3 = load i8*, i8** %2, align 8, !tbaa !24
  ret i8* %3
}

declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv(%"class.std::__cxx11::basic_string"*) local_unnamed_addr #0 align 2 {
  %2 = tail call zeroext i1 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv(%"class.std::__cxx11::basic_string"* %0)
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 2, i32 0
  %5 = load i64, i64* %4, align 8, !tbaa !20
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm(%"class.std::__cxx11::basic_string"* %0, i64 %5) #18
  br label %6

6:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) local_unnamed_addr #14 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #18
  tail call void @_ZSt9terminatev() #20
  unreachable
}

declare dso_local void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm(%"class.std::__cxx11::basic_string"*, i64) local_unnamed_addr #0 align 2 {
  %3 = alloca i8, align 1
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm(%"class.std::__cxx11::basic_string"* %0, i64 %1)
  %4 = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %0)
  %5 = getelementptr inbounds i8, i8* %4, i64 %1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %3) #18
  store i8 0, i8* %3, align 1, !tbaa !20
  call void @_ZNSt11char_traitsIcE6assignERcRKc(i8* dereferenceable(1) %5, i8* nonnull dereferenceable(1) %3) #18
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %3) #18
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag(i8*, i8*) local_unnamed_addr #10 comdat {
  %3 = ptrtoint i8* %1 to i64
  %4 = ptrtoint i8* %0 to i64
  %5 = sub i64 %3, %4
  ret i64 %5
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_(i8** dereferenceable(8)) local_unnamed_addr #10 comdat {
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm(i8*, i8*, i64) local_unnamed_addr #0 align 2 {
  %4 = icmp eq i64 %2, 1
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  tail call void @_ZNSt11char_traitsIcE6assignERcRKc(i8* dereferenceable(1) %0, i8* dereferenceable(1) %1) #18
  br label %8

6:                                                ; preds = %3
  %7 = tail call i8* @_ZNSt11char_traitsIcE4copyEPcPKcm(i8* %0, i8* %1, i64 %2)
  br label %8

8:                                                ; preds = %6, %5
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt11char_traitsIcE6assignERcRKc(i8* dereferenceable(1), i8* dereferenceable(1)) local_unnamed_addr #12 comdat align 2 {
  %3 = load i8, i8* %1, align 1, !tbaa !20
  store i8 %3, i8* %0, align 1, !tbaa !20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i8* @_ZNSt11char_traitsIcE4copyEPcPKcm(i8*, i8*, i64) local_unnamed_addr #12 comdat align 2 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %6, label %5

5:                                                ; preds = %3
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %2, i1 false)
  br label %6

6:                                                ; preds = %3, %5
  ret i8* %0
}

; Function Attrs: uwtable
define available_externally dso_local zeroext i1 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv(%"class.std::__cxx11::basic_string"*) local_unnamed_addr #0 align 2 {
  %2 = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %0)
  %3 = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %0)
  %4 = icmp eq i8* %2, %3
  ret i1 %4
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm(%"class.std::__cxx11::basic_string"*, i64) local_unnamed_addr #12 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = tail call dereferenceable(1) %"class.std::allocator"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv(%"class.std::__cxx11::basic_string"* %0)
  %4 = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %0)
  %5 = add i64 %1, 1
  invoke void @_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm(%"class.std::allocator"* nonnull dereferenceable(1) %3, i8* %4, i64 %5)
          to label %6 unwind label %7

6:                                                ; preds = %2
  ret void

7:                                                ; preds = %2
  %8 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %9 = extractvalue { i8*, i32 } %8, 0
  tail call void @__cxa_call_unexpected(i8* %9) #20
  unreachable
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"*) local_unnamed_addr #12 align 2 {
  %2 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 2
  %3 = bitcast %union.anon.1* %2 to i8*
  %4 = tail call i8* @_ZNSt14pointer_traitsIPKcE10pointer_toERS0_(i8* nonnull dereferenceable(1) %3) #18
  ret i8* %4
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i8* @_ZNSt14pointer_traitsIPKcE10pointer_toERS0_(i8* dereferenceable(1)) local_unnamed_addr #12 comdat align 2 {
  %2 = tail call i8* @_ZSt9addressofIKcEPT_RS1_(i8* nonnull dereferenceable(1) %0) #18
  ret i8* %2
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt9addressofIKcEPT_RS1_(i8* dereferenceable(1)) local_unnamed_addr #10 comdat {
  %2 = tail call i8* @_ZSt11__addressofIKcEPT_RS1_(i8* nonnull dereferenceable(1) %0) #18
  ret i8* %2
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt11__addressofIKcEPT_RS1_(i8* dereferenceable(1)) local_unnamed_addr #10 comdat {
  ret i8* %0
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm(%"class.std::allocator"* dereferenceable(1), i8*, i64) local_unnamed_addr #0 comdat align 2 {
  %4 = bitcast %"class.std::allocator"* %0 to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm(%"class.__gnu_cxx::new_allocator"* nonnull %4, i8* %1, i64 %2)
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local dereferenceable(1) %"class.std::allocator"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv(%"class.std::__cxx11::basic_string"*) local_unnamed_addr #12 align 2 {
  %2 = bitcast %"class.std::__cxx11::basic_string"* %0 to %"class.std::allocator"*
  ret %"class.std::allocator"* %2
}

declare dso_local void @__cxa_call_unexpected(i8*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm(%"class.__gnu_cxx::new_allocator"*, i8*, i64) local_unnamed_addr #12 comdat align 2 {
  tail call void @_ZdlPv(i8* %1) #18
  ret void
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #15

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm(%"class.std::__cxx11::basic_string"*, i64) local_unnamed_addr #12 align 2 {
  %3 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %0, i64 0, i32 1
  store i64 %1, i64* %3, align 8, !tbaa !26
  ret void
}

; Function Attrs: argmemonly nofree nounwind readonly
declare dso_local i64 @strlen(i8* nocapture) local_unnamed_addr #16

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcED2Ev(%"class.__gnu_cxx::new_allocator"*) unnamed_addr #12 comdat align 2 {
  ret void
}

declare dso_local void @_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::random_device"*, %"class.std::__cxx11::basic_string"* dereferenceable(32)) local_unnamed_addr #1

declare dso_local i32 @_ZNSt13random_device9_M_getvalEv(%"class.std::random_device"*) local_unnamed_addr #1

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE4seedEm(%"class.std::mersenne_twister_engine"*, i64) local_unnamed_addr #0 comdat align 2 {
  %3 = tail call i64 @_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_(i64 %1)
  %4 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 0
  store i64 %3, i64* %4, align 8, !tbaa !23
  br label %7

5:                                                ; preds = %7
  %6 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 1
  store i64 624, i64* %6, align 8, !tbaa !27
  ret void

7:                                                ; preds = %7, %2
  %8 = phi i64 [ 1, %2 ], [ %19, %7 ]
  %9 = add nsw i64 %8, -1
  %10 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %9
  %11 = load i64, i64* %10, align 8, !tbaa !23
  %12 = lshr i64 %11, 30
  %13 = xor i64 %12, %11
  %14 = mul i64 %13, 1812433253
  %15 = tail call i64 @_ZNSt8__detail5__modImLm624ELm1ELm0EEET_S1_(i64 %8)
  %16 = add i64 %14, %15
  %17 = tail call i64 @_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_(i64 %16)
  %18 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %8
  store i64 %17, i64* %18, align 8, !tbaa !23
  %19 = add nuw nsw i64 %8, 1
  %20 = icmp eq i64 %19, 624
  br i1 %20, label %5, label %7
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail5__modImLm4294967296ELm1ELm0EEET_S1_(i64) local_unnamed_addr #9 comdat {
  %2 = tail call i64 @_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm(i64 %0)
  ret i64 %2
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail5__modImLm624ELm1ELm0EEET_S1_(i64) local_unnamed_addr #9 comdat {
  %2 = tail call i64 @_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm(i64 %0)
  ret i64 %2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail4_ModImLm4294967296ELm1ELm0ELb1ELb1EE6__calcEm(i64) local_unnamed_addr #12 comdat align 2 {
  %2 = and i64 %0, 4294967295
  ret i64 %2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt8__detail4_ModImLm624ELm1ELm0ELb1ELb1EE6__calcEm(i64) local_unnamed_addr #12 comdat align 2 {
  %2 = urem i64 %0, 624
  ret i64 %2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt25uniform_real_distributionIfE10param_typeC2Eff(%"struct.std::uniform_real_distribution<float>::param_type"*, float, float) unnamed_addr #12 comdat align 2 {
  %4 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %0, i64 0, i32 0
  store float %1, float* %4, align 4, !tbaa !29
  %5 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %0, i64 0, i32 1
  store float %2, float* %5, align 4, !tbaa !31
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local float @_ZNSt25uniform_real_distributionIfEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEfRT_RKNS0_10param_typeE(%"class.std::uniform_real_distribution"*, %"class.std::mersenne_twister_engine"* dereferenceable(5000), %"struct.std::uniform_real_distribution<float>::param_type"* dereferenceable(8)) local_unnamed_addr #0 comdat align 2 {
  %4 = alloca %"struct.std::__detail::_Adaptor", align 8
  %5 = bitcast %"struct.std::__detail::_Adaptor"* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5) #18
  call void @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEC2ERS2_(%"struct.std::__detail::_Adaptor"* nonnull %4, %"class.std::mersenne_twister_engine"* nonnull dereferenceable(5000) %1)
  %6 = call float @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEclEv(%"struct.std::__detail::_Adaptor"* nonnull %4)
  %7 = call float @_ZNKSt25uniform_real_distributionIfE10param_type1bEv(%"struct.std::uniform_real_distribution<float>::param_type"* nonnull %2)
  %8 = call float @_ZNKSt25uniform_real_distributionIfE10param_type1aEv(%"struct.std::uniform_real_distribution<float>::param_type"* nonnull %2)
  %9 = fsub contract float %7, %8
  %10 = fmul contract float %6, %9
  %11 = call float @_ZNKSt25uniform_real_distributionIfE10param_type1aEv(%"struct.std::uniform_real_distribution<float>::param_type"* nonnull %2)
  %12 = fadd contract float %11, %10
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5) #18
  ret float %12
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEC2ERS2_(%"struct.std::__detail::_Adaptor"*, %"class.std::mersenne_twister_engine"* dereferenceable(5000)) unnamed_addr #12 comdat align 2 {
  %3 = getelementptr inbounds %"struct.std::__detail::_Adaptor", %"struct.std::__detail::_Adaptor"* %0, i64 0, i32 0
  store %"class.std::mersenne_twister_engine"* %1, %"class.std::mersenne_twister_engine"** %3, align 8, !tbaa !7
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local float @_ZNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEclEv(%"struct.std::__detail::_Adaptor"*) local_unnamed_addr #0 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::__detail::_Adaptor", %"struct.std::__detail::_Adaptor"* %0, i64 0, i32 0
  %3 = load %"class.std::mersenne_twister_engine"*, %"class.std::mersenne_twister_engine"** %2, align 8, !tbaa !32
  %4 = tail call float @_ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_(%"class.std::mersenne_twister_engine"* dereferenceable(5000) %3)
  ret float %4
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float @_ZNKSt25uniform_real_distributionIfE10param_type1bEv(%"struct.std::uniform_real_distribution<float>::param_type"*) local_unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %0, i64 0, i32 1
  %3 = load float, float* %2, align 4, !tbaa !31
  ret float %3
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float @_ZNKSt25uniform_real_distributionIfE10param_type1aEv(%"struct.std::uniform_real_distribution<float>::param_type"*) local_unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::uniform_real_distribution<float>::param_type", %"struct.std::uniform_real_distribution<float>::param_type"* %0, i64 0, i32 0
  %3 = load float, float* %2, align 4, !tbaa !29
  ret float %3
}

; Function Attrs: uwtable
define linkonce_odr dso_local float @_ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_(%"class.std::mersenne_twister_engine"* dereferenceable(5000)) local_unnamed_addr #0 comdat {
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = bitcast i64* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6) #18
  store i64 24, i64* %2, align 8, !tbaa !23
  %7 = bitcast i64* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #18
  store i64 24, i64* %3, align 8, !tbaa !23
  %8 = call dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* nonnull dereferenceable(8) %2, i64* nonnull dereferenceable(8) %3)
  %9 = load i64, i64* %8, align 8, !tbaa !23
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6) #18
  %10 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv()
  %11 = uitofp i64 %10 to x86_fp80
  %12 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv()
  %13 = uitofp i64 %12 to x86_fp80
  %14 = fsub contract x86_fp80 %11, %13
  %15 = fadd contract x86_fp80 %14, 0xK3FFF8000000000000000
  %16 = call x86_fp80 @_ZSt3loge(x86_fp80 %15)
  %17 = call x86_fp80 @_ZSt3loge(x86_fp80 0xK40008000000000000000)
  %18 = fdiv x86_fp80 %16, %17
  %19 = fptoui x86_fp80 %18 to i64
  %20 = bitcast i64* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %20) #18
  store i64 1, i64* %4, align 8, !tbaa !23
  %21 = bitcast i64* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %21) #18
  %22 = add i64 %9, -1
  %23 = add i64 %22, %19
  %24 = udiv i64 %23, %19
  store i64 %24, i64* %5, align 8, !tbaa !23
  %25 = call dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* nonnull dereferenceable(8) %4, i64* nonnull dereferenceable(8) %5)
  %26 = load i64, i64* %25, align 8, !tbaa !23
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %21) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %20) #18
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %28, label %33

28:                                               ; preds = %33, %1
  %29 = phi float [ 0.000000e+00, %1 ], [ %42, %33 ]
  %30 = phi float [ 1.000000e+00, %1 ], [ %45, %33 ]
  %31 = fdiv float %29, %30
  %32 = fcmp ult float %31, 1.000000e+00
  br i1 %32, label %50, label %48, !prof !34

33:                                               ; preds = %1, %33
  %34 = phi i64 [ %46, %33 ], [ %26, %1 ]
  %35 = phi float [ %45, %33 ], [ 1.000000e+00, %1 ]
  %36 = phi float [ %42, %33 ], [ 0.000000e+00, %1 ]
  %37 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"* nonnull %0)
  %38 = call i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv()
  %39 = sub i64 %37, %38
  %40 = uitofp i64 %39 to float
  %41 = fmul contract float %35, %40
  %42 = fadd contract float %36, %41
  %43 = fpext float %35 to x86_fp80
  %44 = fmul contract x86_fp80 %15, %43
  %45 = fptrunc x86_fp80 %44 to float
  %46 = add i64 %34, -1
  %47 = icmp eq i64 %46, 0
  br i1 %47, label %28, label %33

48:                                               ; preds = %28
  %49 = call float @_ZSt9nextafterff(float 1.000000e+00, float 0.000000e+00)
  br label %50

50:                                               ; preds = %28, %48
  %51 = phi float [ %49, %48 ], [ %31, %28 ]
  ret float %51
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* dereferenceable(8), i64* dereferenceable(8)) local_unnamed_addr #10 comdat {
  %3 = load i64, i64* %1, align 8, !tbaa !23
  %4 = load i64, i64* %0, align 8, !tbaa !23
  %5 = icmp ult i64 %3, %4
  %6 = select i1 %5, i64* %1, i64* %0
  ret i64* %6
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3maxEv() local_unnamed_addr #12 comdat align 2 {
  ret i64 4294967295
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE3minEv() local_unnamed_addr #12 comdat align 2 {
  ret i64 0
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local x86_fp80 @_ZSt3loge(x86_fp80) local_unnamed_addr #10 comdat {
  %2 = tail call x86_fp80 @logl(x86_fp80 %0) #18
  ret x86_fp80 %2
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* dereferenceable(8), i64* dereferenceable(8)) local_unnamed_addr #10 comdat {
  %3 = load i64, i64* %0, align 8, !tbaa !23
  %4 = load i64, i64* %1, align 8, !tbaa !23
  %5 = icmp ult i64 %3, %4
  %6 = select i1 %5, i64* %1, i64* %0
  ret i64* %6
}

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(%"class.std::mersenne_twister_engine"*) local_unnamed_addr #0 comdat align 2 {
  %2 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 1
  %3 = load i64, i64* %2, align 8, !tbaa !27
  %4 = icmp ugt i64 %3, 623
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  tail call void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv(%"class.std::mersenne_twister_engine"* nonnull %0)
  br label %6

6:                                                ; preds = %5, %1
  %7 = load i64, i64* %2, align 8, !tbaa !27
  %8 = add i64 %7, 1
  store i64 %8, i64* %2, align 8, !tbaa !27
  %9 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %7
  %10 = load i64, i64* %9, align 8, !tbaa !23
  %11 = lshr i64 %10, 11
  %12 = and i64 %11, 4294967295
  %13 = xor i64 %12, %10
  %14 = shl i64 %13, 7
  %15 = and i64 %14, 2636928640
  %16 = xor i64 %15, %13
  %17 = shl i64 %16, 15
  %18 = and i64 %17, 4022730752
  %19 = xor i64 %18, %16
  %20 = lshr i64 %19, 18
  %21 = xor i64 %20, %19
  ret i64 %21
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float @_ZSt9nextafterff(float, float) local_unnamed_addr #12 comdat {
  %3 = tail call float @nextafterf(float %0, float %1) #18
  ret float %3
}

; Function Attrs: nofree nounwind
declare dso_local x86_fp80 @logl(x86_fp80) local_unnamed_addr #8

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv(%"class.std::mersenne_twister_engine"*) local_unnamed_addr #12 comdat align 2 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %7, %2 ]
  %4 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %3
  %5 = load i64, i64* %4, align 8, !tbaa !23
  %6 = and i64 %5, -2147483648
  %7 = add nuw nsw i64 %3, 1
  %8 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %7
  %9 = load i64, i64* %8, align 8, !tbaa !23
  %10 = and i64 %9, 2147483646
  %11 = or i64 %10, %6
  %12 = add nuw nsw i64 %3, 397
  %13 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %12
  %14 = load i64, i64* %13, align 8, !tbaa !23
  %15 = lshr exact i64 %11, 1
  %16 = xor i64 %15, %14
  %17 = and i64 %9, 1
  %18 = icmp eq i64 %17, 0
  %19 = select i1 %18, i64 0, i64 2567483615
  %20 = xor i64 %16, %19
  store i64 %20, i64* %4, align 8, !tbaa !23
  %21 = icmp eq i64 %7, 227
  br i1 %21, label %39, label %2

22:                                               ; preds = %39
  %23 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 623
  %24 = load i64, i64* %23, align 8, !tbaa !23
  %25 = and i64 %24, -2147483648
  %26 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 0
  %27 = load i64, i64* %26, align 8, !tbaa !23
  %28 = and i64 %27, 2147483646
  %29 = or i64 %28, %25
  %30 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 396
  %31 = load i64, i64* %30, align 8, !tbaa !23
  %32 = lshr exact i64 %29, 1
  %33 = xor i64 %32, %31
  %34 = and i64 %27, 1
  %35 = icmp eq i64 %34, 0
  %36 = select i1 %35, i64 0, i64 2567483615
  %37 = xor i64 %33, %36
  store i64 %37, i64* %23, align 8, !tbaa !23
  %38 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 1
  store i64 0, i64* %38, align 8, !tbaa !27
  ret void

39:                                               ; preds = %2, %39
  %40 = phi i64 [ %44, %39 ], [ 227, %2 ]
  %41 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %40
  %42 = load i64, i64* %41, align 8, !tbaa !23
  %43 = and i64 %42, -2147483648
  %44 = add nuw nsw i64 %40, 1
  %45 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %44
  %46 = load i64, i64* %45, align 8, !tbaa !23
  %47 = and i64 %46, 2147483646
  %48 = or i64 %47, %43
  %49 = add nsw i64 %40, -227
  %50 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %0, i64 0, i32 0, i64 %49
  %51 = load i64, i64* %50, align 8, !tbaa !23
  %52 = lshr exact i64 %48, 1
  %53 = xor i64 %52, %51
  %54 = and i64 %46, 1
  %55 = icmp eq i64 %54, 0
  %56 = select i1 %55, i64 0, i64 2567483615
  %57 = xor i64 %53, %56
  store i64 %57, i64* %41, align 8, !tbaa !23
  %58 = icmp eq i64 %44, 623
  br i1 %58, label %22, label %39
}

; Function Attrs: nounwind
declare dso_local float @nextafterf(float, float) local_unnamed_addr #2

declare dso_local void @_ZNSt13random_device7_M_finiEv(%"class.std::random_device"*) local_unnamed_addr #1

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000EEEES2_ILl1ELl1000000EElLb1ELb0EE6__castIlS2_ILl1ELl1000000000EEEES4_RKNS1_IT_T0_EE(%"struct.std::chrono::duration"* dereferenceable(8)) local_unnamed_addr #0 comdat align 2 {
  %2 = alloca %"struct.std::chrono::duration.0", align 8
  %3 = alloca i64, align 8
  %4 = bitcast i64* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %4) #18
  %5 = tail call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"* nonnull %0)
  %6 = sdiv i64 %5, 1000000
  store i64 %6, i64* %3, align 8, !tbaa !23
  call void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000EEEC2IlvEERKT_(%"struct.std::chrono::duration.0"* nonnull %2, i64* nonnull dereferenceable(8) %3)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %4) #18
  %7 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %2, i64 0, i32 0
  %8 = load i64, i64* %7, align 8
  ret i64 %8
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"*) local_unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %0, i64 0, i32 0
  %3 = load i64, i64* %2, align 8, !tbaa !35
  ret i64 %3
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000EEEC2IlvEERKT_(%"struct.std::chrono::duration.0"*, i64* dereferenceable(8)) unnamed_addr #12 comdat align 2 {
  %3 = getelementptr inbounds %"struct.std::chrono::duration.0", %"struct.std::chrono::duration.0"* %0, i64 0, i32 0
  %4 = load i64, i64* %1, align 8, !tbaa !23
  store i64 %4, i64* %3, align 8, !tbaa !13
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_(%"struct.std::chrono::duration"* dereferenceable(8), %"struct.std::chrono::duration"* dereferenceable(8)) local_unnamed_addr #0 comdat {
  %3 = alloca %"struct.std::chrono::duration", align 8
  %4 = alloca i64, align 8
  %5 = alloca %"struct.std::chrono::duration", align 8
  %6 = alloca %"struct.std::chrono::duration", align 8
  %7 = bitcast i64* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #18
  %8 = bitcast %"struct.std::chrono::duration"* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %8) #18
  %9 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %0, i64 0, i32 0
  %10 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %5, i64 0, i32 0
  %11 = load i64, i64* %9, align 8, !tbaa !23
  store i64 %11, i64* %10, align 8, !tbaa !23
  %12 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"* nonnull %5)
  %13 = bitcast %"struct.std::chrono::duration"* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %13) #18
  %14 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %1, i64 0, i32 0
  %15 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %6, i64 0, i32 0
  %16 = load i64, i64* %14, align 8, !tbaa !23
  store i64 %16, i64* %15, align 8, !tbaa !23
  %17 = call i64 @_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv(%"struct.std::chrono::duration"* nonnull %6)
  %18 = sub nsw i64 %12, %17
  store i64 %18, i64* %4, align 8, !tbaa !23
  call void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC2IlvEERKT_(%"struct.std::chrono::duration"* nonnull %3, i64* nonnull dereferenceable(8) %4)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %13) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %8) #18
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #18
  %19 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %3, i64 0, i32 0
  %20 = load i64, i64* %19, align 8
  ret i64 %20
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv(%"struct.std::chrono::time_point"*) local_unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %"struct.std::chrono::time_point", %"struct.std::chrono::time_point"* %0, i64 0, i32 0, i32 0
  %3 = load i64, i64* %2, align 8
  ret i64 %3
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC2IlvEERKT_(%"struct.std::chrono::duration"*, i64* dereferenceable(8)) unnamed_addr #12 comdat align 2 {
  %3 = getelementptr inbounds %"struct.std::chrono::duration", %"struct.std::chrono::duration"* %0, i64 0, i32 0
  %4 = load i64, i64* %1, align 8, !tbaa !23
  store i64 %4, i64* %3, align 8, !tbaa !35
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate(%"class.std::basic_ios"*, i32) local_unnamed_addr #0 align 2 {
  %3 = tail call i32 @_ZNKSt9basic_iosIcSt11char_traitsIcEE7rdstateEv(%"class.std::basic_ios"* %0)
  %4 = tail call i32 @_ZStorSt12_Ios_IostateS_(i32 %3, i32 %1)
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"class.std::basic_ios"* %0, i32 %4)
  ret void
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #1

declare dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"class.std::basic_ios"*, i32) local_unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i32 @_ZStorSt12_Ios_IostateS_(i32, i32) local_unnamed_addr #10 comdat {
  %3 = or i32 %1, %0
  ret i32 %3
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i32 @_ZNKSt9basic_iosIcSt11char_traitsIcEE7rdstateEv(%"class.std::basic_ios"*) local_unnamed_addr #12 align 2 {
  %2 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %0, i64 0, i32 0, i32 5
  %3 = load i32, i32* %2, align 8, !tbaa !37
  ret i32 %3
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt5flushIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) local_unnamed_addr #9 {
  %2 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %0)
  ret %"class.std::basic_ostream"* %2
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #1

; Function Attrs: uwtable
define available_externally dso_local signext i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc(%"class.std::basic_ios"*, i8 signext) local_unnamed_addr #0 align 2 {
  %3 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %0, i64 0, i32 5
  %4 = load %"class.std::ctype"*, %"class.std::ctype"** %3, align 8, !tbaa !43
  %5 = tail call dereferenceable(576) %"class.std::ctype"* @_ZSt13__check_facetISt5ctypeIcEERKT_PS3_(%"class.std::ctype"* %4)
  %6 = tail call signext i8 @_ZNKSt5ctypeIcE5widenEc(%"class.std::ctype"* nonnull %5, i8 signext %1)
  ret i8 %6
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local dereferenceable(576) %"class.std::ctype"* @_ZSt13__check_facetISt5ctypeIcEERKT_PS3_(%"class.std::ctype"*) local_unnamed_addr #9 comdat {
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
  %4 = load i8, i8* %3, align 8, !tbaa !46
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %10, label %6

6:                                                ; preds = %2
  %7 = zext i8 %1 to i64
  %8 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %0, i64 0, i32 9, i64 %7
  %9 = load i8, i8* %8, align 1, !tbaa !20
  br label %16

10:                                               ; preds = %2
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %0)
  %11 = bitcast %"class.std::ctype"* %0 to i8 (%"class.std::ctype"*, i8)***
  %12 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %11, align 8, !tbaa !11
  %13 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %12, i64 6
  %14 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %13, align 8
  %15 = tail call signext i8 %14(%"class.std::ctype"* nonnull %0, i8 signext %1)
  br label %16

16:                                               ; preds = %10, %6
  %17 = phi i8 [ %9, %6 ], [ %15, %10 ]
  ret i8 %17
}

; Function Attrs: noreturn
declare dso_local void @_ZSt16__throw_bad_castv() local_unnamed_addr #13

declare dso_local void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #1

declare dso_local i32 @cudaMalloc(i8**, i64) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIlEERSoT_(%"class.std::basic_ostream"*, i64) local_unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZSt3absf(float) local_unnamed_addr #10 comdat {
  %2 = tail call float @llvm.fabs.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #17

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIbEERSoT_(%"class.std::basic_ostream"*, i1 zeroext) local_unnamed_addr #1

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_dot_product.cu() #0 section ".text.startup" {
  tail call fastcc void @__cxx_global_var_init()
  ret void
}

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind }
attributes #4 = { nofree norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { inlinehint norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #14 = { noinline noreturn nounwind }
attributes #15 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #16 = { argmemonly nofree nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #17 = { nounwind readnone speculatable }
attributes #18 = { nounwind }
attributes #19 = { nounwind readonly }
attributes #20 = { noreturn nounwind }
attributes #21 = { noreturn }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"float", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"vtable pointer", !6, i64 0}
!13 = !{!14, !15, i64 0}
!14 = !{!"_ZTSNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE", !15, i64 0}
!15 = !{!"long", !5, i64 0}
!16 = !{!17, !10, i64 0}
!17 = !{!"_ZTS4dim3", !10, i64 0, !10, i64 4, !10, i64 8}
!18 = !{!17, !10, i64 4}
!19 = !{!17, !10, i64 8}
!20 = !{!5, !5, i64 0}
!21 = !{!22, !8, i64 0}
!22 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !8, i64 0}
!23 = !{!15, !15, i64 0}
!24 = !{!25, !8, i64 0}
!25 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !22, i64 0, !15, i64 8, !5, i64 16}
!26 = !{!25, !15, i64 8}
!27 = !{!28, !15, i64 4992}
!28 = !{!"_ZTSSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE", !5, i64 0, !15, i64 4992}
!29 = !{!30, !4, i64 0}
!30 = !{!"_ZTSNSt25uniform_real_distributionIfE10param_typeE", !4, i64 0, !4, i64 4}
!31 = !{!30, !4, i64 4}
!32 = !{!33, !8, i64 0}
!33 = !{!"_ZTSNSt8__detail8_AdaptorISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEfEE", !8, i64 0}
!34 = !{!"branch_weights", i32 2000, i32 1}
!35 = !{!36, !15, i64 0}
!36 = !{!"_ZTSNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEE", !15, i64 0}
!37 = !{!38, !40, i64 32}
!38 = !{!"_ZTSSt8ios_base", !15, i64 8, !15, i64 16, !39, i64 24, !40, i64 28, !40, i64 32, !8, i64 40, !41, i64 48, !5, i64 64, !10, i64 192, !8, i64 200, !42, i64 208}
!39 = !{!"_ZTSSt13_Ios_Fmtflags", !5, i64 0}
!40 = !{!"_ZTSSt12_Ios_Iostate", !5, i64 0}
!41 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !15, i64 8}
!42 = !{!"_ZTSSt6locale", !8, i64 0}
!43 = !{!44, !8, i64 240}
!44 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !8, i64 216, !5, i64 224, !45, i64 225, !8, i64 232, !8, i64 240, !8, i64 248, !8, i64 256}
!45 = !{!"bool", !5, i64 0}
!46 = !{!47, !5, i64 56}
!47 = !{!"_ZTSSt5ctypeIcE", !8, i64 16, !45, i64 24, !8, i64 32, !8, i64 40, !8, i64 48, !5, i64 56, !5, i64 57, !5, i64 313, !5, i64 569}
