; ModuleID = 'MonteCarloMultiGPU.cpp'
source_filename = "MonteCarloMultiGPU.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%class.StopWatchInterface = type { i32 (...)** }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.cudaDeviceProp = type { [256 x i8], %struct.CUuuid_st, [8 x i8], i32, i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32 }
%struct.CUuuid_st = type { [16 x i8] }
%struct.TOptionData = type { float, float, float, float, float }
%struct.TOptionValue = type { float, float }
%struct.TOptionPlan = type { i32, i32, %struct.TOptionData*, %struct.TOptionValue*, %struct.__TOptionValue*, i8*, i8*, i8*, i8*, %struct.curandStateXORWOW*, i32, float, i32 }
%struct.__TOptionValue = type { float, float }
%struct.curandStateXORWOW = type { i32, [5 x i32], i32, i32, float, double }
%class.StopWatchLinux = type { %class.StopWatchInterface, %struct.timeval, float, float, i8, i32 }
%struct.timeval = type { i64, i64 }
%struct.CUstream_st = type opaque
%struct.CUevent_st = type opaque
%struct.timezone = type { i32, i32 }

$_Z5checkI9cudaErrorEvT_PKcS3_i = comdat any

$_Z14sdkCreateTimerPP18StopWatchInterface = comdat any

$_Z13sdkResetTimerPP18StopWatchInterface = comdat any

$_Z16sdkGetTimerValuePP18StopWatchInterface = comdat any

$_ZSt4fabsf = comdat any

$_Z13sdkStartTimerPP18StopWatchInterface = comdat any

$_ZN14StopWatchLinuxC2Ev = comdat any

$_ZN18StopWatchInterfaceC2Ev = comdat any

$_ZN14StopWatchLinuxD0Ev = comdat any

$_ZN14StopWatchLinux5startEv = comdat any

$_ZN14StopWatchLinux4stopEv = comdat any

$_ZN14StopWatchLinux5resetEv = comdat any

$_ZN14StopWatchLinux7getTimeEv = comdat any

$_ZN14StopWatchLinux14getAverageTimeEv = comdat any

$_ZN18StopWatchInterfaceD2Ev = comdat any

$_ZN18StopWatchInterfaceD0Ev = comdat any

$_ZN14StopWatchLinux11getDiffTimeEv = comdat any

$_Z12sdkStopTimerPP18StopWatchInterface = comdat any

$_ZTV14StopWatchLinux = comdat any

$_ZTS14StopWatchLinux = comdat any

$_ZTS18StopWatchInterface = comdat any

$_ZTI18StopWatchInterface = comdat any

$_ZTI14StopWatchLinux = comdat any

$_ZTV18StopWatchInterface = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@pArgc = dso_local local_unnamed_addr global i32* null, align 8
@pArgv = dso_local local_unnamed_addr global i8** null, align 8
@hTimer = dso_local local_unnamed_addr global %class.StopWatchInterface** null, align 8
@.str = private unnamed_addr constant [17 x i8] c"%s Starting...\0A\0A\00", align 1
@.str.1 = private unnamed_addr constant [27 x i8] c"cudaGetDeviceCount(&GPU_N)\00", align 1
@.str.2 = private unnamed_addr constant [23 x i8] c"MonteCarloMultiGPU.cpp\00", align 1
@.str.5 = private unnamed_addr constant [30 x i8] c"Number of GPUs          = %d\0A\00", align 1
@.str.6 = private unnamed_addr constant [30 x i8] c"Total number of options = %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [30 x i8] c"Number of paths         = %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [37 x i8] c"main(): starting %i host threads...\0A\00", align 1
@.str.11 = private unnamed_addr constant [61 x i8] c"cudaGetDeviceProperties(&deviceProp, optionSolver[i].device)\00", align 1
@.str.12 = private unnamed_addr constant [20 x i8] c"GPU Device #%i: %s\0A\00", align 1
@.str.13 = private unnamed_addr constant [22 x i8] c"Options         : %i\0A\00", align 1
@.str.14 = private unnamed_addr constant [22 x i8] c"Simulation paths: %i\0A\00", align 1
@.str.15 = private unnamed_addr constant [23 x i8] c"\0ATotal time (ms.): %f\0A\00", align 1
@.str.17 = private unnamed_addr constant [22 x i8] c"Options per sec.: %f\0A\00", align 1
@.str.20 = private unnamed_addr constant [13 x i8] c"L1 norm: %E\0A\00", align 1
@.str.22 = private unnamed_addr constant [17 x i8] c"cudaSetDevice(i)\00", align 1
@.str.24 = private unnamed_addr constant [21 x i8] c"L1 norm        : %E\0A\00", align 1
@.str.25 = private unnamed_addr constant [21 x i8] c"Average reserve: %f\0A\00", align 1
@.str.26 = private unnamed_addr constant [13 x i8] c"Test passed\0A\00", align 1
@.str.27 = private unnamed_addr constant [14 x i8] c"Test failed!\0A\00", align 1
@_ZTV14StopWatchLinux = linkonce_odr dso_local unnamed_addr constant { [9 x i8*] } { [9 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI14StopWatchLinux to i8*), i8* bitcast (void (%class.StopWatchInterface*)* @_ZN18StopWatchInterfaceD2Ev to i8*), i8* bitcast (void (%class.StopWatchLinux*)* @_ZN14StopWatchLinuxD0Ev to i8*), i8* bitcast (void (%class.StopWatchLinux*)* @_ZN14StopWatchLinux5startEv to i8*), i8* bitcast (void (%class.StopWatchLinux*)* @_ZN14StopWatchLinux4stopEv to i8*), i8* bitcast (void (%class.StopWatchLinux*)* @_ZN14StopWatchLinux5resetEv to i8*), i8* bitcast (float (%class.StopWatchLinux*)* @_ZN14StopWatchLinux7getTimeEv to i8*), i8* bitcast (float (%class.StopWatchLinux*)* @_ZN14StopWatchLinux14getAverageTimeEv to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global i8*
@_ZTS14StopWatchLinux = linkonce_odr dso_local constant [17 x i8] c"14StopWatchLinux\00", comdat, align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS18StopWatchInterface = linkonce_odr dso_local constant [21 x i8] c"18StopWatchInterface\00", comdat, align 1
@_ZTI18StopWatchInterface = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([21 x i8], [21 x i8]* @_ZTS18StopWatchInterface, i32 0, i32 0) }, comdat, align 8
@_ZTI14StopWatchLinux = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @_ZTS14StopWatchLinux, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI18StopWatchInterface to i8*) }, comdat, align 8
@_ZTV18StopWatchInterface = linkonce_odr dso_local unnamed_addr constant { [9 x i8*] } { [9 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI18StopWatchInterface to i8*), i8* bitcast (void (%class.StopWatchInterface*)* @_ZN18StopWatchInterfaceD2Ev to i8*), i8* bitcast (void (%class.StopWatchInterface*)* @_ZN18StopWatchInterfaceD0Ev to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, comdat, align 8
@.str.28 = private unnamed_addr constant [30 x i8] c"cudaSetDevice(plan[i].device)\00", align 1
@.str.29 = private unnamed_addr constant [32 x i8] c"cudaStreamCreate(&(streams[i]))\00", align 1
@.str.30 = private unnamed_addr constant [30 x i8] c"cudaEventCreate(&(events[i]))\00", align 1
@.str.31 = private unnamed_addr constant [53 x i8] c"cudaGetDeviceProperties(&deviceProp, plan[i].device)\00", align 1
@.str.32 = private unnamed_addr constant [24 x i8] c"cudaDeviceSynchronize()\00", align 1
@.str.33 = private unnamed_addr constant [39 x i8] c"cudaEventRecord(events[i], streams[i])\00", align 1
@.str.34 = private unnamed_addr constant [30 x i8] c"cudaStreamDestroy(streams[i])\00", align 1
@.str.35 = private unnamed_addr constant [28 x i8] c"cudaEventDestroy(events[i])\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.36 = private unnamed_addr constant [39 x i8] c"CUDA error at %s:%d code=%d(%s) \22%s\22 \0A\00", align 1
@.str.37 = private unnamed_addr constant [12 x i8] c"cudaSuccess\00", align 1
@.str.38 = private unnamed_addr constant [30 x i8] c"cudaErrorMissingConfiguration\00", align 1
@.str.39 = private unnamed_addr constant [26 x i8] c"cudaErrorMemoryAllocation\00", align 1
@.str.40 = private unnamed_addr constant [29 x i8] c"cudaErrorInitializationError\00", align 1
@.str.41 = private unnamed_addr constant [23 x i8] c"cudaErrorLaunchFailure\00", align 1
@.str.42 = private unnamed_addr constant [28 x i8] c"cudaErrorPriorLaunchFailure\00", align 1
@.str.43 = private unnamed_addr constant [23 x i8] c"cudaErrorLaunchTimeout\00", align 1
@.str.44 = private unnamed_addr constant [30 x i8] c"cudaErrorLaunchOutOfResources\00", align 1
@.str.45 = private unnamed_addr constant [31 x i8] c"cudaErrorInvalidDeviceFunction\00", align 1
@.str.46 = private unnamed_addr constant [30 x i8] c"cudaErrorInvalidConfiguration\00", align 1
@.str.47 = private unnamed_addr constant [23 x i8] c"cudaErrorInvalidDevice\00", align 1
@.str.48 = private unnamed_addr constant [22 x i8] c"cudaErrorInvalidValue\00", align 1
@.str.49 = private unnamed_addr constant [27 x i8] c"cudaErrorInvalidPitchValue\00", align 1
@.str.50 = private unnamed_addr constant [23 x i8] c"cudaErrorInvalidSymbol\00", align 1
@.str.51 = private unnamed_addr constant [31 x i8] c"cudaErrorMapBufferObjectFailed\00", align 1
@.str.52 = private unnamed_addr constant [33 x i8] c"cudaErrorUnmapBufferObjectFailed\00", align 1
@.str.53 = private unnamed_addr constant [28 x i8] c"cudaErrorInvalidHostPointer\00", align 1
@.str.54 = private unnamed_addr constant [30 x i8] c"cudaErrorInvalidDevicePointer\00", align 1
@.str.55 = private unnamed_addr constant [24 x i8] c"cudaErrorInvalidTexture\00", align 1
@.str.56 = private unnamed_addr constant [31 x i8] c"cudaErrorInvalidTextureBinding\00", align 1
@.str.57 = private unnamed_addr constant [34 x i8] c"cudaErrorInvalidChannelDescriptor\00", align 1
@.str.58 = private unnamed_addr constant [32 x i8] c"cudaErrorInvalidMemcpyDirection\00", align 1
@.str.59 = private unnamed_addr constant [27 x i8] c"cudaErrorAddressOfConstant\00", align 1
@.str.60 = private unnamed_addr constant [28 x i8] c"cudaErrorTextureFetchFailed\00", align 1
@.str.61 = private unnamed_addr constant [25 x i8] c"cudaErrorTextureNotBound\00", align 1
@.str.62 = private unnamed_addr constant [30 x i8] c"cudaErrorSynchronizationError\00", align 1
@.str.63 = private unnamed_addr constant [30 x i8] c"cudaErrorInvalidFilterSetting\00", align 1
@.str.64 = private unnamed_addr constant [28 x i8] c"cudaErrorInvalidNormSetting\00", align 1
@.str.65 = private unnamed_addr constant [30 x i8] c"cudaErrorMixedDeviceExecution\00", align 1
@.str.66 = private unnamed_addr constant [25 x i8] c"cudaErrorCudartUnloading\00", align 1
@.str.67 = private unnamed_addr constant [17 x i8] c"cudaErrorUnknown\00", align 1
@.str.68 = private unnamed_addr constant [27 x i8] c"cudaErrorNotYetImplemented\00", align 1
@.str.69 = private unnamed_addr constant [29 x i8] c"cudaErrorMemoryValueTooLarge\00", align 1
@.str.70 = private unnamed_addr constant [31 x i8] c"cudaErrorInvalidResourceHandle\00", align 1
@.str.71 = private unnamed_addr constant [18 x i8] c"cudaErrorNotReady\00", align 1
@.str.72 = private unnamed_addr constant [28 x i8] c"cudaErrorInsufficientDriver\00", align 1
@.str.73 = private unnamed_addr constant [28 x i8] c"cudaErrorSetOnActiveProcess\00", align 1
@.str.74 = private unnamed_addr constant [24 x i8] c"cudaErrorInvalidSurface\00", align 1
@.str.75 = private unnamed_addr constant [18 x i8] c"cudaErrorNoDevice\00", align 1
@.str.76 = private unnamed_addr constant [26 x i8] c"cudaErrorECCUncorrectable\00", align 1
@.str.77 = private unnamed_addr constant [36 x i8] c"cudaErrorSharedObjectSymbolNotFound\00", align 1
@.str.78 = private unnamed_addr constant [32 x i8] c"cudaErrorSharedObjectInitFailed\00", align 1
@.str.79 = private unnamed_addr constant [26 x i8] c"cudaErrorUnsupportedLimit\00", align 1
@.str.80 = private unnamed_addr constant [31 x i8] c"cudaErrorDuplicateVariableName\00", align 1
@.str.81 = private unnamed_addr constant [30 x i8] c"cudaErrorDuplicateTextureName\00", align 1
@.str.82 = private unnamed_addr constant [30 x i8] c"cudaErrorDuplicateSurfaceName\00", align 1
@.str.83 = private unnamed_addr constant [28 x i8] c"cudaErrorDevicesUnavailable\00", align 1
@.str.84 = private unnamed_addr constant [28 x i8] c"cudaErrorInvalidKernelImage\00", align 1
@.str.85 = private unnamed_addr constant [32 x i8] c"cudaErrorNoKernelImageForDevice\00", align 1
@.str.86 = private unnamed_addr constant [35 x i8] c"cudaErrorIncompatibleDriverContext\00", align 1
@.str.87 = private unnamed_addr constant [34 x i8] c"cudaErrorPeerAccessAlreadyEnabled\00", align 1
@.str.88 = private unnamed_addr constant [30 x i8] c"cudaErrorPeerAccessNotEnabled\00", align 1
@.str.89 = private unnamed_addr constant [28 x i8] c"cudaErrorDeviceAlreadyInUse\00", align 1
@.str.90 = private unnamed_addr constant [26 x i8] c"cudaErrorProfilerDisabled\00", align 1
@.str.91 = private unnamed_addr constant [32 x i8] c"cudaErrorProfilerNotInitialized\00", align 1
@.str.92 = private unnamed_addr constant [32 x i8] c"cudaErrorProfilerAlreadyStarted\00", align 1
@.str.93 = private unnamed_addr constant [32 x i8] c"cudaErrorProfilerAlreadyStopped\00", align 1
@.str.94 = private unnamed_addr constant [16 x i8] c"cudaErrorAssert\00", align 1
@.str.95 = private unnamed_addr constant [22 x i8] c"cudaErrorTooManyPeers\00", align 1
@.str.96 = private unnamed_addr constant [37 x i8] c"cudaErrorHostMemoryAlreadyRegistered\00", align 1
@.str.97 = private unnamed_addr constant [33 x i8] c"cudaErrorHostMemoryNotRegistered\00", align 1
@.str.98 = private unnamed_addr constant [25 x i8] c"cudaErrorOperatingSystem\00", align 1
@.str.99 = private unnamed_addr constant [31 x i8] c"cudaErrorPeerAccessUnsupported\00", align 1
@.str.100 = private unnamed_addr constant [32 x i8] c"cudaErrorLaunchMaxDepthExceeded\00", align 1
@.str.101 = private unnamed_addr constant [29 x i8] c"cudaErrorLaunchFileScopedTex\00", align 1
@.str.102 = private unnamed_addr constant [30 x i8] c"cudaErrorLaunchFileScopedSurf\00", align 1
@.str.103 = private unnamed_addr constant [27 x i8] c"cudaErrorSyncDepthExceeded\00", align 1
@.str.104 = private unnamed_addr constant [36 x i8] c"cudaErrorLaunchPendingCountExceeded\00", align 1
@.str.105 = private unnamed_addr constant [22 x i8] c"cudaErrorNotPermitted\00", align 1
@.str.106 = private unnamed_addr constant [22 x i8] c"cudaErrorNotSupported\00", align 1
@.str.107 = private unnamed_addr constant [28 x i8] c"cudaErrorHardwareStackError\00", align 1
@.str.108 = private unnamed_addr constant [28 x i8] c"cudaErrorIllegalInstruction\00", align 1
@.str.109 = private unnamed_addr constant [27 x i8] c"cudaErrorMisalignedAddress\00", align 1
@.str.110 = private unnamed_addr constant [29 x i8] c"cudaErrorInvalidAddressSpace\00", align 1
@.str.111 = private unnamed_addr constant [19 x i8] c"cudaErrorInvalidPc\00", align 1
@.str.112 = private unnamed_addr constant [24 x i8] c"cudaErrorIllegalAddress\00", align 1
@.str.113 = private unnamed_addr constant [20 x i8] c"cudaErrorInvalidPtx\00", align 1
@.str.114 = private unnamed_addr constant [32 x i8] c"cudaErrorInvalidGraphicsContext\00", align 1
@.str.115 = private unnamed_addr constant [24 x i8] c"cudaErrorStartupFailure\00", align 1
@.str.116 = private unnamed_addr constant [24 x i8] c"cudaErrorApiFailureBase\00", align 1
@.str.117 = private unnamed_addr constant [29 x i8] c"cudaErrorNvlinkUncorrectable\00", align 1
@.str.118 = private unnamed_addr constant [10 x i8] c"<unknown>\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_MonteCarloMultiGPU.cpp, i8* null }]
@str = private unnamed_addr constant [19 x i8] c"MonteCarloMultiGPU\00", align 1
@str.119 = private unnamed_addr constant [19 x i8] c"==================\00", align 1
@str.120 = private unnamed_addr constant [33 x i8] c"main(): generating input data...\00", align 1
@str.121 = private unnamed_addr constant [33 x i8] c"main(): GPU statistics, streamed\00", align 1
@str.122 = private unnamed_addr constant [48 x i8] c"\09Note: This is elapsed time for all to compute.\00", align 1
@str.123 = private unnamed_addr constant [59 x i8] c"main(): comparing Monte Carlo and Black-Scholes results...\00", align 1
@str.124 = private unnamed_addr constant [34 x i8] c"main(): running CPU MonteCarlo...\00", align 1
@str.125 = private unnamed_addr constant [17 x i8] c"Shutting down...\00", align 1
@str.126 = private unnamed_addr constant [16 x i8] c"Test Summary...\00", align 1

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

; Function Attrs: nounwind uwtable
define dso_local float @_Z9randFloatff(float, float) local_unnamed_addr #4 {
  %3 = tail call i32 @rand() #17
  %4 = sitofp i32 %3 to float
  %5 = fmul float %4, 0x3E00000000000000
  %6 = fsub float 1.000000e+00, %5
  %7 = fmul float %6, %0
  %8 = fmul float %5, %1
  %9 = fadd float %8, %7
  ret float %9
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: norecurse noreturn uwtable
define dso_local i32 @main(i32, i8**) local_unnamed_addr #6 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca %struct.cudaDeviceProp, align 8
  %6 = alloca %struct.TOptionData, align 8
  %7 = alloca %struct.TOptionValue, align 4
  %8 = alloca %struct.TOptionData, align 8
  store i32 %0, i32* %3, align 4, !tbaa !2
  store i32* %3, i32** @pArgc, align 8, !tbaa !6
  store i8** %1, i8*** @pArgv, align 8, !tbaa !6
  %9 = load i8*, i8** %1, align 8, !tbaa !6
  %10 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str, i64 0, i64 0), i8* %9)
  %11 = bitcast i32* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %11) #17
  %12 = call i32 @cudaGetDeviceCount(i32* nonnull %4)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %12, i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 138)
  %13 = load i32, i32* %4, align 4, !tbaa !2
  %14 = sext i32 %13 to i64
  %15 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %14, i64 8)
  %16 = extractvalue { i64, i1 } %15, 1
  %17 = extractvalue { i64, i1 } %15, 0
  %18 = select i1 %16, i64 -1, i64 %17
  %19 = call i8* @_Znam(i64 %18) #18
  store i8* %19, i8** bitcast (%class.StopWatchInterface*** @hTimer to i8**), align 8, !tbaa !6
  %20 = load i32, i32* %4, align 4, !tbaa !2
  %21 = icmp sgt i32 %20, 0
  br i1 %21, label %42, label %22

22:                                               ; preds = %42, %2
  %23 = phi i32 [ %20, %2 ], [ %51, %42 ]
  %24 = call i8* @_Znam(i64 20480) #18
  %25 = bitcast i8* %24 to %struct.TOptionData*
  %26 = call i8* @_Znam(i64 8192) #18
  %27 = bitcast i8* %26 to %struct.TOptionValue*
  %28 = call i8* @_Znam(i64 4096) #18
  %29 = sext i32 %23 to i64
  %30 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %29, i64 88)
  %31 = extractvalue { i64, i1 } %30, 1
  %32 = extractvalue { i64, i1 } %30, 0
  %33 = select i1 %31, i64 -1, i64 %32
  %34 = call i8* @_Znam(i64 %33) #18
  %35 = call i32 @puts(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @str, i64 0, i64 0))
  %36 = call i32 @puts(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @str.119, i64 0, i64 0))
  %37 = load i32, i32* %4, align 4, !tbaa !2
  %38 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.5, i64 0, i64 0), i32 %37)
  %39 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.6, i64 0, i64 0), i32 1024)
  %40 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.7, i64 0, i64 0), i32 262144)
  %41 = call i32 @puts(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @str.120, i64 0, i64 0))
  call void @srand(i32 123) #17
  br label %54

42:                                               ; preds = %2, %42
  %43 = phi i64 [ %50, %42 ], [ 0, %2 ]
  %44 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %45 = getelementptr inbounds %class.StopWatchInterface*, %class.StopWatchInterface** %44, i64 %43
  %46 = call zeroext i1 @_Z14sdkCreateTimerPP18StopWatchInterface(%class.StopWatchInterface** %45)
  %47 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %48 = getelementptr inbounds %class.StopWatchInterface*, %class.StopWatchInterface** %47, i64 %43
  %49 = call zeroext i1 @_Z13sdkResetTimerPP18StopWatchInterface(%class.StopWatchInterface** %48)
  %50 = add nuw nsw i64 %43, 1
  %51 = load i32, i32* %4, align 4, !tbaa !2
  %52 = sext i32 %51 to i64
  %53 = icmp slt i64 %50, %52
  br i1 %53, label %42, label %22

54:                                               ; preds = %54, %22
  %55 = phi i64 [ 0, %22 ], [ %66, %54 ]
  %56 = call float @_Z9randFloatff(float 5.000000e+00, float 5.000000e+01)
  %57 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %55, i32 0
  store float %56, float* %57, align 4, !tbaa !8
  %58 = call float @_Z9randFloatff(float 1.000000e+01, float 2.500000e+01)
  %59 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %55, i32 1
  store float %58, float* %59, align 4, !tbaa !11
  %60 = call float @_Z9randFloatff(float 1.000000e+00, float 5.000000e+00)
  %61 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %55, i32 2
  store float %60, float* %61, align 4, !tbaa !12
  %62 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %55, i32 3
  store float 0x3FAEB851E0000000, float* %62, align 4, !tbaa !13
  %63 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %55, i32 4
  store float 0x3FB99999A0000000, float* %63, align 4, !tbaa !14
  %64 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %27, i64 %55, i32 0
  store float -1.000000e+00, float* %64, align 4, !tbaa !15
  %65 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %27, i64 %55, i32 1
  store float -1.000000e+00, float* %65, align 4, !tbaa !17
  %66 = add nuw nsw i64 %55, 1
  %67 = icmp eq i64 %66, 1024
  br i1 %67, label %68, label %54

68:                                               ; preds = %54
  %69 = bitcast i8* %28 to float*
  %70 = bitcast i8* %34 to %struct.TOptionPlan*
  %71 = load i32, i32* %4, align 4, !tbaa !2
  %72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.9, i64 0, i64 0), i32 %71)
  %73 = load i32, i32* %4, align 4, !tbaa !2
  %74 = icmp sgt i32 %73, 0
  br i1 %74, label %75, label %78

75:                                               ; preds = %68
  %76 = load i32, i32* %4, align 4, !tbaa !2
  %77 = sext i32 %76 to i64
  br label %84

78:                                               ; preds = %84, %68
  %79 = phi i32 [ %73, %68 ], [ %76, %84 ]
  %80 = srem i32 1024, %79
  %81 = icmp eq i32 %80, 0
  br i1 %81, label %91, label %82

82:                                               ; preds = %78
  %83 = zext i32 %80 to i64
  br label %95

84:                                               ; preds = %75, %84
  %85 = phi i64 [ 0, %75 ], [ %89, %84 ]
  %86 = phi i32 [ %73, %75 ], [ %76, %84 ]
  %87 = sdiv i32 1024, %86
  %88 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %85, i32 1
  store i32 %87, i32* %88, align 4, !tbaa !18
  %89 = add nuw nsw i64 %85, 1
  %90 = icmp slt i64 %89, %77
  br i1 %90, label %84, label %78

91:                                               ; preds = %95, %78
  %92 = icmp sgt i32 %79, 0
  br i1 %92, label %93, label %119

93:                                               ; preds = %91
  %94 = zext i32 %79 to i64
  br label %102

95:                                               ; preds = %95, %82
  %96 = phi i64 [ 0, %82 ], [ %100, %95 ]
  %97 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %96, i32 1
  %98 = load i32, i32* %97, align 4, !tbaa !18
  %99 = add nsw i32 %98, 1
  store i32 %99, i32* %97, align 4, !tbaa !18
  %100 = add nuw nsw i64 %96, 1
  %101 = icmp eq i64 %100, %83
  br i1 %101, label %91, label %95

102:                                              ; preds = %102, %93
  %103 = phi i64 [ 0, %93 ], [ %117, %102 ]
  %104 = phi i32 [ 0, %93 ], [ %116, %102 ]
  %105 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %103, i32 0
  %106 = trunc i64 %103 to i32
  store i32 %106, i32* %105, align 8, !tbaa !20
  %107 = sext i32 %104 to i64
  %108 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %107
  %109 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %103, i32 2
  store %struct.TOptionData* %108, %struct.TOptionData** %109, align 8, !tbaa !21
  %110 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %27, i64 %107
  %111 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %103, i32 3
  store %struct.TOptionValue* %110, %struct.TOptionValue** %111, align 8, !tbaa !22
  %112 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %103, i32 10
  store i32 262144, i32* %112, align 8, !tbaa !23
  %113 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %103, i32 1
  %114 = load i32, i32* %113, align 4, !tbaa !18
  %115 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %103, i32 12
  store i32 %114, i32* %115, align 8, !tbaa !24
  %116 = add nsw i32 %114, %104
  %117 = add nuw nsw i64 %103, 1
  %118 = icmp eq i64 %117, %94
  br i1 %118, label %119, label %102

119:                                              ; preds = %102, %91
  call fastcc void @_ZL11multiSolverP11TOptionPlani(%struct.TOptionPlan* nonnull %70, i32 %79)
  %120 = call i32 @puts(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @str.121, i64 0, i64 0))
  %121 = load i32, i32* %4, align 4, !tbaa !2
  %122 = icmp sgt i32 %121, 0
  br i1 %122, label %123, label %142

123:                                              ; preds = %119
  %124 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %5, i64 0, i32 0, i64 0
  br label %125

125:                                              ; preds = %123, %125
  %126 = phi i64 [ 0, %123 ], [ %138, %125 ]
  call void @llvm.lifetime.start.p0i8(i64 712, i8* nonnull %124) #17
  %127 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %126, i32 0
  %128 = load i32, i32* %127, align 8, !tbaa !20
  %129 = call i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp* nonnull %5, i32 %128)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %129, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @.str.11, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 218)
  %130 = load i32, i32* %127, align 8, !tbaa !20
  %131 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.12, i64 0, i64 0), i32 %130, i8* nonnull %124)
  %132 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %126, i32 1
  %133 = load i32, i32* %132, align 4, !tbaa !18
  %134 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.13, i64 0, i64 0), i32 %133)
  %135 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %70, i64 %126, i32 10
  %136 = load i32, i32* %135, align 8, !tbaa !23
  %137 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.14, i64 0, i64 0), i32 %136)
  call void @llvm.lifetime.end.p0i8(i64 712, i8* nonnull %124) #17
  %138 = add nuw nsw i64 %126, 1
  %139 = load i32, i32* %4, align 4, !tbaa !2
  %140 = sext i32 %139 to i64
  %141 = icmp slt i64 %138, %140
  br i1 %141, label %125, label %142

142:                                              ; preds = %125, %119
  %143 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %144 = call float @_Z16sdkGetTimerValuePP18StopWatchInterface(%class.StopWatchInterface** %143)
  %145 = fpext float %144 to double
  %146 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.15, i64 0, i64 0), double %145)
  %147 = call i32 @puts(i8* getelementptr inbounds ([48 x i8], [48 x i8]* @str.122, i64 0, i64 0))
  %148 = fmul double %145, 1.000000e-03
  %149 = fdiv double 1.024000e+03, %148
  %150 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.17, i64 0, i64 0), double %149)
  %151 = call i32 @puts(i8* getelementptr inbounds ([59 x i8], [59 x i8]* @str.123, i64 0, i64 0))
  %152 = bitcast %struct.TOptionData* %6 to i8*
  br label %153

153:                                              ; preds = %172, %142
  %154 = phi i64 [ 0, %142 ], [ %174, %172 ]
  %155 = phi double [ 0.000000e+00, %142 ], [ %173, %172 ]
  %156 = getelementptr inbounds float, float* %69, i64 %154
  %157 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %154
  %158 = bitcast %struct.TOptionData* %157 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %152, i8* nonnull align 4 %158, i64 20, i1 false), !tbaa.struct !25
  call void @BlackScholesCall(float* nonnull dereferenceable(4) %156, %struct.TOptionData* nonnull byval(%struct.TOptionData) align 8 %6)
  %159 = load float, float* %156, align 4, !tbaa !26
  %160 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %27, i64 %154, i32 0
  %161 = load float, float* %160, align 4, !tbaa !15
  %162 = fsub float %159, %161
  %163 = call float @_ZSt4fabsf(float %162)
  %164 = fpext float %163 to double
  %165 = fcmp ogt double %164, 0x3EB0C6F7A0B5ED8D
  br i1 %165, label %166, label %172

166:                                              ; preds = %153
  %167 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %27, i64 %154, i32 1
  %168 = load float, float* %167, align 4, !tbaa !17
  %169 = fpext float %168 to double
  %170 = fdiv double %169, %164
  %171 = fadd double %155, %170
  br label %172

172:                                              ; preds = %153, %166
  %173 = phi double [ %171, %166 ], [ %155, %153 ]
  %174 = add nuw nsw i64 %154, 1
  %175 = icmp eq i64 %174, 1024
  br i1 %175, label %176, label %153

176:                                              ; preds = %172
  %177 = call i32 @puts(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @str.124, i64 0, i64 0))
  %178 = bitcast %struct.TOptionValue* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %178) #17
  %179 = bitcast %struct.TOptionData* %8 to i8*
  %180 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %7, i64 0, i32 0
  br label %181

181:                                              ; preds = %181, %176
  %182 = phi i64 [ 0, %176 ], [ %198, %181 ]
  %183 = phi double [ 0.000000e+00, %176 ], [ %197, %181 ]
  %184 = phi double [ 0.000000e+00, %176 ], [ %194, %181 ]
  %185 = getelementptr inbounds %struct.TOptionData, %struct.TOptionData* %25, i64 %182
  %186 = bitcast %struct.TOptionData* %185 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %179, i8* nonnull align 4 %186, i64 20, i1 false), !tbaa.struct !25
  call void @MonteCarloCPU(%struct.TOptionValue* nonnull dereferenceable(8) %7, %struct.TOptionData* nonnull byval(%struct.TOptionData) align 8 %8, float* null, i32 262144)
  %187 = load float, float* %180, align 4, !tbaa !15
  %188 = getelementptr inbounds %struct.TOptionValue, %struct.TOptionValue* %27, i64 %182, i32 0
  %189 = load float, float* %188, align 4, !tbaa !15
  %190 = fsub float %187, %189
  %191 = call float @_ZSt4fabsf(float %190)
  %192 = fpext float %191 to double
  %193 = load float, float* %180, align 4, !tbaa !15
  %194 = fadd double %184, %192
  %195 = call float @llvm.fabs.f32(float %193)
  %196 = fpext float %195 to double
  %197 = fadd double %183, %196
  %198 = add nuw nsw i64 %182, 1
  %199 = icmp eq i64 %198, 1024
  br i1 %199, label %200, label %181

200:                                              ; preds = %181
  %201 = fmul double %173, 0x3F50000000000000
  %202 = fdiv double %194, %197
  %203 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.20, i64 0, i64 0), double %202)
  %204 = call i32 @puts(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @str.125, i64 0, i64 0))
  %205 = load i32, i32* %4, align 4, !tbaa !2
  %206 = icmp sgt i32 %205, 0
  br i1 %206, label %207, label %218

207:                                              ; preds = %200, %207
  %208 = phi i64 [ %214, %207 ], [ 0, %200 ]
  %209 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %210 = getelementptr inbounds %class.StopWatchInterface*, %class.StopWatchInterface** %209, i64 %208
  %211 = call zeroext i1 @_Z13sdkStartTimerPP18StopWatchInterface(%class.StopWatchInterface** %210)
  %212 = trunc i64 %208 to i32
  %213 = call i32 @cudaSetDevice(i32 %212)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %213, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.22, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 274)
  %214 = add nuw nsw i64 %208, 1
  %215 = load i32, i32* %4, align 4, !tbaa !2
  %216 = sext i32 %215 to i64
  %217 = icmp slt i64 %214, %216
  br i1 %217, label %207, label %218

218:                                              ; preds = %207, %200
  call void @_ZdaPv(i8* nonnull %34) #19
  call void @_ZdaPv(i8* nonnull %28) #19
  call void @_ZdaPv(i8* nonnull %26) #19
  call void @_ZdaPv(i8* nonnull %24) #19
  %219 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %220 = icmp eq %class.StopWatchInterface** %219, null
  br i1 %220, label %223, label %221

221:                                              ; preds = %218
  %222 = bitcast %class.StopWatchInterface** %219 to i8*
  call void @_ZdaPv(i8* %222) #19
  br label %223

223:                                              ; preds = %221, %218
  %224 = call i32 @puts(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @str.126, i64 0, i64 0))
  %225 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.24, i64 0, i64 0), double %202)
  %226 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.25, i64 0, i64 0), double %201)
  %227 = fcmp ogt double %201, 1.000000e+00
  %228 = select i1 %227, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.26, i64 0, i64 0), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.27, i64 0, i64 0)
  %229 = call i32 (i8*, ...) @printf(i8* %228)
  %230 = xor i1 %227, true
  %231 = zext i1 %230 to i32
  call void @exit(i32 %231) #20
  unreachable
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #7

; Function Attrs: uwtable
define linkonce_odr dso_local void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32, i8*, i8*, i32) local_unnamed_addr #0 comdat {
  %5 = icmp eq i32 %0, 0
  br i1 %5, label %11, label %6

6:                                                ; preds = %4
  %7 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %8 = tail call fastcc i8* @_ZL17_cudaGetErrorEnum9cudaError(i32 %0)
  %9 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %7, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.36, i64 0, i64 0), i8* %2, i32 %3, i32 %0, i8* %8, i8* %1) #21
  %10 = tail call i32 @cudaDeviceReset()
  tail call void @exit(i32 1) #20
  unreachable

11:                                               ; preds = %4
  ret void
}

declare dso_local i32 @cudaGetDeviceCount(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #8

; Function Attrs: nobuiltin nofree
declare dso_local noalias nonnull i8* @_Znam(i64) local_unnamed_addr #9

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local zeroext i1 @_Z14sdkCreateTimerPP18StopWatchInterface(%class.StopWatchInterface**) local_unnamed_addr #10 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = tail call i8* @_Znwm(i64 40) #18
  %3 = bitcast i8* %2 to %class.StopWatchLinux*
  invoke void @_ZN14StopWatchLinuxC2Ev(%class.StopWatchLinux* nonnull %3)
          to label %4 unwind label %6

4:                                                ; preds = %1
  %5 = bitcast %class.StopWatchInterface** %0 to i8**
  store i8* %2, i8** %5, align 8, !tbaa !6
  ret i1 true

6:                                                ; preds = %1
  %7 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZdlPv(i8* nonnull %2) #19
  resume { i8*, i32 } %7
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local zeroext i1 @_Z13sdkResetTimerPP18StopWatchInterface(%class.StopWatchInterface**) local_unnamed_addr #10 comdat {
  %2 = load %class.StopWatchInterface*, %class.StopWatchInterface** %0, align 8, !tbaa !6
  %3 = icmp eq %class.StopWatchInterface* %2, null
  br i1 %3, label %9, label %4

4:                                                ; preds = %1
  %5 = bitcast %class.StopWatchInterface* %2 to void (%class.StopWatchInterface*)***
  %6 = load void (%class.StopWatchInterface*)**, void (%class.StopWatchInterface*)*** %5, align 8, !tbaa !27
  %7 = getelementptr inbounds void (%class.StopWatchInterface*)*, void (%class.StopWatchInterface*)** %6, i64 4
  %8 = load void (%class.StopWatchInterface*)*, void (%class.StopWatchInterface*)** %7, align 8
  tail call void %8(%class.StopWatchInterface* nonnull %2)
  br label %9

9:                                                ; preds = %1, %4
  ret i1 true
}

; Function Attrs: nounwind
declare dso_local void @srand(i32) local_unnamed_addr #2

; Function Attrs: norecurse uwtable
define internal fastcc void @_ZL11multiSolverP11TOptionPlani(%struct.TOptionPlan*, i32) unnamed_addr #11 {
  %3 = alloca %struct.cudaDeviceProp, align 8
  %4 = sext i32 %1 to i64
  %5 = shl nsw i64 %4, 3
  %6 = tail call noalias i8* @malloc(i64 %5) #17
  %7 = bitcast i8* %6 to %struct.CUstream_st**
  %8 = tail call noalias i8* @malloc(i64 %5) #17
  %9 = bitcast i8* %8 to %struct.CUevent_st**
  %10 = icmp sgt i32 %1, 0
  br i1 %10, label %11, label %43

11:                                               ; preds = %2
  %12 = zext i32 %1 to i64
  br label %18

13:                                               ; preds = %18
  %14 = icmp sgt i32 %1, 0
  br i1 %14, label %15, label %43

15:                                               ; preds = %13
  %16 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %3, i64 0, i32 0, i64 0
  %17 = zext i32 %1 to i64
  br label %33

18:                                               ; preds = %18, %11
  %19 = phi i64 [ 0, %11 ], [ %27, %18 ]
  %20 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %0, i64 %19, i32 0
  %21 = load i32, i32* %20, align 8, !tbaa !20
  %22 = tail call i32 @cudaSetDevice(i32 %21)
  tail call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %22, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.28, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 68)
  %23 = getelementptr inbounds %struct.CUstream_st*, %struct.CUstream_st** %7, i64 %19
  %24 = tail call i32 @cudaStreamCreate(%struct.CUstream_st** %23)
  tail call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %24, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 69)
  %25 = getelementptr inbounds %struct.CUevent_st*, %struct.CUevent_st** %9, i64 %19
  %26 = tail call i32 @cudaEventCreate(%struct.CUevent_st** %25)
  tail call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %26, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.30, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 70)
  %27 = add nuw nsw i64 %19, 1
  %28 = icmp eq i64 %27, %12
  br i1 %28, label %13, label %18

29:                                               ; preds = %33
  %30 = icmp sgt i32 %1, 0
  br i1 %30, label %31, label %43

31:                                               ; preds = %29
  %32 = zext i32 %1 to i64
  br label %51

33:                                               ; preds = %33, %15
  %34 = phi i64 [ 0, %15 ], [ %41, %33 ]
  %35 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %0, i64 %34
  %36 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %35, i64 0, i32 0
  %37 = load i32, i32* %36, align 8, !tbaa !20
  %38 = call i32 @cudaSetDevice(i32 %37)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %38, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.28, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 79)
  call void @llvm.lifetime.start.p0i8(i64 712, i8* nonnull %16) #17
  %39 = load i32, i32* %36, align 8, !tbaa !20
  %40 = call i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp* nonnull %3, i32 %39)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %40, i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 82)
  call void @initMonteCarloGPU(%struct.TOptionPlan* %35)
  call void @llvm.lifetime.end.p0i8(i64 712, i8* nonnull %16) #17
  %41 = add nuw nsw i64 %34, 1
  %42 = icmp eq i64 %41, %17
  br i1 %42, label %29, label %33

43:                                               ; preds = %51, %2, %13, %29
  %44 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %45 = call zeroext i1 @_Z13sdkResetTimerPP18StopWatchInterface(%class.StopWatchInterface** %44)
  %46 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %47 = call zeroext i1 @_Z13sdkStartTimerPP18StopWatchInterface(%class.StopWatchInterface** %46)
  %48 = icmp sgt i32 %1, 0
  br i1 %48, label %49, label %77

49:                                               ; preds = %43
  %50 = zext i32 %1 to i64
  br label %63

51:                                               ; preds = %51, %31
  %52 = phi i64 [ 0, %31 ], [ %57, %51 ]
  %53 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %0, i64 %52, i32 0
  %54 = load i32, i32* %53, align 8, !tbaa !20
  %55 = call i32 @cudaSetDevice(i32 %54)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %55, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.28, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 90)
  %56 = call i32 @cudaDeviceSynchronize()
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %56, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 91)
  %57 = add nuw nsw i64 %52, 1
  %58 = icmp eq i64 %57, %32
  br i1 %58, label %43, label %51

59:                                               ; preds = %63
  %60 = icmp sgt i32 %1, 0
  br i1 %60, label %61, label %77

61:                                               ; preds = %59
  %62 = zext i32 %1 to i64
  br label %86

63:                                               ; preds = %63, %49
  %64 = phi i64 [ 0, %49 ], [ %75, %63 ]
  %65 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %0, i64 %64
  %66 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %65, i64 0, i32 0
  %67 = load i32, i32* %66, align 8, !tbaa !20
  %68 = call i32 @cudaSetDevice(i32 %67)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %68, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.28, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 99)
  %69 = getelementptr inbounds %struct.CUstream_st*, %struct.CUstream_st** %7, i64 %64
  %70 = load %struct.CUstream_st*, %struct.CUstream_st** %69, align 8, !tbaa !6
  call void @MonteCarloGPU(%struct.TOptionPlan* %65, %struct.CUstream_st* %70)
  %71 = getelementptr inbounds %struct.CUevent_st*, %struct.CUevent_st** %9, i64 %64
  %72 = load %struct.CUevent_st*, %struct.CUevent_st** %71, align 8, !tbaa !6
  %73 = load %struct.CUstream_st*, %struct.CUstream_st** %69, align 8, !tbaa !6
  %74 = call i32 @cudaEventRecord(%struct.CUevent_st* %72, %struct.CUstream_st* %73)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %74, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.33, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 104)
  %75 = add nuw nsw i64 %64, 1
  %76 = icmp eq i64 %75, %50
  br i1 %76, label %59, label %63

77:                                               ; preds = %59, %43
  %78 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %79 = call zeroext i1 @_Z12sdkStopTimerPP18StopWatchInterface(%class.StopWatchInterface** %78)
  br label %96

80:                                               ; preds = %86
  %81 = load %class.StopWatchInterface**, %class.StopWatchInterface*** @hTimer, align 8, !tbaa !6
  %82 = call zeroext i1 @_Z12sdkStopTimerPP18StopWatchInterface(%class.StopWatchInterface** %81)
  %83 = icmp sgt i32 %1, 0
  br i1 %83, label %84, label %96

84:                                               ; preds = %80
  %85 = zext i32 %1 to i64
  br label %97

86:                                               ; preds = %86, %61
  %87 = phi i64 [ 0, %61 ], [ %94, %86 ]
  %88 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %0, i64 %87, i32 0
  %89 = load i32, i32* %88, align 8, !tbaa !20
  %90 = call i32 @cudaSetDevice(i32 %89)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %90, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.28, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 108)
  %91 = getelementptr inbounds %struct.CUevent_st*, %struct.CUevent_st** %9, i64 %87
  %92 = load %struct.CUevent_st*, %struct.CUevent_st** %91, align 8, !tbaa !6
  %93 = call i32 @cudaEventSynchronize(%struct.CUevent_st* %92)
  %94 = add nuw nsw i64 %87, 1
  %95 = icmp eq i64 %94, %62
  br i1 %95, label %80, label %86

96:                                               ; preds = %97, %77, %80
  ret void

97:                                               ; preds = %97, %84
  %98 = phi i64 [ 0, %84 ], [ %109, %97 ]
  %99 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %0, i64 %98
  %100 = getelementptr inbounds %struct.TOptionPlan, %struct.TOptionPlan* %99, i64 0, i32 0
  %101 = load i32, i32* %100, align 8, !tbaa !20
  %102 = call i32 @cudaSetDevice(i32 %101)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %102, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.28, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 116)
  call void @closeMonteCarloGPU(%struct.TOptionPlan* %99)
  %103 = getelementptr inbounds %struct.CUstream_st*, %struct.CUstream_st** %7, i64 %98
  %104 = load %struct.CUstream_st*, %struct.CUstream_st** %103, align 8, !tbaa !6
  %105 = call i32 @cudaStreamDestroy(%struct.CUstream_st* %104)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %105, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.34, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 118)
  %106 = getelementptr inbounds %struct.CUevent_st*, %struct.CUevent_st** %9, i64 %98
  %107 = load %struct.CUevent_st*, %struct.CUevent_st** %106, align 8, !tbaa !6
  %108 = call i32 @cudaEventDestroy(%struct.CUevent_st* %107)
  call void @_Z5checkI9cudaErrorEvT_PKcS3_i(i32 %108, i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.35, i64 0, i64 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.2, i64 0, i64 0), i32 119)
  %109 = add nuw nsw i64 %98, 1
  %110 = icmp eq i64 %109, %85
  br i1 %110, label %96, label %97
}

declare dso_local i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp*, i32) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local float @_Z16sdkGetTimerValuePP18StopWatchInterface(%class.StopWatchInterface**) local_unnamed_addr #10 comdat {
  %2 = load %class.StopWatchInterface*, %class.StopWatchInterface** %0, align 8, !tbaa !6
  %3 = icmp eq %class.StopWatchInterface* %2, null
  br i1 %3, label %10, label %4

4:                                                ; preds = %1
  %5 = bitcast %class.StopWatchInterface* %2 to float (%class.StopWatchInterface*)***
  %6 = load float (%class.StopWatchInterface*)**, float (%class.StopWatchInterface*)*** %5, align 8, !tbaa !27
  %7 = getelementptr inbounds float (%class.StopWatchInterface*)*, float (%class.StopWatchInterface*)** %6, i64 5
  %8 = load float (%class.StopWatchInterface*)*, float (%class.StopWatchInterface*)** %7, align 8
  %9 = tail call float %8(%class.StopWatchInterface* nonnull %2)
  br label %10

10:                                               ; preds = %1, %4
  %11 = phi float [ %9, %4 ], [ 0.000000e+00, %1 ]
  ret float %11
}

declare dso_local void @BlackScholesCall(float* dereferenceable(4), %struct.TOptionData* byval(%struct.TOptionData) align 8) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #5

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZSt4fabsf(float) local_unnamed_addr #12 comdat {
  %2 = tail call float @llvm.fabs.f32(float %0)
  ret float %2
}

declare dso_local void @MonteCarloCPU(%struct.TOptionValue* dereferenceable(8), %struct.TOptionData* byval(%struct.TOptionData) align 8, float*, i32) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local zeroext i1 @_Z13sdkStartTimerPP18StopWatchInterface(%class.StopWatchInterface**) local_unnamed_addr #10 comdat {
  %2 = load %class.StopWatchInterface*, %class.StopWatchInterface** %0, align 8, !tbaa !6
  %3 = icmp eq %class.StopWatchInterface* %2, null
  br i1 %3, label %9, label %4

4:                                                ; preds = %1
  %5 = bitcast %class.StopWatchInterface* %2 to void (%class.StopWatchInterface*)***
  %6 = load void (%class.StopWatchInterface*)**, void (%class.StopWatchInterface*)*** %5, align 8, !tbaa !27
  %7 = getelementptr inbounds void (%class.StopWatchInterface*)*, void (%class.StopWatchInterface*)** %6, i64 2
  %8 = load void (%class.StopWatchInterface*)*, void (%class.StopWatchInterface*)** %7, align 8
  tail call void %8(%class.StopWatchInterface* nonnull %2)
  br label %9

9:                                                ; preds = %1, %4
  ret i1 true
}

declare dso_local i32 @cudaSetDevice(i32) local_unnamed_addr #1

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdaPv(i8*) local_unnamed_addr #13

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #14

; Function Attrs: nobuiltin nofree
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #9

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN14StopWatchLinuxC2Ev(%class.StopWatchLinux*) unnamed_addr #0 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 0
  tail call void @_ZN18StopWatchInterfaceC2Ev(%class.StopWatchInterface* %2)
  %3 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV14StopWatchLinux, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %3, align 8, !tbaa !27
  %4 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 1
  %5 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 5
  store i32 0, i32* %5, align 4, !tbaa !29
  %6 = bitcast %struct.timeval* %4 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %6, i8 0, i64 25, i1 false)
  ret void
}

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #13

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN18StopWatchInterfaceC2Ev(%class.StopWatchInterface*) unnamed_addr #4 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchInterface, %class.StopWatchInterface* %0, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [9 x i8*] }, { [9 x i8*] }* @_ZTV18StopWatchInterface, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %2, align 8, !tbaa !27
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN14StopWatchLinuxD0Ev(%class.StopWatchLinux*) unnamed_addr #4 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 0
  tail call void @_ZN18StopWatchInterfaceD2Ev(%class.StopWatchInterface* %2) #17
  %3 = bitcast %class.StopWatchLinux* %0 to i8*
  tail call void @_ZdlPv(i8* %3) #19
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN14StopWatchLinux5startEv(%class.StopWatchLinux*) unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 1
  %3 = tail call i32 @gettimeofday(%struct.timeval* nonnull %2, %struct.timezone* null) #17
  %4 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 4
  store i8 1, i8* %4, align 8, !tbaa !34
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZN14StopWatchLinux4stopEv(%class.StopWatchLinux*) unnamed_addr #10 comdat align 2 {
  %2 = tail call float @_ZN14StopWatchLinux11getDiffTimeEv(%class.StopWatchLinux* %0)
  %3 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 2
  store float %2, float* %3, align 8, !tbaa !35
  %4 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 3
  %5 = load float, float* %4, align 4, !tbaa !36
  %6 = fadd float %2, %5
  store float %6, float* %4, align 4, !tbaa !36
  %7 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 4
  store i8 0, i8* %7, align 8, !tbaa !34
  %8 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 5
  %9 = load i32, i32* %8, align 4, !tbaa !29
  %10 = add nsw i32 %9, 1
  store i32 %10, i32* %8, align 4, !tbaa !29
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN14StopWatchLinux5resetEv(%class.StopWatchLinux*) unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 2
  store float 0.000000e+00, float* %2, align 8, !tbaa !35
  %3 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 3
  store float 0.000000e+00, float* %3, align 4, !tbaa !36
  %4 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 5
  store i32 0, i32* %4, align 4, !tbaa !29
  %5 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 4
  %6 = load i8, i8* %5, align 8, !tbaa !34, !range !37
  %7 = icmp eq i8 %6, 0
  br i1 %7, label %11, label %8

8:                                                ; preds = %1
  %9 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 1
  %10 = tail call i32 @gettimeofday(%struct.timeval* nonnull %9, %struct.timezone* null) #17
  br label %11

11:                                               ; preds = %1, %8
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZN14StopWatchLinux7getTimeEv(%class.StopWatchLinux*) unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 3
  %3 = load float, float* %2, align 4, !tbaa !36
  %4 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 4
  %5 = load i8, i8* %4, align 8, !tbaa !34, !range !37
  %6 = icmp eq i8 %5, 0
  br i1 %6, label %10, label %7

7:                                                ; preds = %1
  %8 = tail call float @_ZN14StopWatchLinux11getDiffTimeEv(%class.StopWatchLinux* nonnull %0)
  %9 = fadd float %3, %8
  br label %10

10:                                               ; preds = %1, %7
  %11 = phi float [ %9, %7 ], [ %3, %1 ]
  ret float %11
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZN14StopWatchLinux14getAverageTimeEv(%class.StopWatchLinux*) unnamed_addr #12 comdat align 2 {
  %2 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 5
  %3 = load i32, i32* %2, align 4, !tbaa !29
  %4 = icmp sgt i32 %3, 0
  br i1 %4, label %5, label %10

5:                                                ; preds = %1
  %6 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 3
  %7 = load float, float* %6, align 4, !tbaa !36
  %8 = sitofp i32 %3 to float
  %9 = fdiv float %7, %8
  br label %10

10:                                               ; preds = %1, %5
  %11 = phi float [ %9, %5 ], [ 0.000000e+00, %1 ]
  ret float %11
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN18StopWatchInterfaceD2Ev(%class.StopWatchInterface*) unnamed_addr #4 comdat align 2 {
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN18StopWatchInterfaceD0Ev(%class.StopWatchInterface*) unnamed_addr #4 comdat align 2 {
  tail call void @llvm.trap() #20
  unreachable
}

declare dso_local void @__cxa_pure_virtual() unnamed_addr

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #15

; Function Attrs: nofree nounwind
declare dso_local i32 @gettimeofday(%struct.timeval* nocapture, %struct.timezone* nocapture) local_unnamed_addr #7

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZN14StopWatchLinux11getDiffTimeEv(%class.StopWatchLinux*) local_unnamed_addr #12 comdat align 2 {
  %2 = alloca %struct.timeval, align 8
  %3 = bitcast %struct.timeval* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3) #17
  %4 = call i32 @gettimeofday(%struct.timeval* nonnull %2, %struct.timezone* null) #17
  %5 = getelementptr inbounds %struct.timeval, %struct.timeval* %2, i64 0, i32 0
  %6 = load i64, i64* %5, align 8, !tbaa !38
  %7 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 1, i32 0
  %8 = load i64, i64* %7, align 8, !tbaa !39
  %9 = sub nsw i64 %6, %8
  %10 = sitofp i64 %9 to double
  %11 = fmul double %10, 1.000000e+03
  %12 = getelementptr inbounds %struct.timeval, %struct.timeval* %2, i64 0, i32 1
  %13 = load i64, i64* %12, align 8, !tbaa !40
  %14 = getelementptr inbounds %class.StopWatchLinux, %class.StopWatchLinux* %0, i64 0, i32 1, i32 1
  %15 = load i64, i64* %14, align 8, !tbaa !41
  %16 = sub nsw i64 %13, %15
  %17 = sitofp i64 %16 to double
  %18 = fmul double %17, 1.000000e-03
  %19 = fadd double %11, %18
  %20 = fptrunc double %19 to float
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3) #17
  ret float %20
}

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #7

declare dso_local i32 @cudaStreamCreate(%struct.CUstream_st**) local_unnamed_addr #1

declare dso_local i32 @cudaEventCreate(%struct.CUevent_st**) local_unnamed_addr #1

declare dso_local void @initMonteCarloGPU(%struct.TOptionPlan*) local_unnamed_addr #1

declare dso_local i32 @cudaDeviceSynchronize() local_unnamed_addr #1

declare dso_local void @MonteCarloGPU(%struct.TOptionPlan*, %struct.CUstream_st*) local_unnamed_addr #1

declare dso_local i32 @cudaEventRecord(%struct.CUevent_st*, %struct.CUstream_st*) local_unnamed_addr #1

declare dso_local i32 @cudaEventSynchronize(%struct.CUevent_st*) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local zeroext i1 @_Z12sdkStopTimerPP18StopWatchInterface(%class.StopWatchInterface**) local_unnamed_addr #10 comdat {
  %2 = load %class.StopWatchInterface*, %class.StopWatchInterface** %0, align 8, !tbaa !6
  %3 = icmp eq %class.StopWatchInterface* %2, null
  br i1 %3, label %9, label %4

4:                                                ; preds = %1
  %5 = bitcast %class.StopWatchInterface* %2 to void (%class.StopWatchInterface*)***
  %6 = load void (%class.StopWatchInterface*)**, void (%class.StopWatchInterface*)*** %5, align 8, !tbaa !27
  %7 = getelementptr inbounds void (%class.StopWatchInterface*)*, void (%class.StopWatchInterface*)** %6, i64 3
  %8 = load void (%class.StopWatchInterface*)*, void (%class.StopWatchInterface*)** %7, align 8
  tail call void %8(%class.StopWatchInterface* nonnull %2)
  br label %9

9:                                                ; preds = %1, %4
  ret i1 true
}

declare dso_local void @closeMonteCarloGPU(%struct.TOptionPlan*) local_unnamed_addr #1

declare dso_local i32 @cudaStreamDestroy(%struct.CUstream_st*) local_unnamed_addr #1

declare dso_local i32 @cudaEventDestroy(%struct.CUevent_st*) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #8

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #7

; Function Attrs: norecurse nounwind readnone uwtable
define internal fastcc i8* @_ZL17_cudaGetErrorEnum9cudaError(i32) unnamed_addr #16 {
  switch i32 %0, label %82 [
    i32 0, label %83
    i32 1, label %2
    i32 2, label %3
    i32 3, label %4
    i32 4, label %5
    i32 5, label %6
    i32 6, label %7
    i32 7, label %8
    i32 8, label %9
    i32 9, label %10
    i32 10, label %11
    i32 11, label %12
    i32 12, label %13
    i32 13, label %14
    i32 14, label %15
    i32 15, label %16
    i32 16, label %17
    i32 17, label %18
    i32 18, label %19
    i32 19, label %20
    i32 20, label %21
    i32 21, label %22
    i32 22, label %23
    i32 23, label %24
    i32 24, label %25
    i32 25, label %26
    i32 26, label %27
    i32 27, label %28
    i32 28, label %29
    i32 29, label %30
    i32 30, label %31
    i32 31, label %32
    i32 32, label %33
    i32 33, label %34
    i32 34, label %35
    i32 35, label %36
    i32 36, label %37
    i32 37, label %38
    i32 38, label %39
    i32 39, label %40
    i32 40, label %41
    i32 41, label %42
    i32 42, label %43
    i32 43, label %44
    i32 44, label %45
    i32 45, label %46
    i32 46, label %47
    i32 47, label %48
    i32 48, label %49
    i32 49, label %50
    i32 50, label %51
    i32 51, label %52
    i32 54, label %53
    i32 55, label %54
    i32 56, label %55
    i32 57, label %56
    i32 58, label %57
    i32 59, label %58
    i32 60, label %59
    i32 61, label %60
    i32 62, label %61
    i32 63, label %62
    i32 64, label %63
    i32 65, label %64
    i32 66, label %65
    i32 67, label %66
    i32 68, label %67
    i32 69, label %68
    i32 70, label %69
    i32 71, label %70
    i32 72, label %71
    i32 73, label %72
    i32 74, label %73
    i32 75, label %74
    i32 76, label %75
    i32 77, label %76
    i32 78, label %77
    i32 79, label %78
    i32 127, label %79
    i32 10000, label %80
    i32 80, label %81
  ]

2:                                                ; preds = %1
  br label %83

3:                                                ; preds = %1
  br label %83

4:                                                ; preds = %1
  br label %83

5:                                                ; preds = %1
  br label %83

6:                                                ; preds = %1
  br label %83

7:                                                ; preds = %1
  br label %83

8:                                                ; preds = %1
  br label %83

9:                                                ; preds = %1
  br label %83

10:                                               ; preds = %1
  br label %83

11:                                               ; preds = %1
  br label %83

12:                                               ; preds = %1
  br label %83

13:                                               ; preds = %1
  br label %83

14:                                               ; preds = %1
  br label %83

15:                                               ; preds = %1
  br label %83

16:                                               ; preds = %1
  br label %83

17:                                               ; preds = %1
  br label %83

18:                                               ; preds = %1
  br label %83

19:                                               ; preds = %1
  br label %83

20:                                               ; preds = %1
  br label %83

21:                                               ; preds = %1
  br label %83

22:                                               ; preds = %1
  br label %83

23:                                               ; preds = %1
  br label %83

24:                                               ; preds = %1
  br label %83

25:                                               ; preds = %1
  br label %83

26:                                               ; preds = %1
  br label %83

27:                                               ; preds = %1
  br label %83

28:                                               ; preds = %1
  br label %83

29:                                               ; preds = %1
  br label %83

30:                                               ; preds = %1
  br label %83

31:                                               ; preds = %1
  br label %83

32:                                               ; preds = %1
  br label %83

33:                                               ; preds = %1
  br label %83

34:                                               ; preds = %1
  br label %83

35:                                               ; preds = %1
  br label %83

36:                                               ; preds = %1
  br label %83

37:                                               ; preds = %1
  br label %83

38:                                               ; preds = %1
  br label %83

39:                                               ; preds = %1
  br label %83

40:                                               ; preds = %1
  br label %83

41:                                               ; preds = %1
  br label %83

42:                                               ; preds = %1
  br label %83

43:                                               ; preds = %1
  br label %83

44:                                               ; preds = %1
  br label %83

45:                                               ; preds = %1
  br label %83

46:                                               ; preds = %1
  br label %83

47:                                               ; preds = %1
  br label %83

48:                                               ; preds = %1
  br label %83

49:                                               ; preds = %1
  br label %83

50:                                               ; preds = %1
  br label %83

51:                                               ; preds = %1
  br label %83

52:                                               ; preds = %1
  br label %83

53:                                               ; preds = %1
  br label %83

54:                                               ; preds = %1
  br label %83

55:                                               ; preds = %1
  br label %83

56:                                               ; preds = %1
  br label %83

57:                                               ; preds = %1
  br label %83

58:                                               ; preds = %1
  br label %83

59:                                               ; preds = %1
  br label %83

60:                                               ; preds = %1
  br label %83

61:                                               ; preds = %1
  br label %83

62:                                               ; preds = %1
  br label %83

63:                                               ; preds = %1
  br label %83

64:                                               ; preds = %1
  br label %83

65:                                               ; preds = %1
  br label %83

66:                                               ; preds = %1
  br label %83

67:                                               ; preds = %1
  br label %83

68:                                               ; preds = %1
  br label %83

69:                                               ; preds = %1
  br label %83

70:                                               ; preds = %1
  br label %83

71:                                               ; preds = %1
  br label %83

72:                                               ; preds = %1
  br label %83

73:                                               ; preds = %1
  br label %83

74:                                               ; preds = %1
  br label %83

75:                                               ; preds = %1
  br label %83

76:                                               ; preds = %1
  br label %83

77:                                               ; preds = %1
  br label %83

78:                                               ; preds = %1
  br label %83

79:                                               ; preds = %1
  br label %83

80:                                               ; preds = %1
  br label %83

81:                                               ; preds = %1
  br label %83

82:                                               ; preds = %1
  br label %83

83:                                               ; preds = %1, %82, %81, %80, %79, %78, %77, %76, %75, %74, %73, %72, %71, %70, %69, %68, %67, %66, %65, %64, %63, %62, %61, %60, %59, %58, %57, %56, %55, %54, %53, %52, %51, %50, %49, %48, %47, %46, %45, %44, %43, %42, %41, %40, %39, %38, %37, %36, %35, %34, %33, %32, %31, %30, %29, %28, %27, %26, %25, %24, %23, %22, %21, %20, %19, %18, %17, %16, %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2
  %84 = phi i8* [ getelementptr inbounds ([10 x i8], [10 x i8]* @.str.118, i64 0, i64 0), %82 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.117, i64 0, i64 0), %81 ], [ getelementptr inbounds ([24 x i8], [24 x i8]* @.str.116, i64 0, i64 0), %80 ], [ getelementptr inbounds ([24 x i8], [24 x i8]* @.str.115, i64 0, i64 0), %79 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.114, i64 0, i64 0), %78 ], [ getelementptr inbounds ([20 x i8], [20 x i8]* @.str.113, i64 0, i64 0), %77 ], [ getelementptr inbounds ([24 x i8], [24 x i8]* @.str.112, i64 0, i64 0), %76 ], [ getelementptr inbounds ([19 x i8], [19 x i8]* @.str.111, i64 0, i64 0), %75 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.110, i64 0, i64 0), %74 ], [ getelementptr inbounds ([27 x i8], [27 x i8]* @.str.109, i64 0, i64 0), %73 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.108, i64 0, i64 0), %72 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.107, i64 0, i64 0), %71 ], [ getelementptr inbounds ([22 x i8], [22 x i8]* @.str.106, i64 0, i64 0), %70 ], [ getelementptr inbounds ([22 x i8], [22 x i8]* @.str.105, i64 0, i64 0), %69 ], [ getelementptr inbounds ([36 x i8], [36 x i8]* @.str.104, i64 0, i64 0), %68 ], [ getelementptr inbounds ([27 x i8], [27 x i8]* @.str.103, i64 0, i64 0), %67 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.102, i64 0, i64 0), %66 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.101, i64 0, i64 0), %65 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.100, i64 0, i64 0), %64 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.99, i64 0, i64 0), %63 ], [ getelementptr inbounds ([25 x i8], [25 x i8]* @.str.98, i64 0, i64 0), %62 ], [ getelementptr inbounds ([33 x i8], [33 x i8]* @.str.97, i64 0, i64 0), %61 ], [ getelementptr inbounds ([37 x i8], [37 x i8]* @.str.96, i64 0, i64 0), %60 ], [ getelementptr inbounds ([22 x i8], [22 x i8]* @.str.95, i64 0, i64 0), %59 ], [ getelementptr inbounds ([16 x i8], [16 x i8]* @.str.94, i64 0, i64 0), %58 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.93, i64 0, i64 0), %57 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.92, i64 0, i64 0), %56 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.91, i64 0, i64 0), %55 ], [ getelementptr inbounds ([26 x i8], [26 x i8]* @.str.90, i64 0, i64 0), %54 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.89, i64 0, i64 0), %53 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.88, i64 0, i64 0), %52 ], [ getelementptr inbounds ([34 x i8], [34 x i8]* @.str.87, i64 0, i64 0), %51 ], [ getelementptr inbounds ([35 x i8], [35 x i8]* @.str.86, i64 0, i64 0), %50 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.85, i64 0, i64 0), %49 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.84, i64 0, i64 0), %48 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.83, i64 0, i64 0), %47 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.82, i64 0, i64 0), %46 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.81, i64 0, i64 0), %45 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.80, i64 0, i64 0), %44 ], [ getelementptr inbounds ([26 x i8], [26 x i8]* @.str.79, i64 0, i64 0), %43 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.78, i64 0, i64 0), %42 ], [ getelementptr inbounds ([36 x i8], [36 x i8]* @.str.77, i64 0, i64 0), %41 ], [ getelementptr inbounds ([26 x i8], [26 x i8]* @.str.76, i64 0, i64 0), %40 ], [ getelementptr inbounds ([18 x i8], [18 x i8]* @.str.75, i64 0, i64 0), %39 ], [ getelementptr inbounds ([24 x i8], [24 x i8]* @.str.74, i64 0, i64 0), %38 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.73, i64 0, i64 0), %37 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.72, i64 0, i64 0), %36 ], [ getelementptr inbounds ([18 x i8], [18 x i8]* @.str.71, i64 0, i64 0), %35 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.70, i64 0, i64 0), %34 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.69, i64 0, i64 0), %33 ], [ getelementptr inbounds ([27 x i8], [27 x i8]* @.str.68, i64 0, i64 0), %32 ], [ getelementptr inbounds ([17 x i8], [17 x i8]* @.str.67, i64 0, i64 0), %31 ], [ getelementptr inbounds ([25 x i8], [25 x i8]* @.str.66, i64 0, i64 0), %30 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.65, i64 0, i64 0), %29 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.64, i64 0, i64 0), %28 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.63, i64 0, i64 0), %27 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.62, i64 0, i64 0), %26 ], [ getelementptr inbounds ([25 x i8], [25 x i8]* @.str.61, i64 0, i64 0), %25 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.60, i64 0, i64 0), %24 ], [ getelementptr inbounds ([27 x i8], [27 x i8]* @.str.59, i64 0, i64 0), %23 ], [ getelementptr inbounds ([32 x i8], [32 x i8]* @.str.58, i64 0, i64 0), %22 ], [ getelementptr inbounds ([34 x i8], [34 x i8]* @.str.57, i64 0, i64 0), %21 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.56, i64 0, i64 0), %20 ], [ getelementptr inbounds ([24 x i8], [24 x i8]* @.str.55, i64 0, i64 0), %19 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.54, i64 0, i64 0), %18 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.53, i64 0, i64 0), %17 ], [ getelementptr inbounds ([33 x i8], [33 x i8]* @.str.52, i64 0, i64 0), %16 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.51, i64 0, i64 0), %15 ], [ getelementptr inbounds ([23 x i8], [23 x i8]* @.str.50, i64 0, i64 0), %14 ], [ getelementptr inbounds ([27 x i8], [27 x i8]* @.str.49, i64 0, i64 0), %13 ], [ getelementptr inbounds ([22 x i8], [22 x i8]* @.str.48, i64 0, i64 0), %12 ], [ getelementptr inbounds ([23 x i8], [23 x i8]* @.str.47, i64 0, i64 0), %11 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.46, i64 0, i64 0), %10 ], [ getelementptr inbounds ([31 x i8], [31 x i8]* @.str.45, i64 0, i64 0), %9 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.44, i64 0, i64 0), %8 ], [ getelementptr inbounds ([23 x i8], [23 x i8]* @.str.43, i64 0, i64 0), %7 ], [ getelementptr inbounds ([28 x i8], [28 x i8]* @.str.42, i64 0, i64 0), %6 ], [ getelementptr inbounds ([23 x i8], [23 x i8]* @.str.41, i64 0, i64 0), %5 ], [ getelementptr inbounds ([29 x i8], [29 x i8]* @.str.40, i64 0, i64 0), %4 ], [ getelementptr inbounds ([26 x i8], [26 x i8]* @.str.39, i64 0, i64 0), %3 ], [ getelementptr inbounds ([30 x i8], [30 x i8]* @.str.38, i64 0, i64 0), %2 ], [ getelementptr inbounds ([12 x i8], [12 x i8]* @.str.37, i64 0, i64 0), %1 ]
  ret i8* %84
}

declare dso_local i32 @cudaDeviceReset() local_unnamed_addr #1

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_MonteCarloMultiGPU.cpp() #0 section ".text.startup" {
  tail call fastcc void @__cxx_global_var_init()
  ret void
}

; Function Attrs: nofree nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #3

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { norecurse noreturn uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind readnone speculatable }
attributes #9 = { nobuiltin nofree "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #14 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #15 = { cold noreturn nounwind }
attributes #16 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #17 = { nounwind }
attributes #18 = { builtin }
attributes #19 = { builtin nounwind }
attributes #20 = { noreturn nounwind }
attributes #21 = { cold }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (git@github.com:llvm-mirror/clang.git aebe7c421069cfbd51fded0d29ea3c9c50a4dc91) (git@github.com:llvm-mirror/llvm.git b7d166cebcf619a3691eed3f994384aab3d80fa6)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !10, i64 0}
!9 = !{!"_ZTS11TOptionData", !10, i64 0, !10, i64 4, !10, i64 8, !10, i64 12, !10, i64 16}
!10 = !{!"float", !4, i64 0}
!11 = !{!9, !10, i64 4}
!12 = !{!9, !10, i64 8}
!13 = !{!9, !10, i64 12}
!14 = !{!9, !10, i64 16}
!15 = !{!16, !10, i64 0}
!16 = !{!"_ZTS12TOptionValue", !10, i64 0, !10, i64 4}
!17 = !{!16, !10, i64 4}
!18 = !{!19, !3, i64 4}
!19 = !{!"_ZTS11TOptionPlan", !3, i64 0, !3, i64 4, !7, i64 8, !7, i64 16, !7, i64 24, !7, i64 32, !7, i64 40, !7, i64 48, !7, i64 56, !7, i64 64, !3, i64 72, !10, i64 76, !3, i64 80}
!20 = !{!19, !3, i64 0}
!21 = !{!19, !7, i64 8}
!22 = !{!19, !7, i64 16}
!23 = !{!19, !3, i64 72}
!24 = !{!19, !3, i64 80}
!25 = !{i64 0, i64 4, !26, i64 4, i64 4, !26, i64 8, i64 4, !26, i64 12, i64 4, !26, i64 16, i64 4, !26}
!26 = !{!10, !10, i64 0}
!27 = !{!28, !28, i64 0}
!28 = !{!"vtable pointer", !5, i64 0}
!29 = !{!30, !3, i64 36}
!30 = !{!"_ZTS14StopWatchLinux", !31, i64 8, !10, i64 24, !10, i64 28, !33, i64 32, !3, i64 36}
!31 = !{!"_ZTS7timeval", !32, i64 0, !32, i64 8}
!32 = !{!"long", !4, i64 0}
!33 = !{!"bool", !4, i64 0}
!34 = !{!30, !33, i64 32}
!35 = !{!30, !10, i64 24}
!36 = !{!30, !10, i64 28}
!37 = !{i8 0, i8 2}
!38 = !{!31, !32, i64 0}
!39 = !{!30, !32, i64 8}
!40 = !{!31, !32, i64 8}
!41 = !{!30, !32, i64 16}
