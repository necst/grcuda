/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call price for a
 * given set of European options using Monte Carlo approach.
 * See supplied whitepaper for more explanations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// includes, project
#include "../common/helper_functions.h" // Helper functions (utilities, parsing, timing)
#include "../common/helper_cuda.h"      // helper functions (cuda error checking and initialization)

#include "MonteCarlo_common.h"

int *pArgc = NULL;
char **pArgv = NULL;

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high) {
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(
    TOptionValue &callValue,
    TOptionData optionData,
    float *h_Random,
    int pathN);

//Black-Scholes formula for call options
extern "C" void BlackScholesCall(
    float &CallResult,
    TOptionData optionData);

////////////////////////////////////////////////////////////////////////////////
// GPU-driving host thread
////////////////////////////////////////////////////////////////////////////////
//Timer
StopWatchInterface **hTimer = NULL;

static void multiSolver(TOptionPlan *plan, int nPlans) {

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(nPlans * sizeof(cudaStream_t));
    cudaEvent_t *events = (cudaEvent_t *)malloc(nPlans * sizeof(cudaEvent_t));

    for (int i = 0; i < nPlans; i++) {
        checkCudaErrors(cudaSetDevice(plan[i].device));
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
        checkCudaErrors(cudaEventCreate(&(events[i])));
    }

    //Init Each GPU
    // In CUDA 4.0 we can call cudaSetDevice multiple times to target each device
    // Set the device desired, then perform initializations on that device

    for (int i = 0; i < nPlans; i++) {
        // set the target device to perform initialization on
        checkCudaErrors(cudaSetDevice(plan[i].device));

        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, plan[i].device));

        // Allocate intermediate memory for MC integrator
        // and initialize RNG state
        initMonteCarloGPU(&plan[i]);
    }

    for (int i = 0; i < nPlans; i++) {
        checkCudaErrors(cudaSetDevice(plan[i].device));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    //Start the timer
    sdkResetTimer(&hTimer[0]);
    sdkStartTimer(&hTimer[0]);

    for (int i = 0; i < nPlans; i++) {
        checkCudaErrors(cudaSetDevice(plan[i].device));

        //Main computations
        MonteCarloGPU(&plan[i], streams[i]);

        checkCudaErrors(cudaEventRecord(events[i], streams[i]));
    }

    for (int i = 0; i < nPlans; i++) {
        checkCudaErrors(cudaSetDevice(plan[i].device));
        cudaEventSynchronize(events[i]);
    }

    //Stop the timer
    sdkStopTimer(&hTimer[0]);

    for (int i = 0; i < nPlans; i++) {
        checkCudaErrors(cudaSetDevice(plan[i].device));
        closeMonteCarloGPU(&plan[i]);
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCudaErrors(cudaEventDestroy(events[i]));
    }
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    char *multiMethodChoice = NULL;
    bool bqatest = true;

    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", argv[0]);

    //GPU number present in the system
    int GPU_N;
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));
    int nOptions = 1024;

    // select problem size
    int OPT_N = nOptions;
    int PATH_N = 262144;

    // initialize the timers
    hTimer = new StopWatchInterface *[GPU_N];

    for (int i = 0; i < GPU_N; i++) {
        sdkCreateTimer(&hTimer[i]);
        sdkResetTimer(&hTimer[i]);
    }

    //Input data array
    TOptionData *optionData = new TOptionData[OPT_N];
    //Final GPU MC results
    TOptionValue *callValueGPU = new TOptionValue[OPT_N];
    //"Theoretical" call values by Black-Scholes formula
    float *callValueBS = new float[OPT_N];
    //Solver config
    TOptionPlan *optionSolver = new TOptionPlan[GPU_N];

    int gpuBase, gpuIndex;
    int i;

    float time;

    double delta, ref, sumDelta, sumRef, sumReserve;

    printf("MonteCarloMultiGPU\n");
    printf("==================\n");
    printf("Number of GPUs          = %d\n", GPU_N);
    printf("Total number of options = %d\n", OPT_N);
    printf("Number of paths         = %d\n", PATH_N);

    printf("main(): generating input data...\n");
    srand(123);

    for (i = 0; i < OPT_N; i++) {
        optionData[i].S = randFloat(5.0f, 50.0f);
        optionData[i].X = randFloat(10.0f, 25.0f);
        optionData[i].T = randFloat(1.0f, 5.0f);
        optionData[i].R = 0.06f;
        optionData[i].V = 0.10f;
        callValueGPU[i].Expected = -1.0f;
        callValueGPU[i].Confidence = -1.0f;
    }

    printf("main(): starting %i host threads...\n", GPU_N);

    //Get option count for each GPU
    for (i = 0; i < GPU_N; i++) {
        optionSolver[i].optionCount = OPT_N / GPU_N;
    }

    //Take into account cases with "odd" option counts
    for (i = 0; i < (OPT_N % GPU_N); i++) {
        optionSolver[i].optionCount++;
    }

    //Assign GPU option ranges
    gpuBase = 0;

    for (i = 0; i < GPU_N; i++) {
        optionSolver[i].device = i;
        optionSolver[i].optionData = optionData + gpuBase;
        optionSolver[i].callValue = callValueGPU + gpuBase;
        optionSolver[i].pathN = PATH_N;
        optionSolver[i].gridSize = optionSolver[i].optionCount;
        gpuBase += optionSolver[i].optionCount;
    }

    multiSolver(optionSolver, GPU_N);

    printf("main(): GPU statistics, streamed\n");

    for (i = 0; i < GPU_N; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, optionSolver[i].device));
        printf("GPU Device #%i: %s\n", optionSolver[i].device, deviceProp.name);
        printf("Options         : %i\n", optionSolver[i].optionCount);
        printf("Simulation paths: %i\n", optionSolver[i].pathN);
    }

    time = sdkGetTimerValue(&hTimer[0]);
    printf("\nTotal time (ms.): %f\n", time);
    printf("\tNote: This is elapsed time for all to compute.\n");
    printf("Options per sec.: %f\n", OPT_N / (time * 0.001));

    printf("main(): comparing Monte Carlo and Black-Scholes results...\n");
    sumDelta = 0;
    sumRef = 0;
    sumReserve = 0;

    for (i = 0; i < OPT_N; i++) {
        BlackScholesCall(callValueBS[i], optionData[i]);
        delta = fabs(callValueBS[i] - callValueGPU[i].Expected);
        ref = callValueBS[i];
        sumDelta += delta;
        sumRef += fabs(ref);

        if (delta > 1e-6) {
            sumReserve += callValueGPU[i].Confidence / delta;
        }
        //printf("BS: %f; delta: %E\n", callValueBS[i], delta);
    }

    sumReserve /= OPT_N;

    printf("main(): running CPU MonteCarlo...\n");
    TOptionValue callValueCPU;
    sumDelta = 0;
    sumRef = 0;

    for (i = 0; i < OPT_N; i++) {
        MonteCarloCPU(
            callValueCPU,
            optionData[i],
            NULL,
            PATH_N);
        delta = fabs(callValueCPU.Expected - callValueGPU[i].Expected);
        ref = callValueCPU.Expected;
        sumDelta += delta;
        sumRef += fabs(ref);
        //printf("Exp : %f | %f\t", callValueCPU.Expected, callValueGPU[i].Expected);
        //printf("Conf: %f | %f\n", callValueCPU.Confidence, callValueGPU[i].Confidence);
    }

    printf("L1 norm: %E\n", sumDelta / sumRef);

    printf("Shutting down...\n");

    for (int i = 0; i < GPU_N; i++) {
        sdkStartTimer(&hTimer[i]);
        checkCudaErrors(cudaSetDevice(i));
    }

    delete[] optionSolver;
    delete[] callValueBS;
    delete[] callValueGPU;
    delete[] optionData;
    delete[] hTimer;

    printf("Test Summary...\n");
    printf("L1 norm        : %E\n", sumDelta / sumRef);
    printf("Average reserve: %f\n", sumReserve);
    printf(sumReserve > 1.0f ? "Test passed\n" : "Test failed!\n");
    exit(sumReserve > 1.0f ? EXIT_SUCCESS : EXIT_FAILURE);
}
