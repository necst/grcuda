/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda;

import com.nvidia.grcuda.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cuml.CUMLRegistry;
import com.nvidia.grcuda.functions.BindAllFunction;
import com.nvidia.grcuda.functions.BindFunction;
import com.nvidia.grcuda.functions.BindKernelFunction;
import com.nvidia.grcuda.functions.BuildKernelFunction;
import com.nvidia.grcuda.functions.DeviceArrayFunction;
import com.nvidia.grcuda.functions.GetDeviceFunction;
import com.nvidia.grcuda.functions.GetDevicesFunction;
import com.nvidia.grcuda.functions.map.MapFunction;
import com.nvidia.grcuda.functions.map.ShredFunction;
import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.gpu.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.gpu.computation.memAdvise.AdviserEnum;
import com.nvidia.grcuda.gpu.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.gpu.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.executioncontext.SyncGrCUDAExecutionContext;
import com.nvidia.grcuda.gpu.stream.ChooseDeviceHeuristicEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.tensorrt.TensorRTRegistry;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.TruffleLanguage.Env;
import org.graalvm.options.OptionKey;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Context for the grCUDA language holds reference to CUDA runtime, a function registry and device
 * resources.
 */
public final class GrCUDAContext {

    public static final Integer DEFAULT_NUMBER_GPUs = 1;
    public static final ExecutionPolicyEnum DEFAULT_EXECUTION_POLICY = ExecutionPolicyEnum.DEFAULT;
    public static final DependencyPolicyEnum DEFAULT_DEPENDENCY_POLICY = DependencyPolicyEnum.DEFAULT;
    public static final RetrieveNewStreamPolicyEnum DEFAULT_RETRIEVE_STREAM_POLICY = RetrieveNewStreamPolicyEnum.ALWAYS_NEW;
    public static final RetrieveParentStreamPolicyEnum DEFAULT_PARENT_STREAM_POLICY = RetrieveParentStreamPolicyEnum.DATA_AWARE;
    public static final ChooseDeviceHeuristicEnum DEFAULT_CHOOSE_DEVICE_HEURISTIC = ChooseDeviceHeuristicEnum.DATA_LOCALITY;
    public static final AdviserEnum DEFAULT_MEM_ADVISE = AdviserEnum.NONE;
    public static final PrefetcherEnum DEFAULT_PREFETCHER = PrefetcherEnum.NONE;
    public static final boolean DEFAULT_FORCE_STREAM_ATTACH = false;

    private static final String ROOT_NAMESPACE = "CU";

    private final Env env;
    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;
    private final Namespace rootNamespace;
    private final ArrayList<Runnable> disposables = new ArrayList<>();
    private final AtomicInteger moduleId = new AtomicInteger(0);
    private volatile boolean cudaInitialized = false;
    private final ExecutionPolicyEnum executionPolicy;
    private final DependencyPolicyEnum dependencyPolicy;
    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum;
    private final ChooseDeviceHeuristicEnum chooseDeviceHeuristicEnum;
    private final AdviserEnum memAdvise;
    private final PrefetcherEnum inputPrefetch;
    private final boolean forceStreamAttach;
    private final int numberOfGPUs;
    private final boolean timeComputation;
    // this is used to look up pre-existing call targets for "map" operations, see MapArrayNode
    private final ConcurrentHashMap<Class<?>, CallTarget> uncachedMapCallTargets = new ConcurrentHashMap<>();

    public GrCUDAContext(Env env) {
        this.env = env;

        // Retrieve the stream retrieval policy;
        retrieveNewStreamPolicy = parseRetrieveStreamPolicy(env.getOptions().get(GrCUDAOptions.RetrieveNewStreamPolicy));
        
        // Retrieve how streams are obtained from parent computations;
        retrieveParentStreamPolicyEnum = parseParentStreamPolicy(env.getOptions().get(GrCUDAOptions.RetrieveParentStreamPolicy));

        // Retrieve the number of GPUs to be used;
        numberOfGPUs = Integer.parseInt(env.getOptions().get(GrCUDAOptions.NumberOfGPUs));

        // Retrieve how streams are obtained from parent computations;
        chooseDeviceHeuristicEnum = parseChooseDeviceHeuristic(env.getOptions().get(GrCUDAOptions.ChooseDeviceHeuristic));

        // Retrieve if we should time the computation;
        timeComputation = env.getOptions().get(GrCUDAOptions.TimeComputation);

        // Retrieve if we should force array stream attachment;
        forceStreamAttach = env.getOptions().get(GrCUDAOptions.ForceStreamAttach);

        // Retrieve if we should prefetch input data to GPU;
        inputPrefetch = parsePrefetcher(env.getOptions().get(GrCUDAOptions.InputPrefetch),env.getOptions().get(GrCUDAOptions.ExecutionPolicy));

        // Retrieve if we should implement memAdvise function during computation;
        memAdvise = parseMemAdvise(env.getOptions().get(GrCUDAOptions.memAdviseOption));

        // Retrieve the dependency computation policy;
        dependencyPolicy = parseDependencyPolicy(env.getOptions().get(GrCUDAOptions.DependencyPolicy));
        // System.out.println("-- using " + dependencyPolicy.getName() + " dependency policy");

        // Retrieve the execution policy;
        executionPolicy = parseExecutionPolicy(env.getOptions().get(GrCUDAOptions.ExecutionPolicy));
        // Initialize the execution policy;
        // System.out.println("-- using " + executionPolicy.getName() + " execution policy");

        switch (executionPolicy) {
            case SYNC:
                this.grCUDAExecutionContext = new SyncGrCUDAExecutionContext(this, env, dependencyPolicy, inputPrefetch, memAdvise);
                break;
            case DEFAULT:
            default:
                this.grCUDAExecutionContext = new GrCUDAExecutionContext(this, env, dependencyPolicy, inputPrefetch, memAdvise);
        }


        Namespace namespace = new Namespace(ROOT_NAMESPACE);
        namespace.addNamespace(namespace);
        namespace.addFunction(new BindFunction());
        namespace.addFunction(new DeviceArrayFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new BindAllFunction(this));
        namespace.addFunction(new MapFunction());
        namespace.addFunction(new ShredFunction());
        namespace.addFunction(new BindKernelFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new BuildKernelFunction(this.grCUDAExecutionContext));
        namespace.addFunction(new GetDevicesFunction(this.grCUDAExecutionContext.getCudaRuntime()));
        namespace.addFunction(new GetDeviceFunction(this.grCUDAExecutionContext.getCudaRuntime()));
        this.grCUDAExecutionContext.getCudaRuntime().registerCUDAFunctions(namespace);
        if (this.getOption(GrCUDAOptions.CuMLEnabled)) {
            Namespace ml = new Namespace(CUMLRegistry.NAMESPACE);
            namespace.addNamespace(ml);
            new CUMLRegistry(this).registerCUMLFunctions(ml);
        }
        if (this.getOption(GrCUDAOptions.CuBLASEnabled)) {
            Namespace blas = new Namespace(CUBLASRegistry.NAMESPACE);
            namespace.addNamespace(blas);
            new CUBLASRegistry(this).registerCUBLASFunctions(blas);
        }
        if (this.getOption(GrCUDAOptions.TensorRTEnabled)) {
            Namespace trt = new Namespace(TensorRTRegistry.NAMESPACE);
            namespace.addNamespace(trt);
            new TensorRTRegistry(this).registerTensorRTFunctions(trt);
        }
        this.rootNamespace = namespace;
    }



    public Env getEnv() {
        return env;
    }

    public AbstractGrCUDAExecutionContext getGrCUDAExecutionContext() {
        return grCUDAExecutionContext;
    }

    public CUDARuntime getCUDARuntime() {
        return this.grCUDAExecutionContext.getCudaRuntime();
    }

    public Namespace getRootNamespace() {
        return rootNamespace;
    }

    public void addDisposable(Runnable disposable) {
        disposables.add(disposable);
    }

    public void disposeAll() {
        for (Runnable runnable : disposables) {
            runnable.run();
        }
    }

    public int getNextModuleId() {
        return moduleId.incrementAndGet();
    }

    public boolean isCUDAInitialized() {
        return cudaInitialized;
    }

    public void setCUDAInitialized() {
        cudaInitialized = true;
    }

    public ConcurrentHashMap<Class<?>, CallTarget> getMapCallTargets() {
        return uncachedMapCallTargets;
    }

    public RetrieveNewStreamPolicyEnum getRetrieveNewStreamPolicy() {
        return retrieveNewStreamPolicy;
    }

    public int getNumberOfGPUs(){
        return this.numberOfGPUs;
    }
    
    public RetrieveParentStreamPolicyEnum getRetrieveParentStreamPolicyEnum() {
        return retrieveParentStreamPolicyEnum;
    }

    public ChooseDeviceHeuristicEnum getChooseDeviceHeuristicEnum() { return chooseDeviceHeuristicEnum; }

    public boolean isForceStreamAttach() {
        return forceStreamAttach;
    }

    public boolean isTimeComputation(){
        return this.timeComputation;
    }

    /**
     * Compute the maximum number of concurrent threads that can be spawned by GrCUDA.
     * This value is usually smaller or equal than the number of logical CPU threads available on the machine.
     * @return the maximum number of concurrent threads that can be spawned by GrCUDA
     */
    public int getNumberOfThreads() {
        return Runtime.getRuntime().availableProcessors();
    }

    @TruffleBoundary
    public <T> T getOption(OptionKey<T> key) {
        return env.getOptions().get(key);
    }

    @TruffleBoundary
    private static ExecutionPolicyEnum parseExecutionPolicy(String policyString) {
        switch(policyString) {
            case "sync":
                return ExecutionPolicyEnum.SYNC;
            case "default":
                return ExecutionPolicyEnum.DEFAULT;
            default:
                System.out.println("Warning: unknown execution policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_EXECUTION_POLICY);
                return GrCUDAContext.DEFAULT_EXECUTION_POLICY;
        }
    }

    @TruffleBoundary
    private static AdviserEnum parseMemAdvise(String policyString) {
        switch(policyString) {
            case "read_mostly":
                return AdviserEnum.ADVISE_READ_MOSTLY;
            case "preferred":
                return AdviserEnum.ADVISE_PREFERRED_LOCATION;
            case "none":
            default:
                return AdviserEnum.NONE;
        }
    }

    @TruffleBoundary
    private static PrefetcherEnum parsePrefetcher(String policyString, String execPolicyString) {
        switch(policyString) {
            case "active":
                switch (execPolicyString){
                    case "sync":
                        return PrefetcherEnum.SYNC;
                    case "default":
                    default:
                        return PrefetcherEnum.DEFAULT;
                }
            case "none":
            default:
                return PrefetcherEnum.NONE;
        }
    }

    @TruffleBoundary
    private static DependencyPolicyEnum parseDependencyPolicy(String policyString) {
        switch(policyString) {
            case "with_const":
                return DependencyPolicyEnum.WITH_CONST;
            case "default":
                return DependencyPolicyEnum.DEFAULT;
            default:
                System.out.println("Warning: unknown dependency policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_DEPENDENCY_POLICY);
                return GrCUDAContext.DEFAULT_DEPENDENCY_POLICY;
        }
    }

    @TruffleBoundary
    private static RetrieveNewStreamPolicyEnum parseRetrieveStreamPolicy(String policyString) {
        switch(policyString) {
            case "fifo":
                return RetrieveNewStreamPolicyEnum.FIFO;
            case "always_new":
                return RetrieveNewStreamPolicyEnum.ALWAYS_NEW;
            default:
                System.out.println("Warning: unknown new stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY);
                return GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY;
        }
    }
    @TruffleBoundary
    private static RetrieveParentStreamPolicyEnum parseParentStreamPolicy(String policyString) {
        switch(policyString) {
            case "data_aware":
                return RetrieveParentStreamPolicyEnum.DATA_AWARE;
            case "stream_aware":
                return RetrieveParentStreamPolicyEnum.STREAM_AWARE;
            case "disjoint_data_aware":
                return RetrieveParentStreamPolicyEnum.DISJOINT_DATA_AWARE;
            case "disjoint":
                return RetrieveParentStreamPolicyEnum.DISJOINT;
            case "default":
                return RetrieveParentStreamPolicyEnum.DEFAULT;
            default:
                System.out.println("Warning: unknown parent stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY);
                return GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY;
        }
    }
    @TruffleBoundary
    private static ChooseDeviceHeuristicEnum parseChooseDeviceHeuristic (String policyString) {
        switch(policyString) {
            case "data_locality":
                return ChooseDeviceHeuristicEnum.DATA_LOCALITY;
            case "data_locality_new":
                return ChooseDeviceHeuristicEnum.DATA_LOCALITY_NEW;
            case "best_transfer_time_max":
                return ChooseDeviceHeuristicEnum.TRANSFER_TIME_MAX;
            case "best_transfer_time_min":
                return ChooseDeviceHeuristicEnum.TRANSFER_TIME_MIN;
            default:
                System.out.println("Warning: unknown heuristic for choosing devices=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_CHOOSE_DEVICE_HEURISTIC);
                return GrCUDAContext.DEFAULT_CHOOSE_DEVICE_HEURISTIC;
        }
    }


    /**
     * Cleanup the GrCUDA context at the end of the execution;
     */
    public void cleanup() {
        this.grCUDAExecutionContext.cleanup();
    }
}
