/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
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
package com.nvidia.grcuda.runtime.executioncontext;

import com.nvidia.grcuda.Binding;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.Kernel;
import com.nvidia.grcuda.runtime.computation.streamattach.StreamAttachArchitecturePolicy;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.dependency.DefaultDependencyComputationBuilder;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyComputationBuilder;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.dependency.WithConstDependencyComputationBuilder;
import com.nvidia.grcuda.runtime.computation.prefetch.AbstractArrayPrefetcher;
import com.nvidia.grcuda.runtime.computation.prefetch.DefaultArrayPrefetcher;
import com.nvidia.grcuda.runtime.computation.prefetch.NoneArrayPrefetcher;
import com.nvidia.grcuda.runtime.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.runtime.computation.prefetch.SyncArrayPrefetcher;
import com.oracle.truffle.api.TruffleLanguage;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.HashSet;
import java.util.Set;

/**
 * Abstract class that defines how {@link GrCUDAComputationalElement} are registered and scheduled for execution.
 * It monitor sthe state of GrCUDA execution, keep track of memory allocated,
 * kernels and other executable functions, and dependencies between elements.
 */
public abstract class AbstractGrCUDAExecutionContext {

    /**
     * Reference to the inner {@link CUDARuntime} used to execute kernels and other {@link GrCUDAComputationalElement}
     */
    protected final CUDARuntime cudaRuntime;

    /**
     * Set that contains all the arrays allocated so far.
     */
    protected final Set<AbstractArray> arraySet = new HashSet<>();

    /**
     * Set that contains all the CUDA kernels declared so far.
     */
    protected final Set<Kernel> kernelSet = new HashSet<>();

    /**
     * Reference to the computational DAG that represents dependencies between computations;
     */
    protected final ExecutionDAG dag;

    /**
     * Reference to how dependencies between computational elements are computed within this execution context;
     */
    private final DependencyComputationBuilder dependencyBuilder;
    /**
     * Identify the policy name associated to this execution context;
     */
    private final ExecutionPolicyEnum executionPolicy;

    /**
     * Reference to the prefetching strategy to use in this execution context;
     */
    protected final AbstractArrayPrefetcher arrayPrefetcher;

    public AbstractGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy, ExecutionPolicyEnum executionPolicy) {
        this(new CUDARuntime(context, env), dependencyPolicy, PrefetcherEnum.NONE, executionPolicy);
    }

    public AbstractGrCUDAExecutionContext(GrCUDAContext context, TruffleLanguage.Env env, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch, ExecutionPolicyEnum executionPolicy) {
        this(new CUDARuntime(context, env), dependencyPolicy, inputPrefetch, executionPolicy);
    }

    public AbstractGrCUDAExecutionContext(CUDARuntime cudaRuntime, DependencyPolicyEnum dependencyPolicy, PrefetcherEnum inputPrefetch, ExecutionPolicyEnum executionPolicy) {
        this.cudaRuntime = cudaRuntime;
        this.executionPolicy = executionPolicy;
        // Compute the dependency policy to use;
        switch (dependencyPolicy) {
            case WITH_CONST:
                this.dependencyBuilder = new WithConstDependencyComputationBuilder();
                break;
            case NO_CONST:
                this.dependencyBuilder = new DefaultDependencyComputationBuilder();
                break;
            default:
                this.dependencyBuilder = new DefaultDependencyComputationBuilder();
        }
        // Compute the prefetcher to use;
        boolean pascalGpu;
        switch (inputPrefetch) {
            case ASYNC:
                pascalGpu = this.cudaRuntime.isArchitectureIsPascalOrNewer();
                arrayPrefetcher = pascalGpu ? new DefaultArrayPrefetcher(this.cudaRuntime) : new NoneArrayPrefetcher(this.cudaRuntime);
                break;
            case SYNC:
                pascalGpu = this.cudaRuntime.isArchitectureIsPascalOrNewer();
                arrayPrefetcher = pascalGpu ? new SyncArrayPrefetcher(this.cudaRuntime) : new NoneArrayPrefetcher(this.cudaRuntime);
                break;
            default:
                arrayPrefetcher = new NoneArrayPrefetcher(this.cudaRuntime);
        }
        this.dag = new ExecutionDAG(dependencyPolicy);
    }

    /**
     * Register this computation for future execution by the {@link AbstractGrCUDAExecutionContext},
     * and add it to the current computational DAG.
     * The actual execution might be deferred depending on the inferred data dependencies;
     */
    abstract public Object registerExecution(GrCUDAComputationalElement computation) throws UnsupportedTypeException;

    public void registerArray(AbstractArray array) {
        arraySet.add(array);
    }

    public void registerKernel(Kernel kernel) {
        kernelSet.add(kernel);
    }

    public ExecutionDAG getDag() {
        return dag;
    }

    public CUDARuntime getCudaRuntime() {
        return cudaRuntime;
    }

    public DependencyComputationBuilder getDependencyBuilder() {
        return dependencyBuilder;
    }

    public ExecutionPolicyEnum getExecutionPolicy() {
        return executionPolicy;
    }

    // Functions used to interface directly with the CUDA runtime;

    public Kernel loadKernel(Binding binding) {
        return cudaRuntime.loadKernel(this, binding);
    }

    public Kernel buildKernel(String code, String kernelName, String signature) {
        return cudaRuntime.buildKernel(this, code, kernelName, signature);
    }

    public StreamAttachArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return cudaRuntime.getArrayStreamArchitecturePolicy();
    }

    /**
     * Check if any computation is currently marked as active, and is running on a stream managed by this context.
     * If so, scheduling of new computations is likely to require synchronizations of some sort;
     * @return if any computation is considered active on a stream managed by this context
     */
    public abstract boolean isAnyComputationActive();

    /**
     * Delete internal structures that require manual cleanup operations;
     */
    public void cleanup() { }
}