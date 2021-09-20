package com.nvidia.grcuda.test.util;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;

public class GrCUDATestOptionsStruct {
    public final ExecutionPolicyEnum policy;
    public final boolean inputPrefetch;
    public final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    public final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy;
    public final DependencyPolicyEnum dependencyPolicy;
    public final boolean forceStreamAttach;

    /**
     * A simple struct that holds a combination of GrCUDA options, extracted from the output of {@link GrCUDATestUtil#getAllOptionCombinations}
     */
    public GrCUDATestOptionsStruct(ExecutionPolicyEnum policy,
                                   boolean inputPrefetch,
                                   RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy,
                                   RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy,
                                   DependencyPolicyEnum dependencyPolicy,
                                   boolean forceStreamAttach) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        this.retrieveParentStreamPolicy = retrieveParentStreamPolicy;
        this.dependencyPolicy = dependencyPolicy;
        this.forceStreamAttach = forceStreamAttach;
    }
}