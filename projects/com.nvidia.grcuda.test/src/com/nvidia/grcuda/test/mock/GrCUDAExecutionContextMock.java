package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.runtime.computation.streamassociation.ArrayStreamArchitecturePolicy;
import com.nvidia.grcuda.runtime.computation.streamassociation.PrePascalArrayStreamAssociation;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.runtime.executioncontext.GrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class GrCUDAExecutionContextMock extends GrCUDAExecutionContext {

    public GrCUDAExecutionContextMock() {
        super(null, null,
                new GrCUDAStreamManagerMock(null), DependencyPolicyEnum.NO_CONST, PrefetcherEnum.NONE);
    }

    public GrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy) {
        super(null, null,
                new GrCUDAStreamManagerMock(null), dependencyPolicy, PrefetcherEnum.NONE);
    }

    public GrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy,
                                      RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                                      RetrieveParentStreamPolicyEnum parentStreamPolicyEnum) {
        super(null, null,
                new GrCUDAStreamManagerMock(null, retrieveStreamPolicy, parentStreamPolicyEnum), dependencyPolicy, PrefetcherEnum.NONE);
    }

    public ArrayStreamArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return new PrePascalArrayStreamAssociation();
    }
}
