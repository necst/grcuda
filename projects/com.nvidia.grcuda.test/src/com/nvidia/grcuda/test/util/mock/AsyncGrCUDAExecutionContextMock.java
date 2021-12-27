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
package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.computation.streamattach.PostPascalStreamAttachPolicy;
import com.nvidia.grcuda.runtime.computation.streamattach.StreamAttachArchitecturePolicy;
import com.nvidia.grcuda.runtime.computation.streamattach.PrePascalStreamAttachPolicy;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.computation.prefetch.PrefetcherEnum;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;

/**
 * Mock class to test the GrCUDAExecutionContextTest, it has a null CUDARuntime;
 */
public class AsyncGrCUDAExecutionContextMock extends AsyncGrCUDAExecutionContext {

    // Store it here to avoid using a mocked runtime;
    private final boolean architectureIsPascalOrNewer;

    public void setCurrentGPU(int gpu) {
        this.getStreamManager().getStreamPolicy().getDevicesManager().setCurrentGPU(gpu);
    }

    public int getCurrentGPU() {
        return this.getStreamManager().getStreamPolicy().getDevicesManager().getCurrentGPU().getDeviceId();
    }

    public AsyncGrCUDAExecutionContextMock() {
        super(null,
                new GrCUDAOptionMap(new OptionValuesMockBuilder()
                        .add(GrCUDAOptions.DependencyPolicy, DependencyPolicyEnum.NO_CONST.toString())
                        .add(GrCUDAOptions.InputPrefetch, false).build()),
                new GrCUDAStreamManagerMock(null));
        this.architectureIsPascalOrNewer = true;
    }

    public AsyncGrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy) {
        super(null,
                new GrCUDAOptionMap(new OptionValuesMockBuilder()
                        .add(GrCUDAOptions.DependencyPolicy, dependencyPolicy.toString())
                        .add(GrCUDAOptions.InputPrefetch, false).build()),
                new GrCUDAStreamManagerMock(null));
        this.architectureIsPascalOrNewer = true;
    }

    public AsyncGrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy,
                                           RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                                           RetrieveParentStreamPolicyEnum parentStreamPolicyEnum) {
        this(dependencyPolicy, retrieveStreamPolicy, parentStreamPolicyEnum, true, 1, 1);
    }

    public AsyncGrCUDAExecutionContextMock(DependencyPolicyEnum dependencyPolicy,
                                           RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                                           RetrieveParentStreamPolicyEnum parentStreamPolicyEnum,
                                           boolean architectureIsPascalOrNewer,
                                           int numberOfAvailableGPUs,
                                           int numberOfGPUsToUse) {
        super(null,
                new GrCUDAOptionMap(new OptionValuesMockBuilder()
                        .add(GrCUDAOptions.DependencyPolicy, dependencyPolicy.toString())
                        .add(GrCUDAOptions.InputPrefetch, false).build()),
                new GrCUDAStreamManagerMock(null, retrieveStreamPolicy, parentStreamPolicyEnum, numberOfAvailableGPUs, numberOfGPUsToUse));
        this.architectureIsPascalOrNewer = architectureIsPascalOrNewer;
    }

    public StreamAttachArchitecturePolicy getArrayStreamArchitecturePolicy() {
        return architectureIsPascalOrNewer ? new PrePascalStreamAttachPolicy() : new PostPascalStreamAttachPolicy();
    }

    @Override
    public boolean isArchitecturePascalOrNewer() {
        return architectureIsPascalOrNewer;
    }
}