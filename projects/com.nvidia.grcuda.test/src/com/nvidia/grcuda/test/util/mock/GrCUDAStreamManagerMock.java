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

import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.GrCUDAStreamManager;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class GrCUDAStreamManagerMock extends GrCUDAStreamManager {

    /**
     * Static number of fake GPU streams, as we don't have access to the CUDA runtime in the mocked class hierarchy;
     */
    public static int numUserAllocatedStreams = 0;

    GrCUDAStreamManagerMock(CUDARuntime runtime,
                            RetrieveNewStreamPolicyEnum retrieveStreamPolicy,
                            RetrieveParentStreamPolicyEnum parentStreamPolicy,
                            DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
                            String bandwidthMatrixPath,
                            int numberOfAvailableGPUs,
                            int numberOfGPUsToUse) {
        // Possibly use a number of GPUs lower than the number of available GPUs;
        super(runtime, false, new GrCUDAStreamPolicyMock(retrieveStreamPolicy, parentStreamPolicy, deviceSelectionPolicyEnum, bandwidthMatrixPath, numberOfAvailableGPUs, numberOfGPUsToUse));
        // Reset the number of streams;
        numUserAllocatedStreams = 0;
    }

    @Override
    public void assignEventStart(ExecutionDAG.DAGVertex vertex) { }

    @Override
    public void assignEventStop(ExecutionDAG.DAGVertex vertex) { }

    @Override
    public void syncStream(CUDAStream stream) { }

    @Override
    protected void setComputationFinishedInner(GrCUDAComputationalElement computation) {
        computation.setComputationFinished();
    }

    @Override
    protected void syncStreamsUsingEvents(ExecutionDAG.DAGVertex vertex) { }

    @Override
    protected void syncDevice() { }

    /**
     * Return all the streams allocated on every device;
     * @return all the streams that have been created
     */
    public List<CUDAStream> getStreams() {
        List<CUDAStream> allStreams = new ArrayList<>();
        this.getStreamPolicy().getDevicesManager().getDeviceList().forEach(d -> allStreams.addAll(d.getStreams()));
        return allStreams;
    }

    public Map<CUDAStream, Set<GrCUDAComputationalElement>> getActiveComputationsMap() {
        Map<CUDAStream, Set<GrCUDAComputationalElement>> activeComputations = new HashMap<>();
        for (Map.Entry<CUDAStream, Set<ExecutionDAG.DAGVertex>> e : this.activeComputationsPerStream.entrySet()) {
            activeComputations.put(e.getKey(), e.getValue().stream().map(ExecutionDAG.DAGVertex::getComputation).collect(Collectors.toSet()));
        }
        return activeComputations;
    }
}
