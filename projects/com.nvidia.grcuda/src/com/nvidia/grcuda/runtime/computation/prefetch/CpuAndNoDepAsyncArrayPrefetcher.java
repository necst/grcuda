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
package com.nvidia.grcuda.runtime.computation.prefetch;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.CUDARuntime;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.stream.CUDAStream;

import java.util.List;
import java.util.stream.Collectors;

public class CpuAndNoDepAsyncArrayPrefetcher extends AbstractArrayPrefetcher {

    public CpuAndNoDepAsyncArrayPrefetcher(CUDARuntime runtime) {
        super(runtime);
    }

    /**
     * The default array prefetcher schedules asynchronous prefetching on the arrays used by the computation.
     * Only the arrays which don't create dependencies are prefetched.
     * The prefetcher assumes that the GPU allows prefetching (architecture since Pascal) and the arrays are visible to the stream where they are prefetched.
     * Technically, we need prefetching only if the array has been modified by the CPU, and we could prefetch only the part that has been modified;
     * this simple prefetcher still prefetches everything though.
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    @Override
    public void prefetchToGpu(GrCUDAComputationalElement computation) {
        List<AbstractArray> argumentsThatCanCreateDependencies =  computation.getArrayArguments();
        List<ExecutionDAG.DAGEdge> parents = computation.getVertex().getParents();
        // List of all Array dependencies created by computation
        List<AbstractArray> argumentsThatCreateDependencies = parents.stream()
                .flatMap(parent -> parent.getDependencies().stream())
                .map(ComputationArgumentWithValue::getArgumentValue)
                .filter(AbstractArray.class::isInstance)
                .map(AbstractArray.class::cast)
                .collect(Collectors.toList());
        CUDAStream streamToPrefetch = computation.getStream();

        for (AbstractArray array : argumentsThatCanCreateDependencies) {
            // The array has been used by the CPU, so we could prefetch it;
            if (array.isArrayUpdatedOnCPU()) {
                runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                // Add the new array location
                array.addArrayUpToDateLocations(streamToPrefetch.getStreamDeviceId());
            }
            // We don't have to waite any parents, so we could prefetch it;
            else if(!argumentsThatCreateDependencies.contains(array) && !array.isArrayUpdatedInLocation(streamToPrefetch.getStreamDeviceId())){
                runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                // Add the new array location
                array.addArrayUpToDateLocations(streamToPrefetch.getStreamDeviceId());
            }
        }
    }
}
