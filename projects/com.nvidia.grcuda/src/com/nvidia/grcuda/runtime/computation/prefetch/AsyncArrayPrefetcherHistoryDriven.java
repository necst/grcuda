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
import com.nvidia.grcuda.runtime.stream.CUDAStream;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.Map;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;

public class AsyncArrayPrefetcherHistoryDriven extends AbstractArrayPrefetcher {

    public AsyncArrayPrefetcherHistoryDriven(CUDARuntime runtime) {
        super(runtime);
    }

    /**
     * The default array prefetcher schedules asynchronous prefetching on the arrays used by the computation.
     * Only the arrays whose last operation has been a CPU access are prefetched, as the other are already up-to-date on GPU.
     * The prefetcher assumes that the GPU allows prefetching (architecture since Pascal) and the arrays are visible to the stream where they are prefetched.
     * Technically, we need prefetching only if the array has been modified by the CPU, and we could prefetch only the part that has been modified;
     * this simple prefetcher still prefetches everything though.
     * @param computation a computational element whose array inputs can be prefetched from host to GPU
     */
    @Override
    public void prefetchToGpu(GrCUDAComputationalElement computation) {

        List<ComputationArgumentWithValue> argumentsThatCanCreateDependencies =  computation.getArgumentsThatCanCreateDependencies();
        List<GrCUDAComputationalElement> parents = computation.getParentVertices().stream().map(ExecutionDAG.DAGVertex::getComputation).collect(Collectors.toList());
        // Partition the `parents` list into two lists based on the condition `isComputationFinished()`
        Map<Boolean, List<GrCUDAComputationalElement>> partitionedParents = parents.stream()
                .collect(Collectors.partitioningBy(GrCUDAComputationalElement::isComputationFinished));
        // Retrieve the lists from the partitioned map
        List<GrCUDAComputationalElement> finishedParents = partitionedParents.get(true);
        List<GrCUDAComputationalElement> unfinishedParents = partitionedParents.get(false);
        List<AbstractArray> arraysThatCanCreateDependencies = new ArrayList<>();
        CUDAStream streamToPrefetch = computation.getStream();
        // first of all, prefetch arrays updated on cpu and creating a list of arrays which aren't on CPU
        for (ComputationArgumentWithValue a : argumentsThatCanCreateDependencies) {
            if (a.getArgumentValue() instanceof AbstractArray) {
                AbstractArray array = (AbstractArray) a.getArgumentValue();
                // The array has been used by the CPU, so we should prefetch it;
                if (array.isArrayUpdatedOnCPU()) {
                    runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                }
                else{
                    arraysThatCanCreateDependencies.add(array);
                }
            }
        }

        // prefetch arrays whose related computation is finished
        for (GrCUDAComputationalElement p : finishedParents){
            // if computation and parent are on the same GPU, we shouldn't do anything
            if (computation.getStream().getStreamDeviceId() != p.getStream().getStreamDeviceId()){
                for (AbstractArray array : p.getArrayArguments()){
                    if(arraysThatCanCreateDependencies.contains(array) && array.isArrayUpdatedInLocation(p.getStream().getStreamDeviceId())){
                        runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                        // we remove the array to avoid multiple prefetch
                        arraysThatCanCreateDependencies.remove(array);
                    }
                }
            }
        }

        // order unfinished parents by predictionTime
        Collections.sort(unfinishedParents, Comparator.comparing(GrCUDAComputationalElement::getPredictionTime));

        // prefetch arrays whose related computation is finished but waiting for the end of the parent
        for (GrCUDAComputationalElement p : unfinishedParents){
            // if computation and parent are on the same GPU, we shouldn't do anything
            if (computation.getStream().getStreamDeviceId() != p.getStream().getStreamDeviceId()){
                runtime.cudaStreamWaitEvent(streamToPrefetch, p.getEventStop().get());
                for (AbstractArray array : p.getArrayArguments()){
                    if(arraysThatCanCreateDependencies.contains(array)){
                        runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                        // we remove the array to avoid multiple prefetch
                        arraysThatCanCreateDependencies.remove(array);
                    }
                }
            }
        }
    }
}
