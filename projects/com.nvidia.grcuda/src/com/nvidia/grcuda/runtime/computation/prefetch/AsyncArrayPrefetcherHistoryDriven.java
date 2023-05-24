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
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;

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
        List<ExecutionDAG.DAGEdge> parents = computation.getVertex().getParents();
        CUDAStream streamToPrefetch = computation.getStream();
        // order edge by start vertex predictionTime
        parents.sort(Comparator.comparingDouble(edge -> edge.getStart().getComputation().getPredictionTime()));
        // prefetch arrays that create dependencies
        for (ExecutionDAG.DAGEdge p : parents){
            GrCUDAComputationalElement parentComputation = p.getStart().getComputation();
            // if parent and child are in the same gpu we don't have to prefetch
            if(parentComputation.getStream().getStreamDeviceId() != computation.getStream().getStreamDeviceId()){
                // before prefetching we should wait the parent's computation end
                runtime.cudaStreamWaitEvent(streamToPrefetch, parentComputation.getEventStop().get());
                // prefetch AbstractArrays which create dependencies
                for (ComputationArgumentWithValue a : p.getDependencies()){
                    if (a.getArgumentValue() instanceof AbstractArray) {
                        AbstractArray array = (AbstractArray) a.getArgumentValue();
                        // control if array is updated on parent device, maybe we might avoid it
                        if (array.isArrayUpdatedInLocation(parentComputation.getStream().getStreamDeviceId())) {
                            runtime.cudaMemPrefetchAsync(array, streamToPrefetch);
                        }
                    }
                }
            }
        }
    }
}
