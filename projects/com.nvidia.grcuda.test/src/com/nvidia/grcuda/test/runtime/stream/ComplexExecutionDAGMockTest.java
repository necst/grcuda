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
package com.nvidia.grcuda.test.runtime.stream;

import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.GrCUDAStreamManagerMock;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.util.mock.SyncExecutionMock;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.executeMockComputation;
import static com.nvidia.grcuda.test.util.mock.GrCUDAComputationsMock.imageMockComputation;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(Parameterized.class)
public class ComplexExecutionDAGMockTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {

        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {RetrieveNewStreamPolicyEnum.ALWAYS_NEW, RetrieveNewStreamPolicyEnum.REUSE},
                {RetrieveParentStreamPolicyEnum.DISJOINT, RetrieveParentStreamPolicyEnum.SAME_AS_PARENT},
                {DependencyPolicyEnum.WITH_CONST, DependencyPolicyEnum.NO_CONST}
        }));
    }

    private final RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy;
    private final RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy;
    private final DependencyPolicyEnum dependencyPolicy;

    public ComplexExecutionDAGMockTest(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicy,
                                       RetrieveParentStreamPolicyEnum retrieveParentStreamPolicy,
                                       DependencyPolicyEnum dependencyPolicy) {
        this.retrieveNewStreamPolicy = retrieveNewStreamPolicy;
        this.retrieveParentStreamPolicy = retrieveParentStreamPolicy;
        this.dependencyPolicy = dependencyPolicy;
    }

    @Test
    public void hitsMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy).setRetrieveParentStreamPolicy(this.retrieveParentStreamPolicy)
                .setDependencyPolicy(this.dependencyPolicy).build();

        int numIterations = 10;
        KernelExecutionMock c1 = null;
        KernelExecutionMock c2 = null;
        for (int i = 0; i < numIterations; i++) {
            // hub1 -> auth2
            c1 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1, true), new ArgumentMock(2)));
            c1.schedule();
            // auth1 -> hub2
            c2 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(3, true), new ArgumentMock(4)));
            c2.schedule();

            // Without disjoint policy the computation collapses on a single stream after the first iteration;
            int stream = (retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) || i == 0) ? 0 : 1;
            assertEquals(stream, c1.getStream().getStreamNumber());
            assertEquals(1, c2.getStream().getStreamNumber());

            // auth2 -> auth_norm
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(2, true), new ArgumentMock(5))).schedule();
            // hub2 -> hub_norm
            new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4, true), new ArgumentMock(6))).schedule();
            // auth2, auth_norm -> auth1
            c1 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(2, true), new ArgumentMock(5, true), new ArgumentMock(3)));
            c1.schedule();
            // hub2, hub_norm -> hub1
            c2 = new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(4, true), new ArgumentMock(6, true), new ArgumentMock(1)));
            c2.schedule();
        }

        assertEquals(2, context.getStreamManager().getNumberOfStreams());

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();
        assertTrue(context.getStreamManager().isStreamFree(c1.getStream()));
        int activeComps = retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) ? 2 : 0;
        assertEquals(activeComps, context.getStreamManager().getNumActiveComputationsOnStream(c2.getStream()));

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        assertTrue(context.getStreamManager().isStreamFree(c1.getStream()));
        assertTrue(context.getStreamManager().isStreamFree(c2.getStream()));
    }

    @Test
    public void imageMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setRetrieveNewStreamPolicy(this.retrieveNewStreamPolicy).setRetrieveParentStreamPolicy(this.retrieveParentStreamPolicy)
                .setDependencyPolicy(this.dependencyPolicy).build();
        executeMockComputation(imageMockComputation(context));

        int numStreams = 3;
        if (retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.DISJOINT) && dependencyPolicy.equals(DependencyPolicyEnum.WITH_CONST)) {
            numStreams = 4;
        }
        else if (retrieveParentStreamPolicy.equals(RetrieveParentStreamPolicyEnum.SAME_AS_PARENT) && dependencyPolicy.equals(DependencyPolicyEnum.NO_CONST)) {
            numStreams = 1;
        }
        assertEquals(numStreams, context.getStreamManager().getNumberOfStreams());
        for (CUDAStream stream : ((GrCUDAStreamManagerMock) context.getStreamManager()).getStreams()) {
            assertTrue(context.getStreamManager().isStreamFree(stream));
        }
    }
}
