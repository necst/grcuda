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
package com.nvidia.grcuda.test.runtime;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.test.util.mock.ArgumentMock;
import com.nvidia.grcuda.test.util.mock.AsyncGrCUDAExecutionContextMock;
import com.nvidia.grcuda.test.util.mock.DeviceArrayMock;
import com.nvidia.grcuda.test.util.mock.GrCUDAExecutionContextMockBuilder;
import com.nvidia.grcuda.test.util.mock.KernelExecutionMock;
import com.nvidia.grcuda.test.util.mock.SyncExecutionMock;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ExecutionDAGMockTest {

    @Test
    public void executionDAGConstructorTest() {
        ExecutionDAG dag = new ExecutionDAG(DependencyPolicyEnum.NO_CONST);
        assertTrue(dag.getVertices().isEmpty());
        assertTrue(dag.getEdges().isEmpty());
        assertTrue(dag.getFrontier().isEmpty());
        assertEquals(0, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
    }

    @Test
    public void addVertexToDAGTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new AsyncGrCUDAExecutionContextMock();
        // Create two mock kernel executions;
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3))).schedule();

        ExecutionDAG dag = context.getDag();

        assertEquals(1, dag.getNumVertices());
        assertEquals(0, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        assertTrue(dag.getFrontier().get(0).isFrontier());
        assertTrue(dag.getFrontier().get(0).isStart());

        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3))).schedule();

        assertEquals(2, dag.getNumVertices());
        assertEquals(1, dag.getNumEdges());
        assertEquals(1, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(dag.getVertices().get(1), dag.getFrontier().get(0));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isStart());
        // Check if the first vertex is a parent of the second;
        assertEquals(dag.getVertices().get(0), dag.getVertices().get(1).getParentVertices().get(0));
        // Check if the second vertex is a child of the first;
        assertEquals(dag.getVertices().get(1), dag.getVertices().get(0).getChildVertices().get(0));
    }

    @Test
    public void dependencyPipelineSimpleMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new AsyncGrCUDAExecutionContextMock();
        // Create 4 mock kernel executions. In this case, kernel 3 requires 1 and 2 to finish,
        //   and kernel 4 requires kernel 3 to finish. The final frontier is composed of kernel 3 (arguments "1" and "2" are active),
        //   and kernel 4 (argument "3" is active);
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context,
                Arrays.asList(new ArgumentMock(1), new ArgumentMock(2), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(3))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(4, dag.getNumVertices());
        assertEquals(3, dag.getNumEdges());
        assertEquals(2, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3))),
                new HashSet<>(dag.getFrontier()));
        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertTrue(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        // Check if the third vertex is a child of first and second;
        assertEquals(2, dag.getVertices().get(2).getParents().size());
        assertEquals(new HashSet<>(dag.getVertices().get(2).getParentVertices()),
                new HashSet<>(Arrays.asList(dag.getVertices().get(0), dag.getVertices().get(1))));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(0).getChildVertices().get(0));
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(1).getChildVertices().get(0));
        // Check if the fourth vertex is a child of the third;
        assertEquals(1, dag.getVertices().get(3).getParents().size());
        assertEquals(1, dag.getVertices().get(2).getChildren().size());
        assertEquals(dag.getVertices().get(2), dag.getVertices().get(3).getParentVertices().get(0));
        assertEquals(dag.getVertices().get(3), dag.getVertices().get(2).getChildVertices().get(0));
    }

    @Test
    public void complexFrontierMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new AsyncGrCUDAExecutionContextMock();

        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //    \----> C(2)
        // The final frontier is composed by C(2), D(3), E(1), F(4);
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(4, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3), dag.getVertices().get(4), dag.getVertices().get(5))),
                new HashSet<>(dag.getFrontier()));

        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertTrue(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());
    }

    @Test
    public void complexFrontierWithSyncMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new AsyncGrCUDAExecutionContextMock(DependencyPolicyEnum.NO_CONST,
                RetrieveNewStreamPolicyEnum.REUSE, RetrieveParentStreamPolicyEnum.DISJOINT);

        // This time, simulate the synchronization process between kernels;
        // A(1,2) -> B(1) -> D(1,3) -> E(1,4) -> F(4)
        //   \-> C(2)
        // Synchronize C, then F
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(1))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(3))).schedule();
        new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(1), new ArgumentMock(4))).schedule();
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();

        ExecutionDAG dag = context.getDag();

        // Check the DAG structure;
        assertEquals(6, dag.getNumVertices());
        assertEquals(5, dag.getNumEdges());
        assertEquals(4, dag.getFrontier().size());
        // Check updates to frontier and start status;
        assertEquals(new HashSet<>(Arrays.asList(dag.getVertices().get(2), dag.getVertices().get(3),
                dag.getVertices().get(4), dag.getVertices().get(5))),
                new HashSet<>(dag.getFrontier()));

        assertFalse(dag.getVertices().get(0).isFrontier());
        assertTrue(dag.getVertices().get(0).isStart());
        assertFalse(dag.getVertices().get(1).isFrontier());
        assertFalse(dag.getVertices().get(1).isStart());
        assertTrue(dag.getVertices().get(2).isFrontier());
        assertFalse(dag.getVertices().get(2).isStart());
        assertTrue(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(3).isStart());
        assertTrue(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(4).isStart());
        assertTrue(dag.getVertices().get(5).isFrontier());
        assertFalse(dag.getVertices().get(5).isStart());

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(2))).schedule();
        assertEquals(3, dag.getFrontier().size());
        assertFalse(dag.getVertices().get(2).isFrontier());

        new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(4))).schedule();
        assertEquals(0, dag.getFrontier().size());
        assertFalse(dag.getVertices().get(3).isFrontier());
        assertFalse(dag.getVertices().get(4).isFrontier());
        assertFalse(dag.getVertices().get(5).isFrontier());
    }

    @Test
    public void concurrentReadMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setArchitecturePascalOrNewer(true).build();

        // This time, simulate a computation on the GPU, and a concurrent CPU read.
        // As the array is not modified, there should be no dependency between them.
        // However, we have to schedule the write to ensure that the GPU computation has finished before we update data;
        DeviceArrayMock x = new DeviceArrayMock(context);
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(x, true))).schedule();
        assertTrue(x.canSkipSchedulingRead());
        assertFalse(x.canSkipSchedulingWrite());
    }

    @Test
    public void concurrentReadMockTest2() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setArchitecturePascalOrNewer(false).build();
        // This time, simulate a computation on the GPU, and a concurrent CPU read & write.
        // As the GPU is pre-pascal, and we are running the kernel on the default stream, we must have a sync;
        DeviceArrayMock x = new DeviceArrayMock(context);
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(x, true))).schedule();
        assertFalse(x.canSkipSchedulingRead());
        assertFalse(x.canSkipSchedulingWrite());
    }

    @Test
    public void concurrentReadNoConstMockTest() throws UnsupportedTypeException {
        AsyncGrCUDAExecutionContext context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.NO_CONST)
                .setArchitecturePascalOrNewer(true).build();
        // This time, simulate a computation on the GPU, and a concurrent CPU read & write.
        // As we are not considering "const", there should be a dependency;
        DeviceArrayMock x = new DeviceArrayMock(context);
        new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(x, true))).schedule();
        assertFalse(x.canSkipSchedulingRead());
        assertFalse(x.canSkipSchedulingWrite());
    }

    // Test that if we have a kernel that uses an array read-only, and we schedule a write on CPU,
    // the scheduling of the write is not skipped and we have a dependency between the kernel and the write;
    @Test
    public void writeIsNotSkippedMockTest() throws UnsupportedTypeException, InvalidArrayIndexException {
        AsyncGrCUDAExecutionContextMock context = new GrCUDAExecutionContextMockBuilder()
                .setDependencyPolicy(DependencyPolicyEnum.WITH_CONST)
                .setRetrieveNewStreamPolicy(RetrieveNewStreamPolicyEnum.REUSE)
                .setRetrieveParentStreamPolicy(RetrieveParentStreamPolicyEnum.DISJOINT)
                .setArchitecturePascalOrNewer(true).build();

        DeviceArray array1 = new DeviceArrayMock(context);
        ExecutionDAG dag = context.getDag();
        // K1(const A1, A2);
        KernelExecutionMock k = new KernelExecutionMock(context, Collections.singletonList(new ComputationArgumentWithValue("array1", Type.NFI_POINTER, ComputationArgument.Kind.POINTER_IN, array1)));
        k.schedule();
        assertEquals(2, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedInLocation(0));
        assertTrue(array1.isArrayUpdatedOnCPU());
        // Write on the array;
        array1.writeArrayElement(0, 0, null, null);
        // Check that the array update status is tracked correctly;
        assertEquals(1, array1.getArrayUpToDateLocations().size());
        assertTrue(array1.isArrayUpdatedOnCPU());
        // Check the existence of a dependency;
        assertEquals(2, dag.getNumVertices());
        assertEquals(1, dag.getNumEdges());
    }
}
