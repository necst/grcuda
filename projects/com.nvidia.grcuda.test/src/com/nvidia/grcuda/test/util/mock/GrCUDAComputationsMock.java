package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class GrCUDAComputationsMock {

    /**
     * Schedule for execution a sequence of mock GrCUDAComputationalElement;
     */
    public static void executeMockComputation(List<GrCUDAComputationalElement> computations) throws UnsupportedTypeException {
        executeMockComputationAndValidateInner(computations, new ArrayList<>(), false, false);
    }

    /**
     * Schedule for execution a sequence of mock GrCUDAComputationalElement;
     */
    public static void executeMockComputation(List<GrCUDAComputationalElement> computations, boolean debug) throws UnsupportedTypeException {
        executeMockComputationAndValidateInner(computations, new ArrayList<>(), false, debug);
    }

    /**
     * Schedule for execution a sequence of mock GrCUDAComputationalElement,
     * and validate that the GPU scheduling of the computation is the one expected.
     * @param computations a sequence of computations to be scheduled
     * @param gpuScheduling a list of gpu identifiers. Each identifier "i" represents the GPU scheduling for the i-th computation;
     * @throws UnsupportedTypeException
     */
    public static void executeMockComputationAndValidate(List<GrCUDAComputationalElement> computations, List<Integer> gpuScheduling) throws UnsupportedTypeException {
        executeMockComputationAndValidate(computations, gpuScheduling, false);
    }

    /**
     * Schedule for execution a sequence of mock GrCUDAComputationalElement,
     * and validate that the GPU scheduling of the computation is the one expected.
     * @param computations a sequence of computations to be scheduled
     * @param gpuScheduling a list of gpu identifiers. Each identifier "i" represents the GPU scheduling for the i-th computation;
     * @param debug if true, print debug information about the scheduling
     * @throws UnsupportedTypeException
     */
    public static void executeMockComputationAndValidate(List<GrCUDAComputationalElement> computations, List<Integer> gpuScheduling, boolean debug) throws UnsupportedTypeException {
        executeMockComputationAndValidateInner(computations, gpuScheduling, true, debug);
    }

    private static void executeMockComputationAndValidateInner(
            List<GrCUDAComputationalElement> computations,
            List<Integer> gpuScheduling,
            boolean validate,
            boolean debug) throws UnsupportedTypeException {
        if (validate) {
            assertEquals(computations.size(), gpuScheduling.size());
        }
        for (int i = 0; i < computations.size(); i++) {
            GrCUDAComputationalElement c = computations.get(i);
            c.schedule();
            int actual = c.getStream().getStreamDeviceId();
            if (debug) {
                System.out.println(c);
            }
            if (validate) {
                int expected = gpuScheduling.get(i);
                if (expected != actual) {
                    System.out.println("wrong GPU allocation for kernel " + i + "=" + c + "; expected=" + expected + "; actual=" + actual);
                }
                assertEquals(expected, actual);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////
    // Simple GPU computations, to test standard DAG patterns (e.g. fork-join), //
    // and corner-cases;                                                        //
    //////////////////////////////////////////////////////////////////////////////

    // Simply schedule 10 kernels on independent data;
    public static List<GrCUDAComputationalElement> manyIndependentKernelsMockComputation(AsyncGrCUDAExecutionContext context) {
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10)))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(new DeviceArrayMock(10))))
        );
    }

    // (Ar) --> (A, B) --> (A, B, C) -> (A, B, D)
    // (Br) -/         /             /
    // (Cr) ----------/             /
    // (Dr) -----------------------/
    public static List<GrCUDAComputationalElement> joinPipelineMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b), new ArgumentMock(c))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b), new ArgumentMock(d)))
        );
    }

    // (Ar) --> (Ar, B) --> (A, B, C)
    // (Br) -/           /
    // (Cr) --> (C, D) -/
    // (Dr) -/
    public static List<GrCUDAComputationalElement> joinPipeline2MockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(c), new ArgumentMock(d))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b), new ArgumentMock(c)))
        );
    }

    // (Ar) --> (Ar, B) --> (B, C)
    // (Br) -/           /
    // (Cr) --> (C, D) -/
    // (Dr) -/
    public static List<GrCUDAComputationalElement> joinPipeline3MockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, false))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(c), new ArgumentMock(d))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(b), new ArgumentMock(c)))
        );
    }

    // (Ar) --> (Ar, B) -> (A, C, D) -> (A, C)
    // (Br) -/          /
    // (Cr) -----------/
    // (Dr) ----------/
    public static List<GrCUDAComputationalElement> joinPipeline4MockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(100);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c, true))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d, true))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(c), new ArgumentMock(d))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(c)))
        );
    }

    // (X) --> (Z) --> (A)
    // (Y) -/      \-> (B)
    public static List<GrCUDAComputationalElement> forkJoinMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a1 = new DeviceArrayMock(10);
        DeviceArrayMock a2 = new DeviceArrayMock(10);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a1))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a2))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1), new ArgumentMock(a2))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a1))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a2)))
        );
    }


    // K0 -> K4 -> K8 ---> K10
    // K1 -> K5 /     \--> K11
    // K2 -> K6 -> K9 -\-> K12
    // K3 -> K7 /------\-> K13
    public static List<GrCUDAComputationalElement> manyKernelsMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a = new DeviceArrayMock(10);
        DeviceArrayMock b = new DeviceArrayMock(10);
        DeviceArrayMock c = new DeviceArrayMock(10);
        DeviceArrayMock d = new DeviceArrayMock(10);
        return Arrays.asList(
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d))),

                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(b))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(c))),
                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(d))),

                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a), new ArgumentMock(b))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(c), new ArgumentMock(d))),

                new KernelExecutionMock(context, Collections.singletonList(new ArgumentMock(a, true))),
                // When using stream-aware and 4 GPUs, this is scheduled on device 2 (of 4) as device 1 has synced the computation on it (with K8),
                // and device 2 is the first device with fewer streams;
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(b))),
                // When using stream-aware and 4 GPUs, this is scheduled on device 3 (reuse the stream of K9);
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(c))),
                // When using stream-aware and 4 GPUs, this is scheduled on device 2 (device with fewer streams, device 1 has 2);
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a, true), new ArgumentMock(d)))
        );
    }

    ///////////////////////////////////////////////////////////////////////
    // Complex GPU benchmarks, inspired by the real benchmarks in GrCUDA //
    ///////////////////////////////////////////////////////////////////////

    // 0: K0(1c, 2) -> 2: K3(2c, 5) -> 4: K5(2c, 5c, 3) -> Repeat -> S(3)
    //              \--------------\/
    //              /--------------/\
    // 1: K1(3c, 4) -> 3: K4(4c, 6) -> 5: K6(4c, 6c, 1) -> Repeat -> S(1)
    public static List<GrCUDAComputationalElement> hitsMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a1 = new DeviceArrayMock(10);
        DeviceArrayMock a2 = new DeviceArrayMock(10);
        DeviceArrayMock a3 = new DeviceArrayMock(10);
        DeviceArrayMock a4 = new DeviceArrayMock(10);
        DeviceArrayMock a5 = new DeviceArrayMock(1);
        DeviceArrayMock a6 = new DeviceArrayMock(1);
        List<GrCUDAComputationalElement> computations = new ArrayList<>();
        int numIterations = 2;
        for (int i = 0; i < numIterations; i++) {
            // hub1 -> auth2
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1, true), new ArgumentMock(a2))));
            // auth1 -> hub2
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a3, true), new ArgumentMock(a4))));
            // auth2 -> auth_norm
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a2, true), new ArgumentMock(a5))));
            // hub2 -> hub_norm
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a4, true), new ArgumentMock(a6))));
            // auth2, auth_norm -> auth1
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a2, true), new ArgumentMock(a5, true), new ArgumentMock(a3))));
            // hub2, hub_norm -> hub1
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a4, true), new ArgumentMock(a6, true), new ArgumentMock(a1))));
        }
        computations.add(new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(a3))));
        computations.add(new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(a1))));
        return computations;
    }

    // 0: B1(1c, 2) -> 3: S1(2c, 5) -------------------------------------------------------------> 10: C(10c, 2c, 5c, 11) -> X
    // 1: B2(1c, 3) -> 4: S2(3c, 6) -------------------------------------> 9: C(9c, 3c, 6c, 10) /
    // 2: B3(1c, 4) -> 5: E1(4c, 7) -> 7: E3(4, 7, 8) -> 8: U(1c, 4, 9) /
    //             \-> 6: E2(4c, 8) /
    public static List<GrCUDAComputationalElement> imageMockComputation(AsyncGrCUDAExecutionContext context) {
        DeviceArrayMock a1 = new DeviceArrayMock(10);
        DeviceArrayMock a2 = new DeviceArrayMock(10);
        DeviceArrayMock a3 = new DeviceArrayMock(10);
        DeviceArrayMock a4 = new DeviceArrayMock(10);
        DeviceArrayMock a5 = new DeviceArrayMock(10);
        DeviceArrayMock a6 = new DeviceArrayMock(10);
        DeviceArrayMock a7 = new DeviceArrayMock(1);
        DeviceArrayMock a8 = new DeviceArrayMock(1);
        DeviceArrayMock a9 = new DeviceArrayMock(10);
        DeviceArrayMock a10 = new DeviceArrayMock(10);
        DeviceArrayMock a11 = new DeviceArrayMock(10);
        return Arrays.asList(
                // blur
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1, true), new ArgumentMock(a2))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1, true), new ArgumentMock(a3))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1, true), new ArgumentMock(a4))),
                // sobel
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a2, true), new ArgumentMock(a5))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a3, true), new ArgumentMock(a6))),
                // extend
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a4, true), new ArgumentMock(a7))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a4, true), new ArgumentMock(a8))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a4), new ArgumentMock(a7), new ArgumentMock(a8))),
                // unsharpen
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a1, true), new ArgumentMock(a4), new ArgumentMock(a9))),
                // combine
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a9, true), new ArgumentMock(a3, true),
                        new ArgumentMock(a6, true), new ArgumentMock(a10))),
                new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(a10, true), new ArgumentMock(a2, true),
                        new ArgumentMock(a5, true), new ArgumentMock(a11))),
                new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(a11)))
        );
    }

    public static final int partitionsVec = 16;

    // K1(Xr, X1) --> K3(X1r, Y1r, R)
    // K2(Yr, Y1) -/
    // A simple join pattern, with X, Y, X1, Y1, R being split into P partitions, to parallelize the computation
    // on multiple GPUs;
    public static List<GrCUDAComputationalElement> vecMultiGPUMockComputation(AsyncGrCUDAExecutionContext context) {
        // Arrays have P partitions;
        int P = partitionsVec;
        int N = 1000;
        DeviceArrayMock[] x = new DeviceArrayMock[P];
        DeviceArrayMock[] y = new DeviceArrayMock[P];
        DeviceArrayMock[] x1 = new DeviceArrayMock[P];
        DeviceArrayMock[] y1 = new DeviceArrayMock[P];
        DeviceArrayMock[] res = new DeviceArrayMock[P];
        for (int i = 0; i < P; i++) {
            x[i] = new DeviceArrayMock(N);
            y[i] = new DeviceArrayMock(N);
            x1[i] = new DeviceArrayMock(N);
            y1[i] = new DeviceArrayMock(N);
            res[i] = new DeviceArrayMock(1);
        }
        List<GrCUDAComputationalElement> computations = new ArrayList<>();
        // Schedule the computations;
        for (int i = 0; i < P; i++) {
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(x[i], true), new ArgumentMock(x1[i])), "SQ1-" + i));
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(y[i], true), new ArgumentMock(y1[i])), "SQ2-" + i));
            computations.add(new KernelExecutionMock(context, Arrays.asList(new ArgumentMock(x1[i], true), new ArgumentMock(y1[i], true), new ArgumentMock(res[i])), "SUM-" + i));
        }
        for (int i = 0; i < P; i++) {
            computations.add(new SyncExecutionMock(context, Collections.singletonList(new ArgumentMock(res[i]))));
        }
        return computations;
    }
}
