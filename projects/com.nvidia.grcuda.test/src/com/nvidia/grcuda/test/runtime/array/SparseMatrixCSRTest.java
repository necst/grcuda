package com.nvidia.grcuda.test.runtime.array;

import java.util.Arrays;
import java.util.Collection;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrixCSR;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;

@RunWith(Parameterized.class)
public class SparseMatrixCSRTest {
    private static final int TEST_NNZ = 100;

    private static int rows = 100;
    private static int cols = 100;
    private final int numNnz;
    private final String idxType;
    private final String valueType;


    public SparseMatrixCSRTest(String indexType, String valueType, int numNnz, int rows, int cols) {
        SparseMatrixCSRTest.rows = rows;
        this.valueType = valueType;
        SparseMatrixCSRTest.cols = cols;
        this.numNnz = numNnz;
        this.idxType = indexType;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        //TODO: convert it to the cartesian product of all combinations
        return Arrays.asList(new Object[][]{
                //{"int", "float", TEST_NNZ, rows, cols},
                //{"long", "float", TEST_NNZ, rows, cols},
                //{"int", "double", TEST_NNZ, rows, cols},
                {"long", "double", TEST_NNZ, rows, cols}
        });
    }

    private Value createSpMatrixCSR(Context context, float edgeValue) {
        final int numElements = numNnz;
        Value sparseMatrixCSRCtor = context.eval("grcuda", "SparseMatrixCSR");
        Value deviceArrayCtor = context.eval("grcuda", "DeviceArray");
        Value rowPtr = deviceArrayCtor.execute("int", this.numNnz + 1);
        Value colIdx = deviceArrayCtor.execute("int", this.numNnz);
        Value nnz = deviceArrayCtor.execute("float", this.numNnz);


        for (int i = 0; i < numElements; ++i) {
            rowPtr.setArrayElement(i, i);
            colIdx.setArrayElement(i, i);
            nnz.setArrayElement(i, edgeValue);
        }

        rowPtr.setArrayElement(numElements, numElements);

        Value spMat = sparseMatrixCSRCtor.execute(colIdx, rowPtr, nnz, rows, cols);
        return spMat;
    };

    @Test
    public void testFreeMatrix() {
        try (Context context = GrCUDATestUtil.buildTestContext().option(
                "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {
            Value spMatrixCSR = createSpMatrixCSR(context, 0);
            // The memory is not freed when the array has just been created
            assertFalse(spMatrixCSR.getMember("isMemoryFreed").asBoolean());
            // First free, should succeed
            spMatrixCSR.getMember("free").execute();
            assertTrue(spMatrixCSR.getMember("isMemoryFreed").asBoolean());
        }
    }

    @Test
    public void testSpMVCSR() {
        try (Context context = GrCUDATestUtil.buildTestContext().option(
                "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {

            final float edgeValue = (float) Math.random();

            Value spMatrixCSR = createSpMatrixCSR(context, edgeValue);

            Value cu = context.eval("grcuda", "CU");

            final int numElements = numNnz;
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);


            alpha.setArrayElement(0, 1);
            beta.setArrayElement(0, 0);

            for (int i = 0; i < numElements; ++i) {
                dnVec.setArrayElement(i, 1.0);
            }

            spMatrixCSR.getMember("SpMV").execute(alpha, beta, dnVec, outVec);

            Value sync = context.eval("grcuda", "cudaDeviceSynchronize");
            sync.execute();

            for (int i = 0; i < numElements; ++i) {
                assertEquals(edgeValue, outVec.getArrayElement(i).asFloat(), 1e-5);
            }
        }
    }
}
