package com.nvidia.grcuda.test.runtime.array;

import java.util.Arrays;
import java.util.Collection;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;

@RunWith(Parameterized.class)
public class SparseMatrixCOOTest {
    private static final int TEST_NNZ = 100;

    private static int rows = 100;
    private static int cols = 100;
    private final int numNnz;
    private final String idxType;
    private final String valueType;


    public SparseMatrixCOOTest(String indexType, String valueType, int numNnz, int rows, int cols) {
        SparseMatrixCOOTest.rows = rows;
        this.valueType = valueType;
        SparseMatrixCOOTest.cols = cols;
        this.numNnz = numNnz;
        this.idxType = indexType;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        //TODO: convert it to the cartesian product of all combinations
        return Arrays.asList(new Object[][]{
                {"int", "float", TEST_NNZ, rows, cols},
                {"long", "float", TEST_NNZ, rows, cols},
                {"int", "double", TEST_NNZ, rows, cols},
                {"long", "double", TEST_NNZ, rows, cols}
        });
    }

    private Value[] createSpMatrixCOO(Context context) {
        Value sparseMatrixCOOCtor = context.eval("grcuda", "SparseMatrixCOO");
        Value deviceArrayCtor = context.eval("grcuda", "DeviceArray");
        Value rowIdx = deviceArrayCtor.execute(this.idxType, this.numNnz);
        Value colIdx = deviceArrayCtor.execute(this.idxType, this.numNnz);
        Value nnz = deviceArrayCtor.execute(this.valueType, this.numNnz);
        Value spMat = sparseMatrixCOOCtor.execute(colIdx, rowIdx, nnz, rows, cols, false);
        return new Value[]{spMat, rowIdx, colIdx, nnz};
    };

    @Test
    public void testFreeMatrix() {
        try (Context context = GrCUDATestUtil.buildTestContext().option(
                "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {
            Value[] spMatrixValues = createSpMatrixCOO(context);
            // The memory is not freed when the array has just been created
            Value spMatrixCOO = spMatrixValues[0];
            Value rowIdx = spMatrixValues[1];
            Value colIdx = spMatrixValues[2];
            Value nnz = spMatrixValues[3];
            assertFalse(spMatrixCOO.getMember("isMemoryFreed").asBoolean());
            // First free, should succeed
            spMatrixCOO.getMember("free").execute();
            assertTrue(spMatrixCOO.getMember("isMemoryFreed").asBoolean());
        }
    }
}
