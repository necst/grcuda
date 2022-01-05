package com.nvidia.grcuda.test.runtime.array;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collection;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.nvidia.grcuda.test.util.GrCUDATestUtil;

@RunWith(Parameterized.class)
public class SparseVectorTest {
    private static final int TEST_NNZ = 1000;
    private final String indexType;
    private final String valueType;
    private final int numNnz;

    public SparseVectorTest(String indexType, String valueType, int numNnz) {
        this.indexType = indexType;
        this.valueType = valueType;
        this.numNnz = numNnz;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        //TODO: convert it to the cartesian product of all combinations
        return Arrays.asList(new Object[][]{
                {"int", "float", TEST_NNZ},
                {"long", "float", TEST_NNZ},
                {"int", "double", TEST_NNZ},
                {"long", "double", TEST_NNZ}
        });
    }

    private Value[] createSpVector(Context context) {
        Value deviceArrayCtor = context.eval("grcuda", "DeviceArray");
        Value sparseVectorCtor = context.eval("grcuda", "SparseVector");
        Value idx = deviceArrayCtor.execute(this.indexType, this.numNnz); // TODO non dovrebbe essere viceversa?
        Value val = deviceArrayCtor.execute(this.valueType, this.numNnz);
        Value spVector = sparseVectorCtor.execute(idx, val, this.numNnz);
        return new Value[]{spVector, idx, val};
    }

    @Test(expected = Exception.class)
    public void testFree() {
        try (Context context = GrCUDATestUtil.buildTestContext().build()) {
            Value[] vectors = createSpVector(context);
            Value spVector = vectors[0];
            Value idx = vectors[1];
            Value val = vectors[2];
            // The memory is not freed when the array has just been created
            assertFalse(spVector.getMember("isMemoryFreed").asBoolean());
            assertFalse(idx.getMember("isMemoryFreed").asBoolean());
            assertFalse(val.getMember("isMemoryFreed").asBoolean());
            // First free, should succeed
            spVector.getMember("free").execute();
            assertTrue(spVector.getMember("isMemoryFreed").asBoolean());
            // Second free, should fail
            spVector.getMember("free").execute();

        }
    }
}
