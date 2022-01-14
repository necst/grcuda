package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.LittleEndianNativeArrayView;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;

public class MultiDimDeviceArrayMock extends MultiDimDeviceArray {
    public MultiDimDeviceArrayMock(long[] dimensions, boolean columnMajor) {
        super(new AsyncGrCUDAExecutionContextMock(), Type.SINT32, dimensions, columnMajor);
    }

    @Override
    protected LittleEndianNativeArrayView allocateMemory() {
        return null;
    }
}
