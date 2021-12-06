package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.LittleEndianNativeArrayView;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;

public class DeviceArrayMock extends DeviceArray {
    public DeviceArrayMock() {
        super(new AsyncGrCUDAExecutionContextMock(), 0, Type.SINT32);
    }

    public DeviceArrayMock(AbstractGrCUDAExecutionContext context) {
        super(context, 0, Type.SINT32);
    }

    @Override
    protected LittleEndianNativeArrayView allocateMemory() {
        return null;
    }
}

