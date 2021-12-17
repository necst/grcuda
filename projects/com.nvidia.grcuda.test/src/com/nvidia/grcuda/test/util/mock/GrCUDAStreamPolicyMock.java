package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.GrCUDADevicesManager;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.GrCUDAStreamPolicy;

public class GrCUDAStreamPolicyMock extends GrCUDAStreamPolicy {
    private int streamCount;

    public GrCUDAStreamPolicyMock(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum, RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum) {
        super(new GrCUDADevicesManagerMock(), retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum);
    }

    @Override
    public CUDAStream createStream() {
        // FIXME: we really need a mocked runtime to avoid using a static variable!
        CUDAStream newStream = new CUDAStream(0, streamCount++, AsyncGrCUDAExecutionContextMock.currentGPU);
        streams.add(newStream);
        return newStream;
    }
}
