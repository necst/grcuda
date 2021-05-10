package com.nvidia.grcuda.test.mock;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.nvidia.grcuda.gpu.GrCUDADevicesManager;
import com.nvidia.grcuda.gpu.stream.CUDAStream;
import com.nvidia.grcuda.gpu.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.gpu.stream.StreamPolicy;

public class StreamPolicyMock extends StreamPolicy{
    private int streamCount;
    private final GrCUDADevicesManager devicesManager;
    public StreamPolicyMock(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum, RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum, GrCUDADevicesManager devicesManager, CUDARuntime runtime){
        super(retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum, devicesManager, runtime);
        this.streamCount = 0;
        this.devicesManager = devicesManager;
    }

    @Override
    public CUDAStream createStream(int deviceId) {
        CUDAStream newStream = new CUDAStream(0, this.streamCount++, deviceId);

        devicesManager.addStreamCount(deviceId);
        return newStream;
    }

    

}
