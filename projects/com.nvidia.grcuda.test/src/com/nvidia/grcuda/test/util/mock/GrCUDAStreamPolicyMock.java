package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.GrCUDAStreamPolicy;

public class GrCUDAStreamPolicyMock extends GrCUDAStreamPolicy {
    public GrCUDAStreamPolicyMock(RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum, RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum, int numberOfAvailableGPUs, int numberOfGPUsToUse) {
        super(new GrCUDADevicesManagerMock(new DeviceListMock(numberOfAvailableGPUs), numberOfGPUsToUse), retrieveNewStreamPolicyEnum, retrieveParentStreamPolicyEnum);
    }
}
