package com.nvidia.grcuda.gpu.stream;

import com.nvidia.grcuda.gpu.executioncontext.ExecutionDAG;

public abstract class ChooseDeviceHeuristic {

    abstract int getDevice(ExecutionDAG.DAGVertex vertex);

}

