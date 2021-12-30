package com.nvidia.grcuda.runtime.stream.policy;

import com.nvidia.grcuda.runtime.Device;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionDAG;

/**
 * When using multiple GPUs, selecting the stream where a computation is executed implies
 * the selection of a GPU, as each stream is uniquely associated to a single GPU.
 * This abstract class defines how a {@link GrCUDAStreamPolicy}
 * selects a {@link com.nvidia.grcuda.runtime.Device} on which a {@link com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement}
 * is executed. Device selection is performed by {@link RetrieveNewStreamPolicy} (when creating a new stream)
 * and {@link RetrieveParentStreamPolicy} (when the parent's stream cannot be directly reused).
 * For example, we can select the device that requires the least data transfer.
 */
public abstract class DeviceSelectionPolicy {
    abstract Device retrieve(ExecutionDAG.DAGVertex vertex);

    /**
     * Cleanup the internal state of the class, if required;
     */
    void cleanup() { }
}
