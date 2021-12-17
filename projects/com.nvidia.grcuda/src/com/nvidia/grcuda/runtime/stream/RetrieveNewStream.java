/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime.stream;

import com.nvidia.grcuda.runtime.Device;

import java.util.Collection;

/**
 * This abstract class defines how a {@link GrCUDAStreamManager}
 * will assign a {@link CUDAStream} to a {@link com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement}
 * that has no dependency on active computations.
 * For example, it could create a new stream or provide an existing stream that is currently not used;
 */
public abstract class RetrieveNewStream {
    abstract CUDAStream retrieve(Device device);

    /**
     * Initialize the class with the provided stream,
     * for example a new stream that can be provided by {@link RetrieveNewStream#retrieve(Device device)}
     * @param device the device for which we want to update the stream status
     * @param stream a stream that should be associated to the class
     */
    void update(Device device, CUDAStream stream) { }

    /**
     * Initialize the class with the provided streams,
     * for example new streams that can be provided by {@link RetrieveNewStream#retrieve(Device device)}
     * @param device the device for which we want to update the stream status
     */
    void update(Device device) { }

    /**
     * Cleanup the internal state of the class, if required;
     */
    void cleanup() { }
}
