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
package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.profiles.ValueProfile;

public class DeviceArrayWriteExecution extends ArrayAccessExecution<DeviceArray> {

    private final long index;
    private final Object value;
    private final InteropLibrary valueLibrary;
    private final ValueProfile elementTypeProfile;

    public DeviceArrayWriteExecution(DeviceArray array,
                                     long index,
                                     Object value,
                                     InteropLibrary valueLibrary,
                                     ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new ArrayAccessExecutionInitializer<>(array, false), array);
        this.index = index;
        this.value = value;
        this.valueLibrary = valueLibrary;
        this.elementTypeProfile = elementTypeProfile;
    }

    @Override
    public void updateLocationOfArrays() {
        // Clear the list of up-to-date locations: only the CPU has the updated array;
        array.resetArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
    }

    @Override
    public Object execute() throws UnsupportedTypeException {
        array.writeNativeView(index, value, valueLibrary, elementTypeProfile);
        this.setComputationFinished();
        return NoneValue.get();
    }

    @Override
    public String toString() {
//        return "DeviceArrayWriteExecution(" +
//                "array=" + array +
//                ", index=" + index +
//                ", value=" + value +
//                ")";
        return "array write on " + System.identityHashCode(array) + "; index=" + index + "; value=" + value + "; stream=" + getStream().getStreamNumber();
    }
}
