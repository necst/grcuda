/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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
package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.DeviceArray;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.*;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class DeviceArrayPrefetchAsyncFunction implements TruffleObject {

    private final DeviceArray deviceArray;

    public DeviceArrayPrefetchAsyncFunction(DeviceArray deviceArray) {
        this.deviceArray = deviceArray;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    private static int extractNumber(Object valueObj, String argumentName, InteropLibrary access) throws UnsupportedTypeException {
        try {
            return access.asInt(valueObj);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{valueObj}, "integer expected for " + argumentName);
        }
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "3") InteropLibrary numElementsAccess,
                    @CachedLibrary(limit = "3") InteropLibrary deviceIdAccess) throws UnsupportedTypeException, ArityException, IndexOutOfBoundsException {
        long numElements;
        int deviceId;
        if (arguments.length == 0) {
            numElements = deviceArray.getArraySize();
            deviceId = 0;
        } else if (arguments.length == 1) {
            numElements = extractNumber(arguments[0], "numElements", numElementsAccess);
            deviceId = 0;
        } else if (arguments.length == 2) {
            numElements = extractNumber(arguments[0], "numElements", numElementsAccess);
            deviceId = extractNumber(arguments[1], "deviceId", deviceIdAccess);
        } else {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(2, arguments.length);
        }
        deviceArray.prefetchAsync(numElements, deviceId);
        return deviceArray;
    }

    @Override
    public String toString() {
        return "DeviceArrayPrefetchAsyncFunction(deviceArray=" + deviceArray + ", " + ")";
    }
}
