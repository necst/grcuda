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

import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.gpu.computation.ArrayReadWriteFunctionExecutionDefault;
import com.nvidia.grcuda.gpu.computation.ArrayReadWriteFunctionExecutionMalloc;
import com.nvidia.grcuda.gpu.computation.DeviceArrayCopyException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class DeviceArrayCopyFunction implements TruffleObject {

    public enum CopyDirection {
        FROM_POINTER,
        TO_POINTER
    }

    private final AbstractArray array;
    private final CopyDirection direction;

    public DeviceArrayCopyFunction(AbstractArray array, CopyDirection direction) {
        this.array = array;
        this.direction = direction;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    private static long extractPointer(Object valueObj, InteropLibrary access) throws UnsupportedMessageException {
        if (access.isPointer(valueObj)) {
            return access.asPointer(valueObj);
        } else {
            return access.asLong(valueObj);
        }
    }

    private static int extractNumber(Object valueObj, InteropLibrary access) throws UnsupportedTypeException {
        try {
            return access.asInt(valueObj);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{valueObj}, "integer expected for numElements");
        }
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "3") InteropLibrary pointerAccess,
                    @CachedLibrary(limit = "3") InteropLibrary numElementsAccess) throws UnsupportedTypeException, ArityException, IndexOutOfBoundsException, DeviceArrayCopyException {
        // Obtain the number of elements to copy;
        long numElements;
        if (arguments.length == 1) {
            numElements = array.getArraySize();
        } else if (arguments.length == 2) {
            numElements = extractNumber(arguments[1], numElementsAccess);
        } else {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(1, arguments.length);
        }
        // Obtain what kind of copy (pointer or array) should be executed;
        try {
            // Try using the native pointer implementation;
            long pointer = extractPointer(arguments[0], pointerAccess);
            new ArrayReadWriteFunctionExecutionMalloc(array, direction, numElements, pointer).schedule();
        } catch (UnsupportedMessageException e) {
            // Try using the array implementation;
            if (pointerAccess.hasArrayElements(arguments[0])) {
                new ArrayReadWriteFunctionExecutionDefault(array, direction, numElements, pointerAccess, arguments[0]).schedule();
            } else {
                // The target object is not an array;
                CompilerDirectives.transferToInterpreter();
                throw UnsupportedTypeException.create(new Object[]{arguments[0]}, "integer expected for " + (direction.equals(CopyDirection.FROM_POINTER) ? "fromPointer" : "toPointer"));
            }
        }
        return array;
    }

    @Override
    public String toString() {
        return "DeviceArrayCopyFunction(deviceArray=" + array + ", direction=" + direction.name() + ")";
    }
}
