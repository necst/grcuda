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
package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;

import java.io.Closeable;
import java.io.IOException;
import java.lang.instrument.Instrumentation;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class KernelArguments implements Closeable {

    private final Object[] originalArgs;
    /**
     * Associate each input object to the characteristics of its argument, such as its type and if it's constant;
     */
    private final List<ComputationArgumentWithValue> kernelArgumentWithValues = new ArrayList<>();
    private final UnsafeHelper.PointerArray argumentArray;
    private final ArrayList<Closeable> argumentValues = new ArrayList<>();

    public KernelArguments(Object[] args, ComputationArgument[] kernelArgumentList) {
        this.originalArgs = args;
        this.argumentArray = UnsafeHelper.createPointerArray(args.length);
        assert(args.length == kernelArgumentList.length);
        // Initialize the list of arguments and object references;
        for (int i = 0; i < args.length; i++) {
            kernelArgumentWithValues.add(new ComputationArgumentWithValue(kernelArgumentList[i], args[i]));
        }
    }

    public void setArgument(int argIdx, UnsafeHelper.MemoryObject obj) {
        argumentArray.setValueAt(argIdx, obj.getAddress());
        argumentValues.add(obj);
    }

    long getPointer() {
        return argumentArray.getAddress();
    }

    public Object[] getOriginalArgs() {
        return originalArgs;
    }

    public Object getOriginalArg(int index) {
        return originalArgs[index];
    }

    public List<ComputationArgumentWithValue> getKernelArgumentWithValues() {
        return kernelArgumentWithValues;
    }

    /**
     * Calculate the total size in bytes of the arguments that are {@link DeviceArray}
     * @return total size in bytes
     */
    public long getTotalKernelDeviceArraySize(){

        return Arrays.stream(originalArgs)
                .filter(o -> o.getClass() == DeviceArray.class)
                .mapToLong(o -> ((DeviceArray) o).getSizeBytes()).sum();

    }

    @Override
    public String toString() {
        return "KernelArgs=" + Arrays.toString(originalArgs);
    }

    @Override
    public void close() {
        this.argumentArray.close();
        for (Closeable c : argumentValues) {
            try {
                c.close();
            } catch (IOException e) {
                /* ignored */
            }
        }
    }
}
