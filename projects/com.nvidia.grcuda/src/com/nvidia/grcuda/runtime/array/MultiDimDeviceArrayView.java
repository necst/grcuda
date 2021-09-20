/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.runtime.array;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.runtime.computation.arraycomputation.MultiDimDeviceArrayViewReadExecution;
import com.nvidia.grcuda.runtime.computation.arraycomputation.MultiDimDeviceArrayViewWriteExecution;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public class MultiDimDeviceArrayView extends AbstractArray implements TruffleObject {

    private final MultiDimDeviceArray mdDeviceArray;
    private final int thisDimension;
    private final long offset;
    private final long stride;

    /**
     * A (N - 1)-dimensional view of an N-dimensional dense array.
     * From the host language perspective (i.e. from the user perspective), an array view should be no different than
     * a standard (N - 1)-dimensional array.
     * From the internal implementation of GrCUDA, it should always be considered that the view is part of a larger memory
     * chunk managed (also) by the GPU. Some CUDA APIs do not allow operating on memory chunks, but require access to the full array.
     * As such, array views also provide access to information about the full array they belong to.
     * For example, let's assume that the original array has 4 dimensions. We can create 3, 2, 1 dimensional views from it (in this order).
     * Let's say that we are creating a 2-dimensional view
     * @param mdDeviceArray the full array from which this view is created (the 4-dimensional array, in the example)
     * @param dim the dimension identifier of this view (e.g. 2, in the example)
     * @param offset the index (in the full array) at which this array view start
     * @param stride value used to jump to consecutive values in the array, and determined by the slice that has been extracted
     */
    public MultiDimDeviceArrayView(MultiDimDeviceArray mdDeviceArray, int dim, long offset, long stride) {
        super(mdDeviceArray.grCUDAExecutionContext, mdDeviceArray.elementType, mdDeviceArray.isLastComputationArrayAccess());
        this.mdDeviceArray = mdDeviceArray;
        this.thisDimension = dim;
        this.offset = offset; // Index at which this array view starts;
        this.stride = stride;
        // Register the array in the GrCUDAExecutionContext;
        this.registerArray();
    }

    public int getDimension() {
        return thisDimension;
    }

    public long getOffset() {
        return offset;
    }

    public long getStride() {
        return stride;
    }

    @Override
    public long getPointer() {
        return mdDeviceArray.getPointer() + offset * elementType.getSizeBytes();
    }

    @Override
    public long getFullArrayPointer() {
        return mdDeviceArray.getFullArrayPointer();
    }

    @Override
    public boolean isColumnMajorFormat() {
        return mdDeviceArray.isColumnMajorFormat();
    }

    /**
     * Propagate the flag to the parent array, so other temporary views are aware of this computation;
     * @param lastComputationArrayAccess if the last computation on this array is a host read/write
     */
    @Override
    public void setLastComputationArrayAccess(boolean lastComputationArrayAccess) {
        super.setLastComputationArrayAccess(lastComputationArrayAccess);
        this.mdDeviceArray.setLastComputationArrayAccess(lastComputationArrayAccess);
    }

    @Override
    public boolean isLastComputationArrayAccess() { return this.mdDeviceArray.isLastComputationArrayAccess(); }

    /**
     * Propagate the stream mapping to the parent array, so other temporary views are aware of this mapping;
     * @param streamMapping the stream to which this array is associated
     */
    @Override
    public void setStreamMapping(CUDAStream streamMapping) {
        this.mdDeviceArray.setStreamMapping(streamMapping);
        this.streamMapping = streamMapping;
    }

    /**
     * Return the parent stream mapping, to guarantee that all views have the same mapping;
     * @return the stream to which this array is associated
     */
    @Override
    public CUDAStream getStreamMapping() {
        return this.mdDeviceArray.getStreamMapping();
    }

    @Override
    public long getSizeBytes() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return mdDeviceArray.getElementsInDimension(thisDimension) * elementType.getSizeBytes();
    }

    @Override
    public long getFullArraySizeBytes() {
        return mdDeviceArray.getFullArraySizeBytes();
    }

    @Override
    public void freeMemory() {
        // This should not be called directly on a view;
        CompilerDirectives.transferToInterpreter();
        throw new GrCUDAException("Freeing memory directly on a MultiDimDeviceArrayView is not allowed");
    }

    @Override
    public String toString() {
        return String.format("MultiDimDeviceArrayView(dim=%d, offset=%d, stride=%d)\n",
                        thisDimension, offset, stride);
    }

    //
    // Implementation of Interop Library
    //

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @Override
    public long getArraySize() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return mdDeviceArray.getElementsInDimension(thisDimension);
    }

    @ExportMessage
    boolean isArrayElementReadable(long index) {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return index >= 0 && index < mdDeviceArray.getElementsInDimension(thisDimension);
    }

    @ExportMessage
    boolean isArrayElementModifiable(long index) {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return (thisDimension + 1) == mdDeviceArray.getNumberDimensions() &&
                        index >= 0 && index < mdDeviceArray.getElementsInDimension(thisDimension);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

    @ExportMessage
    Object readArrayElement(long index,
                    @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws InvalidArrayIndexException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((index < 0) || (index >= mdDeviceArray.getElementsInDimension(thisDimension))) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        try {
            if (this.canSkipScheduling()) {
                // Fast path, skip the DAG scheduling;
                return readNativeView(index, elementTypeProfile);
            } else {
                return new MultiDimDeviceArrayViewReadExecution(this, index, elementTypeProfile).schedule();
            }
        } catch (UnsupportedTypeException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public Object readNativeView(long index, @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) {
        if ((thisDimension + 1) == mdDeviceArray.getNumberDimensions()) {
            long flatIndex = offset + index * stride;
            return AbstractArray.readArrayElementNative(this.mdDeviceArray.getNativeView(), flatIndex, this.mdDeviceArray.getElementType(), elementTypeProfile);
        } else {
            long off = offset + index * stride;
            long newStride = mdDeviceArray.getStrideInDimension(thisDimension + 1);
            return new MultiDimDeviceArrayView(mdDeviceArray, thisDimension + 1, off, newStride);
        }
    }

    @ExportMessage
    void writeArrayElement(long index, Object value,
                    @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                    @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException, InvalidArrayIndexException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((index < 0) || (index >= mdDeviceArray.getElementsInDimension(thisDimension))) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(index);
        }
        if (this.canSkipScheduling()) {
            // Fast path, skip the DAG scheduling;
            writeNativeView(index, value, valueLibrary, elementTypeProfile);
        } else {
            new MultiDimDeviceArrayViewWriteExecution(this, index, value, valueLibrary, elementTypeProfile).schedule();
        }
    }

    @Override
    public void writeNativeView(long index, Object value,
                                @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                                @Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException {
        if ((thisDimension + 1) == mdDeviceArray.getNumberDimensions()) {
            long flatIndex = offset + index * stride;
            AbstractArray.writeArrayElementNative(this.mdDeviceArray.getNativeView(), flatIndex, value, this.mdDeviceArray.getElementType(), valueLibrary, elementTypeProfile);
        } else {
            CompilerDirectives.transferToInterpreter();
            throw new IllegalStateException("tried to write non-last dimension in MultiDimDeviceArrayView");
        }
    }

    public MultiDimDeviceArray getMdDeviceArray() {
        return mdDeviceArray;
    }
}