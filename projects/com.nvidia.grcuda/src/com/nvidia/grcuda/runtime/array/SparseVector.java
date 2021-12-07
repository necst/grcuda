/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
package com.nvidia.grcuda.runtime.array;

import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.runtime.LittleEndianNativeArrayView;
import com.nvidia.grcuda.runtime.computation.arraycomputation.DeviceArrayReadExecution;
import com.nvidia.grcuda.runtime.computation.arraycomputation.DeviceArrayWriteExecution;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Cached.Shared;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public class SparseVector implements TruffleObject {

    public enum SparseDimension{
        SP_VEC_VALUES,
        SP_VEC_INDICES;
    }

    /**
     * Values of non zero elements stored in the array.
     */
    private final Type valueElementType;
    private final Type indexElementType;
    private boolean vectorFreed;

    /**
     * Indices of non
     */
    private final DeviceArray indices;
    private final DeviceArray nnz;

    /**
     * Number non zero elements stored in the array.
     * */
    private final long numNnz;

    public SparseVector(AbstractGrCUDAExecutionContext grCUDAExecutionContext, long numNnz, Type valueElementType, Type indexElementType) {
        this.numNnz = numNnz;
        this.nnz = new DeviceArray(grCUDAExecutionContext, numNnz, valueElementType);
        this.indices = new DeviceArray(grCUDAExecutionContext, numNnz, indexElementType);
        this.valueElementType = valueElementType;
        this.indexElementType = indexElementType;
    }

    private DeviceArray asDimensionArray(SparseDimension sparseDimension) {
        switch (sparseDimension){
            case SP_VEC_INDICES:
                return indices;
            case SP_VEC_VALUES:
                return nnz;
        }
        throw new GrCUDAException("Invalid Dimension = " + sparseDimension.name());
    }

    private void checkFreeVector() {
        if (vectorFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("Vector freed already");
        }
    }

    final long indexOfNonZero(long idx) throws InvalidArrayIndexException, UnsupportedMessageException {
        // returns the index corresponds to nnz element if present, -1 otherwise
        long index = -1;
        for(int i = 0; i < numNnz; i++){
            if(((long) this.indices.readArrayElement(i)) == idx){
                index = i;
            }
        }
        return index;
    }

    public long getSizeBytes() {
        checkFreeVector();
        return numNnz * (valueElementType.getSizeBytes() + indexElementType.getSizeBytes());
    }

    public long getPointer(SparseDimension sparseDimension) {
        checkFreeVector();
        return asDimensionArray(sparseDimension).getPointer();
    }

    public Type getElementType(SparseDimension sparseDimension) {
        checkFreeVector();
        return asDimensionArray(sparseDimension).getElementType();
    }

    public String toString() {
        return "SparseVector{" +
                "valueElementType=" + valueElementType +
                ", indexElementType=" + indexElementType +
                ", indices=" + indices +
                ", nnz=" + nnz +
                ", numNnz=" + numNnz +
                '}';
    }

    protected void finalize() throws Throwable {
        if (!vectorFreed) {
            this.freeMemory();
        }
        super.finalize();
    }

    public void freeMemory() {
        checkFreeVector();
        nnz.freeMemory();
        indices.freeMemory();
        vectorFreed = true;
    }

//    @ExportMessage
    boolean isArrayElementModifiable(long index) {
        return index >= 0 && index < numNnz;
    }

//    @ExportMessage
    boolean isArrayElementReadable(long index) {
        return !vectorFreed && isArrayElementModifiable(index);
    }

    @SuppressWarnings("static-method")
//    @ExportMessage
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

//    @ExportMessage
    Object readArrayElement(long idx) throws InvalidArrayIndexException, UnsupportedMessageException {
        checkFreeVector();
        if (!isArrayElementModifiable(idx)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(idx);
        }
        // TODO: conasider DAG scheduling related issues
        Object element = 0;
        long index = indexOfNonZero(idx);
        if(index != -1){
            element = nnz.readArrayElement(index);
        }
        return element;
//            if (this.canSkipScheduling()) {
//                // check whether index corresponds to a value's position
//                Object element = 0;
//                for(int i = 0; i < numNnz; i++){
//                    if(((long) this.indices.readArrayElement(i)) == index){ // si fa così?
//                        element = AbstractArray.readArrayElementNative(this.nativeView, i, this.elementType, elementTypeProfile);
//                    }
//                }
//                return element;
//            } else {
//                return new DeviceArrayReadExecution(this.nnz, index, elementTypeProfile).schedule(); // controllare se ha senso
//                return new DeviceArrayReadExecution(this.indices, index, elementTypeProfile).schedule(); // controllare se ha senso
//            }
    }

//    @ExportMessage
    public void writeSparseArrayElement(long position, Object idx, Object value, InteropLibrary valueLibrary, ValueProfile elementTypeProfile) throws UnsupportedTypeException, InvalidArrayIndexException {
        checkFreeVector();
        if (!isArrayElementModifiable((long) idx)) { // to avoid casting we should change elements' modifiability totally, maybe it's not the case
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create((long) idx);
        }
        if (position >= numNnz){
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(position);
        }
        // TODO: consider DAG scheduling related issues
        this.nnz.writeArrayElement(position, value, valueLibrary, elementTypeProfile);
        this.indices.writeArrayElement(position, idx, valueLibrary, elementTypeProfile);
//        if (this.canSkipScheduling()) {
//            // Fast path, skip the DAG scheduling;
//            this.indices.writeArrayElement(numNnz, index, valueLibrary, elementTypeProfile); // vanno riordinati
//            this.nnz.writeArrayElement(numNnz, index, valueLibrary, elementTypeProfile);
//        } else {
//            new DeviceArrayWriteExecution(this.nnz, index, value, valueLibrary, elementTypeProfile).schedule(); // non si può fare, che ci inventiamo? idem per read
//            new DeviceArrayWriteExecution(this.indices, index, index, valueLibrary, elementTypeProfile).schedule(); // non si può fare, che ci inventiamo? idem per read
//
//        }
    }


}
