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
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
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
public class SparseMatrixCOO implements TruffleObject {

    public enum CooDimension {
        COO_DIMENSION_ROW,
        COO_DIMENSION_COL,
        COO_DIMENSION_VAL;
    }

    /**
     * Row and column dimensions.
     */
    private final long dimRows;
    private final long dimCols;
    private final long numNnz;
    private final Type valueElementType;
    private final Type indexElementType;
    private boolean matrixFreed;

// /** Stride in each dimension. */
// private final long[] stridePerDimension;
// non credo serva

    /**
     * Row and column indices of nnz elements.
     */
    private final DeviceArray rowIdx;
    private final DeviceArray colIdx;

    /**
     * Values of nnz elements.
     */
    private final DeviceArray nnzValues;

    public SparseMatrixCOO(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type valueElementType, Type indexElementType, long dimRows, long dimCols, long numNnz) {
        this.dimRows = dimRows;
        this.dimCols = dimCols;
        this.numNnz = numNnz;
        this.rowIdx = new DeviceArray(grCUDAExecutionContext, numNnz, indexElementType);
        this.colIdx = new DeviceArray(grCUDAExecutionContext, numNnz, indexElementType);
        this.nnzValues = new DeviceArray(grCUDAExecutionContext, numNnz, valueElementType);
        this.valueElementType = valueElementType;
        this.indexElementType = indexElementType;
    }

    private DeviceArray asDimensionArray(CooDimension cooDimension) {
        switch (cooDimension) {
            case COO_DIMENSION_ROW:
                return rowIdx;
            case COO_DIMENSION_COL:
                return colIdx;
            case COO_DIMENSION_VAL:
                return nnzValues;
        }
        throw new GrCUDAException("Invalid Dimension = " + cooDimension.name());
    }

    private void checkFreeMatrix() {
        if (matrixFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("Matrix freed already");
        }
    }

    // removed get elements, we have dimCols and dimRows in the constructor already

    final boolean isIndexValid(long row, long col) throws InvalidArrayIndexException, UnsupportedMessageException {
        return ((row >= 0) && (row < dimRows) && (col >= 0) && (col < dimCols));
    }

    final long indexOfNonZero(long row, long col) throws InvalidArrayIndexException, UnsupportedMessageException {
        // returns the index corresponds to nnz element if present, -1 otherwise
        long index = -1;
        // check rows
        for (int i = 0; i < this.nnzValues.getArraySize(); i++) {
            if (((long) this.rowIdx.readArrayElement(i) == row) && ((long) this.colIdx.readArrayElement(i) == col)) {
                index = i;
            }
        }
        return index;
    }

    final public long getSizeBytes() {
        return nnzValues.getSizeBytes() + rowIdx.getSizeBytes() + colIdx.getSizeBytes(); // must be same size
    }

    public final long getPointer(CooDimension cooDimension) { // add exception, must not be called
        checkFreeMatrix();
        return asDimensionArray(cooDimension).getPointer();
    }

    public String toString() {
        return "SparseMatrix{" +
                        "dimRows=" + dimRows +
                        ", dimCols=" + dimCols +
                        ", numNnz=" + numNnz +
                        ", rowIdx=" + rowIdx +
                        ", colIdx=" + colIdx +
                        ", nnzValues=" + nnzValues +
                        '}';
    }

    protected void finalize() throws Throwable {
        if (!matrixFreed) {
            this.freeMemory();
        }
        super.finalize();
    }

    public void freeMemory() {
        checkFreeMatrix();
        nnzValues.freeMemory();
        rowIdx.freeMemory();
        colIdx.freeMemory();
        matrixFreed = true;
    }

//    @ExportMessage
//    @SuppressWarnings("static-method")
//    public long getArraySize() {
//        checkFreeMatrix();
//        return nnzValues.getArraySize();
//    }
// I don't think we need it anymore since numNnz belongs to the constructor already

//    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementModifiable(long row, long col) {
        return row >= 0 && col >= 0 && row < dimRows && col < dimCols;
    }

//    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementReadable(long row, long col) {
        return !matrixFreed && isArrayElementModifiable(row, col);
    }

//    @ExportMessage
    Object readMatrixElement(long row, long col) throws InvalidArrayIndexException, UnsupportedMessageException {
        checkFreeMatrix();
        if (!isIndexValid(row, col)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(row);
        }
        Object element = 0;
        long index = indexOfNonZero(row, col);
        if (index != -1) {
            element = nnzValues.readArrayElement(index);
        }
        return element;
    }
    void writeMatrixElement(long row, long col, long position, Object value, InteropLibrary valueLibrary, ValueProfile elementTypeProfile) throws InvalidArrayIndexException, UnsupportedMessageException, UnsupportedTypeException {
        checkFreeMatrix();
        if (!isIndexValid(row, col)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(row);
        }
        if (position >= numNnz){
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(position);
        }
        nnzValues.writeArrayElement(position, value, valueLibrary, elementTypeProfile);
        rowIdx.writeArrayElement(position, row, valueLibrary, elementTypeProfile);
        colIdx.writeArrayElement(position, col, valueLibrary, elementTypeProfile);
    }

    Object readCooDimension(CooDimension cooDimension, long index) throws InvalidArrayIndexException, UnsupportedMessageException {
        checkFreeMatrix();
        return asDimensionArray(cooDimension).readArrayElement(index);
    }
}
