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
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public class SparseMatrixCSR implements TruffleObject {

    public enum CsrDimension {
        CSR_DIMENSION_COL,
        CSR_DIMENSION_CUMULATIVE_NNZ,
        CSR_DIMENSION_VAL;
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
    private final DeviceArray cumulativeNnz;
    private final DeviceArray colIdx;

    /**
     * Values of nnz elements.
     */
    private final DeviceArray nnzValues;

    public SparseMatrixCSR(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type valueElementType, Type indexElementType, long dimRows, long dimCols, long numNnz) {
        this.dimRows = dimRows;
        this.dimCols = dimCols;
        this.numNnz = numNnz;
        this.cumulativeNnz = new DeviceArray(grCUDAExecutionContext, (dimRows + 1), indexElementType);
        this.colIdx = new DeviceArray(grCUDAExecutionContext, numNnz, indexElementType);
        this.nnzValues = new DeviceArray(grCUDAExecutionContext, numNnz, valueElementType);
        this.valueElementType = valueElementType;
        this.indexElementType = indexElementType;
    }

    private DeviceArray asDimensionArray(CsrDimension csrDimension) {
        switch (csrDimension) {
            case CSR_DIMENSION_COL:
                return colIdx;
            case CSR_DIMENSION_CUMULATIVE_NNZ:
                return cumulativeNnz;
            case CSR_DIMENSION_VAL:
                return nnzValues;
        }
        throw new GrCUDAException("Invalid Dimension = " + csrDimension.name());
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

    final boolean isCumulativeVectorValid() throws InvalidArrayIndexException, UnsupportedMessageException { // might be useful
        return ((long) this.cumulativeNnz.readArrayElement(0) == 0);
    }

    final boolean isIndexOfNonZero(long row, long col) throws InvalidArrayIndexException, UnsupportedMessageException {
        // returns true if element is nnz
        boolean check = false;
        // check rows
        long tmpIndex = (long) this.cumulativeNnz.readArrayElement(row);
        if (((long) this.cumulativeNnz.readArrayElement(row + 1) > tmpIndex) && ((long) this.colIdx.readArrayElement(tmpIndex) == col)){ // make sure that there's at least one element in that row
            check = true;
        }
        return check;
    }


    final public long getSizeBytes() {
        return colIdx.getSizeBytes() + cumulativeNnz.getSizeBytes() + nnzValues.getSizeBytes();
    }

    public final long getPointer(CsrDimension csrDimension) { // add exception, must not be called
        checkFreeMatrix();
        return asDimensionArray(csrDimension).getPointer();
    }

    @Override
    public String toString() {
        return "SparseMatrixCSR{" +
                "dimRows=" + dimRows +
                ", dimCols=" + dimCols +
                ", numNnz=" + numNnz +
                ", valueElementType=" + valueElementType +
                ", indexElementType=" + indexElementType +
                ", matrixFreed=" + matrixFreed +
                ", cumulativeNnz=" + cumulativeNnz +
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
        cumulativeNnz.freeMemory();
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
        if (isIndexOfNonZero(row, col)) {
            element = nnzValues.readArrayElement((long) this.cumulativeNnz.readArrayElement(row));
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
        colIdx.writeArrayElement(position, col, valueLibrary, elementTypeProfile);
        Object tmp = cumulativeNnz.readArrayElement((row + 1));
        cumulativeNnz.writeArrayElement((row + 1), tmp, valueLibrary, elementTypeProfile);
    }

    Object readCsrDimension(CsrDimension csrDimension, long index) throws InvalidArrayIndexException, UnsupportedMessageException {
        checkFreeMatrix();
        return asDimensionArray(csrDimension).readArrayElement(index);
    }
}
