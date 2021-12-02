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

import java.util.Arrays;

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
public class SparseMatrix extends AbstractArray implements TruffleObject {

    /**
     * Row and column dimensions.
     * */
    private final long dimRows;
    private final long dimCols;

//    /** Stride in each dimension. */
//    private final long[] stridePerDimension;
//    non credo serva

    /**
     * Row and column indices of nnz elements.
     * */
    private final DeviceArray rowIdx = null;
    private final DeviceArray colIdx = null;

    /**
     * Values of nnz elements.
     */
    private final DeviceArray nnzValues = null;

//
//    /** true if data is stored in column-major format (Fortran), false row-major (C). */
//    private final boolean columnMajor;

    /** Mutable view onto the underlying memory buffer. */
    private final LittleEndianNativeArrayView nativeView;

    public SparseMatrix(AbstractGrCUDAExecutionContext grCUDAExecutionContext, Type elementType, long dimRows, long dimCols, DeviceArray rowIdx,
                        DeviceArray colIdx, DeviceArray nnzValues) {
        super(grCUDAExecutionContext, elementType);
        this.dimRows = dimRows;
        this.dimCols = dimCols;
        System.arraycopy(rowIdx, 0, this.rowIdx, 0, (int) rowIdx.getArraySize());
        System.arraycopy(colIdx, 0, this.colIdx, 0, (int) colIdx.getArraySize());
        System.arraycopy(nnzValues, 0, this.nnzValues, 0, (int) nnzValues.getArraySize());
//        this.rowIdx = rowIdx;
//        this.colIdx = colIdx;
//        this.nnzValues = nnzValues;
        // Allocate the GPU memory;
        this.nativeView = allocateMemory();
        // Register the array in the GrCUDAExecutionContext;
        this.registerArray();
    }

    /**
     * Allocate the GPU memory. It can be overridden to mock the array;
     * @return a reference to the GPU memory
     */
    protected LittleEndianNativeArrayView allocateMemory() {
        return this.grCUDAExecutionContext.getCudaRuntime().cudaMallocManaged(getSizeBytes());
    }

// useless functions
//    private long obtainTotalSize(long[] dimensions) {
//        if (dimensions.length < 2) {
//            CompilerDirectives.transferToInterpreter();
//            throw new IllegalArgumentException(
//                    "MultiDimDeviceArray requires at least two dimension, use DeviceArray instead");
//        }
//        // check arguments
//        long prod = 1;
//        for (long n : dimensions) {
//            if (n < 1) {
//                CompilerDirectives.transferToInterpreter();
//                throw new IllegalArgumentException("invalid size of dimension " + n);
//            }
//            prod *= n;
//        }
//        return prod;
//    }

//    private static long[] computeStride(long[] dimensions, boolean columnMajor) {
//        long prod = 1;
//        long[] stride = new long[dimensions.length];
//        if (columnMajor) {
//            for (int i = 0; i < dimensions.length; i++) {
//                stride[i] = prod;
//                prod *= dimensions[i];
//            }
//        } else {
//            for (int i = dimensions.length - 1; i >= 0; i--) {
//                stride[i] = prod;
//                prod *= dimensions[i];
//            }
//        }
//        return stride;
//    }

//    public final int getNumberDimensions() {
//        return elementsPerDimension.length;
//    }
//    public final long[] getShape() {
//        long[] shape = new long[elementsPerDimension.length];
//        System.arraycopy(elementsPerDimension, 0, shape, 0, elementsPerDimension.length);
//        return shape;
//    }

    public final long getElementsInDimension(int dimension) {
        if (dimension < 0 || dimension >= 2) {
            CompilerDirectives.transferToInterpreter();
            throw new IllegalArgumentException("invalid dimension index ");
        } else if (dimension == 0){
            return this.dimRows;
        } else {
            return this.dimCols;
        }
    }

//    public long getStrideInDimension(int dimension) {
//        if (dimension < 0 || dimension >= stridePerDimension.length) {
//            CompilerDirectives.transferToInterpreter();
//            throw new IllegalArgumentException("invalid dimension index " + dimension + ", valid [0, " + stridePerDimension.length + ']');
//        }
//        return stridePerDimension[dimension];
//    }

    final boolean isIndexValidInDimension(long row, long col) throws InvalidArrayIndexException, UnsupportedMessageException {
        // returns true if the index corresponds to nnz element
        boolean check = false;
        // check rows
        for(int i = 0; i < this.nnzValues.getArraySize(); i++){
            if(((long) this.rowIdx.readArrayElement(i) == row)&&((long) this.colIdx.readArrayElement(i) == col)){
                check = true;
            }
        }
        return check;
    }

//    @Override
//    public final boolean isColumnMajorFormat() {
//        return columnMajor;
//    }

    long getNumElements() {
        // returns the number of nnz elements
        return nnzValues.getArraySize();
    }

    @Override
    final public long getSizeBytes() {
        return nnzValues.getArraySize() * elementType.getSizeBytes() * 3; // must be same size
    }

    @Override
    public final long getPointer() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return nativeView.getStartAddress();
    }

    final LittleEndianNativeArrayView getNativeView() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return nativeView;
    }

    @Override
    public String toString() {
        return "SparseMatrixCOOFormat(elementType=" + elementType +
                        ", rows=" + dimRows +
                        ", cols=" + dimCols +
                        ", nnzElements=" + nnzValues.getArraySize() +
                        ", size=" + getSizeBytes() + " bytes" +
                        ", nativeView=" + nativeView + ')';
    }

    @Override
    protected void finalize() throws Throwable {
        if (!arrayFreed) {
            grCUDAExecutionContext.getCudaRuntime().cudaFree(nativeView);
        }
        super.finalize();
    }

    @Override
    public void freeMemory() {
        if (arrayFreed) {
            throw new GrCUDAException("device array already freed");
        }
        grCUDAExecutionContext.getCudaRuntime().cudaFree(nativeView);
        arrayFreed = true;
    }

    /**
     * Direct access to the native view underlying the multidimensional array;
     * @param index index used to access the array
     * @param elementTypeProfile type of the array
     * @return value read from the array
     */
    @Override
    public Object readNativeView(long index,
                                 @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) {
        return AbstractArray.readArrayElementNative(this.nativeView, index, this.elementType, elementTypeProfile);
    }

    /**
     * Direct access to the native view underlying the multidimensional array;
     * @param index index used to access the array
     * @param value value to write in the array
     * @param valueLibrary interop access of the value, required to understand its type
     * @param elementTypeProfile profiling of the element type, to speed up the native view access
     * @throws UnsupportedTypeException if writing the wrong type in the array
     */
    @Override
    public void writeNativeView(long index, Object value,
                                @CachedLibrary(limit = "3") InteropLibrary valueLibrary,
                                @Cached.Shared("elementType") @Cached("createIdentityProfile()") ValueProfile elementTypeProfile) throws UnsupportedTypeException {
        AbstractArray.writeArrayElementNative(this.nativeView, index, value, this.elementType, valueLibrary, elementTypeProfile);
    }

    //
    // Implementation of InteropLibrary
    //

    @ExportMessage
    @SuppressWarnings("static-method")
    @Override
    public long getArraySize() {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        return nnzValues.getArraySize();
    }

    @ExportMessage
    @SuppressWarnings("static-method")
//    @Override
    boolean isArrayElementReadable(long row, long col) {
        return row >= 0 && col >=0 && row < dimRows && col < dimCols; // questa cosa non so risolverla
    }

    @ExportMessage
//    @Override
    Object readArrayElement(long row, long col) throws InvalidArrayIndexException, UnsupportedMessageException {
        if (arrayFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException(ACCESSED_FREED_MEMORY_MESSAGE);
        }
        if ((row < 0) || (row >= dimRows) || (col < 0) || (col >= dimCols)) {
            CompilerDirectives.transferToInterpreter();
            throw InvalidArrayIndexException.create(row);
        }
//        long offset = index * stridePerDimension[0];
        Object element = 0;
        if(isIndexValidInDimension(row, col)){
            for(int i = 0; i < this.nnzValues.getArraySize(); i++){
                if(((long) rowIdx.readArrayElement(i) == row) && ((long) colIdx.readArrayElement(i)==col)){
                    element = nnzValues.readArrayElement(i);
                }
            }
        }
        return element;
    }
}
