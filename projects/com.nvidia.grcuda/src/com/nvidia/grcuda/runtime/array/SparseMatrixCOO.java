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
import com.nvidia.grcuda.MemberSet;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.profiles.ValueProfile;

@ExportLibrary(InteropLibrary.class)
public class SparseMatrixCOO implements TruffleObject {

    private static final String UNSUPPORTED_WRITE_ON_SPARSE_DS = "unsupported write on sparse data structures";
    private static final String UNSUPPORTED_DIRECT_READ_ON_SPARSE_DS = "unsupported direct read on sparse data structures";
    private boolean matrixFreed;

    /**
     * Row and column dimensions
     */
    private final long dimRows;
    private final long dimCols;

    /**
     * Attributes to be exported
     */

    protected static final String FREE = "free";
    protected static final String IS_MEMORY_FREED = "isMemoryFreed";
    protected static final String VALUES = "values";
    protected static final String ROW_INDICES = "rowIndices";
    protected static final String COL_INDICES = "colIndices";

    /**
     * Row and column indices of nnz elements.
     */
    private final DeviceArray rowIndices;
    private final DeviceArray colIndices;

    /**
     * Values of nnz elements.
     */
    private final DeviceArray values;

    public DeviceArray getRowIndices() {
        return rowIndices;
    }

    public DeviceArray getColIndices() {
        return colIndices;
    }

    public DeviceArray getValues() {
        return values;
    }

    protected static final MemberSet MEMBERS = new MemberSet(FREE, IS_MEMORY_FREED, VALUES, ROW_INDICES, COL_INDICES);

    public SparseMatrixCOO(AbstractGrCUDAExecutionContext grCUDAExecutionContext, DeviceArray rowIdx, DeviceArray colIdx, DeviceArray nnzValues, long dimRows, long dimCols) {
        this.dimRows = dimRows;
        this.dimCols = dimCols;
        this.rowIndices = rowIdx;
        this.colIndices = colIdx;
        this.values = nnzValues;
    }

    private void checkFreeMatrix() {
        if (matrixFreed) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("Matrix freed already");
        }
    }

    final public long getSizeBytes() {
        checkFreeMatrix();
        return values.getSizeBytes() + rowIndices.getSizeBytes() + colIndices.getSizeBytes();
    }

    @ExportMessage
    final long getArraySize() throws UnsupportedMessageException {
        throw new GrCUDAException("Matrix has no Array Size");
    }

    protected void finalize() throws Throwable {
        if (!matrixFreed) {
            this.freeMemory();
        }
        super.finalize();
    }

    @Override
    public String toString() {
        return "SparseMatrixCOO{" +
                "matrixFreed=" + matrixFreed +
                ", dimRows=" + dimRows +
                ", dimCols=" + dimCols +
                ", rowIndices=" + rowIndices +
                ", colIndices=" + colIndices +
                ", values=" + values +
                '}';
    }

    public void freeMemory() {
        checkFreeMatrix();
        values.freeMemory();
        rowIndices.freeMemory();
        colIndices.freeMemory();
        matrixFreed = true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementModifiable(long index) {
        return !matrixFreed && index >= 0 && index < values.getArraySize();
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isArrayElementReadable(long index) {
        return !matrixFreed && isArrayElementModifiable(index);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean hasMembers() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    Object getMembers(@SuppressWarnings("unused") boolean includeInternal) {
        return MEMBERS;
    }

    @SuppressWarnings("static-method")
    @ExportMessage
    boolean isArrayElementInsertable(@SuppressWarnings("unused") long index) {
        return false;
    }

    @ExportMessage
    final boolean hasArrayElements() {
        return true;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberReadable(String memberName,
                             @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) {
        String name = memberProfile.profile(memberName);
        return FREE.equals(name) || IS_MEMORY_FREED.equals(name) || VALUES.equals(name) || ROW_INDICES.equals(name) || COL_INDICES.equals(name);
    }

    @ExportMessage
    Object readMember(String memberName,
                      @Cached.Shared("memberName") @Cached("createIdentityProfile()") ValueProfile memberProfile) throws UnknownIdentifierException {
        if (!isMemberReadable(memberName, memberProfile)) {
            CompilerDirectives.transferToInterpreter();
            throw UnknownIdentifierException.create(memberName);
        }
        if (FREE.equals(memberName)) {
            return new SparseMatrixCOOFreeFunction();
        }


        if (IS_MEMORY_FREED.equals(memberName)) {
            return matrixFreed;
        }

        if(VALUES.equals(memberName)){
            return getValues();
        }

        if(COL_INDICES.equals(memberName)){
            return getColIndices();
        }

        if(ROW_INDICES.equals(memberName)){
            return getRowIndices();
        }

        CompilerDirectives.transferToInterpreter();
        throw UnknownIdentifierException.create(memberName);
    }

    @ExportMessage
    final void writeArrayElement(@SuppressWarnings("unused") long index, @SuppressWarnings("unused") Object value) throws GrCUDAException {
        throw new GrCUDAException(UNSUPPORTED_WRITE_ON_SPARSE_DS);
    }

    @ExportMessage
    final Object readArrayElement(@SuppressWarnings("unused") long index) throws GrCUDAException {
        throw new GrCUDAException(UNSUPPORTED_DIRECT_READ_ON_SPARSE_DS);
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isMemberInvocable(String memberName) {
        return FREE.equals(memberName);
    }

    @ExportMessage
    Object invokeMember(String memberName,
                        Object[] arguments,
                        @CachedLibrary("this") InteropLibrary interopRead,
                        @CachedLibrary(limit = "1") InteropLibrary interopExecute)
            throws UnsupportedTypeException, ArityException, UnsupportedMessageException, UnknownIdentifierException {
        return interopExecute.execute(interopRead.readMember(this, memberName), arguments);
    }

    @ExportLibrary(InteropLibrary.class)
    final class SparseMatrixCOOFreeFunction implements TruffleObject {
        @ExportMessage
        @SuppressWarnings("static-method")
        boolean isExecutable() {
            return true;
        }

        @ExportMessage
        Object execute(Object[] arguments) throws ArityException {
            checkFreeMatrix();
            if (arguments.length != 0) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(0, arguments.length);
            }
            freeMemory();
            return NoneValue.get();
        }
    }
}
