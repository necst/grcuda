/*
 * Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.
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
package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;
import static com.nvidia.grcuda.functions.Function.expectLong;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxySpMV extends CUSPARSEProxy {

    private final int nArgsRaw = 9; // args for library function

    public CUSPARSEProxySpMV(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException {
        this.initializeNfi();
        if (rawArgs.length == nArgsRaw) {
            return rawArgs;
        } else {
            args = new Object[nArgsRaw];

            // v1 and v2 can be X, Y, rowPtr
            DeviceArray v1 = (DeviceArray) rawArgs[5];
            DeviceArray v2 = (DeviceArray) rawArgs[6];
            DeviceArray values = (DeviceArray) rawArgs[7];

            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object matDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();

            CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[0])];
            DeviceArray alpha = (DeviceArray) rawArgs[1];
            long rows = expectLong(rawArgs[2]);
            long cols = expectLong(rawArgs[3]);
            long nnz = expectLong(rawArgs[4]);
            CUSPARSERegistry.cusparseIndexType_t idxType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(rawArgs[8])];
            CUSPARSERegistry.cusparseIndexBase_t idxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[9])];
            CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[10])];
            DeviceArray valuesX = (DeviceArray) rawArgs[11];
            CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[12])];
            DeviceArray beta = (DeviceArray) rawArgs[13];
            DeviceArray valuesY = (DeviceArray) rawArgs[14];
            CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cusparseSpMVAlg_t.values()[expectInt(rawArgs[15])];

            if ((v1.getArraySize() == v2.getArraySize()) && (v2.getArraySize() == values.getArraySize())) { // coo

                // create coo matrix descriptor
                try {
                    Object resultCoo = INTEROP.execute(cusparseCreateCooFunction, matDescr.getAddress(), rows, cols, nnz, v1, v2, values, idxType.ordinal(), idxBase.ordinal(),
                                    valueType.ordinal());
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                } // TODO: re-throw an exception if sth goes wrong

            } else { // csr

                // create csr matrix descriptor
                try {
                    Object resultCsr = INTEROP.execute(cusparseCreateCsrFunction, matDescr.getAddress(), rows, cols, nnz, v1, v2, values, idxType.ordinal(),
                                    idxType.ordinal(), idxBase.ordinal(), valueType.ordinal());
// System.out.println("created csr descriptor with output = " + resultCsr);
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                } // TODO: re-throw an exception if sth goes wrong

            }

            // create dense vectors X and Y descriptors
            try {
                Object resultX = INTEROP.execute(cusparseCreateDnVecFunction, dnVecXDescr.getAddress(), cols, valuesX, valueTypeVec.ordinal());
                Object resultY = INTEROP.execute(cusparseCreateDnVecFunction, dnVecYDescr.getAddress(), cols, valuesY, valueTypeVec.ordinal());
            } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                e.printStackTrace();
            }

            // create buffer
            try {
                Object resultBufferSize = INTEROP.execute(cusparseSpMV_bufferSizeFunction, handle, opA.ordinal(), alpha, matDescr.getValue(), dnVecXDescr.getValue(), beta,
                                dnVecYDescr.getValue(), valueType.ordinal(), alg.ordinal(), bufferSize.getAddress());
            } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                e.printStackTrace();
            }

            long numElements;

            if (bufferSize.getValue() == 0) {
                numElements = 1;
            } else {
                numElements = (long) bufferSize.getValue() / 4;
            }

            DeviceArray buffer = new DeviceArray(alpha.getGrCUDAExecutionContext(), numElements, alpha.getElementType());

            // format new arguments
            args[0] = opA.ordinal();
            args[1] = alpha;
            args[2] = matDescr.getValue();
            args[3] = dnVecXDescr.getValue();
            args[4] = beta;
            args[5] = dnVecYDescr.getValue();
            args[6] = valueType.ordinal();
            args[7] = alg.ordinal();
            args[8] = buffer;

            return args;
        }
    }
}
