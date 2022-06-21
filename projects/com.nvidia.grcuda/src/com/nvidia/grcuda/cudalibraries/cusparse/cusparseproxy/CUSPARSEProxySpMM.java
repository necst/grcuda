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

import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUDADataType;
import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrix;
import com.nvidia.grcuda.runtime.array.SparseVector;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxySpMM extends CUSPARSEProxy {

    private final int nArgsRaw = 10; // args for library function

    public CUSPARSEProxySpMM(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException, UnsupportedMessageException, ArityException {
        this.initializeNfi();
        args = new Object[nArgsRaw];

        UnsafeHelper.Integer64Object matBDescr = UnsafeHelper.createInteger64Object();
        UnsafeHelper.Integer64Object matCDescr = UnsafeHelper.createInteger64Object();

        CUSPARSERegistry.CUSPARSEOperation opA = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(rawArgs[0])];
        CUSPARSERegistry.CUSPARSEOperation opB = CUSPARSERegistry.CUSPARSEOperation.values()[expectInt(rawArgs[1])];
        SparseVector alpha = (SparseVector) rawArgs[2];
        SparseMatrix matA = (SparseMatrix) rawArgs[3];
        DeviceArray matB = (DeviceArray) rawArgs[4];
        DeviceArray beta = (DeviceArray) rawArgs[5];
        DeviceArray matC = (DeviceArray) rawArgs[6];
        int alg = expectInt(rawArgs[7]);

        //CUDADataType valueType = vecX.getDataType();
        //UnsafeHelper.Integer64Object vecXDescr = vecX.getSpVecDescr();

        //long size = vecX.getN();

        // create dense vectors X and Y descriptors
        /*INTEROP.execute(cusparseCreateDnMatFunction, matBDescr.getAddress(), size, vecYData, valueType.ordinal());
        INTEROP.execute(cusparseCreateDnMatFunction, matCDescr.getAddress(), size, vecYData, valueType.ordinal());

        // create buffer
        Object resultBufferSize = INTEROP.execute(cusparseSpVV_bufferSizeFunction, handle, opX.ordinal(), vecX.getSpVecDescr().getValue(),
                            dnVecYDescr.getValue(), result, valueType.ordinal(), bufferSize.getAddress());


        long numElements;

        if (bufferSize.getValue() == 0) {
            numElements = 1;
        } else {
            numElements = (long) bufferSize.getValue() / 4;
        }

        DeviceArray buffer = new DeviceArray(vecYData.getGrCUDAExecutionContext(), numElements, vecYData.getElementType());

        cudaDeviceSynchronize();
        // format new arguments
        args[0] = opX.ordinal();
        args[1] = vecXDescr.getValue();
        args[2] = dnVecYDescr.getValue();
        args[3] = result;
        args[4] = valueType.ordinal();
        args[5] = buffer;
*/
        return args;
    }

    @Override
    public boolean requiresHandle() {
        return true;
    }
}
