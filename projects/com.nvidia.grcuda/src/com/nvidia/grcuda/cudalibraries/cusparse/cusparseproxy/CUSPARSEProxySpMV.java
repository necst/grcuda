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

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUDADataType;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUSPARSESpMVAlg;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrix;
import com.nvidia.grcuda.runtime.computation.ComputationArgument;
import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxySpMV extends CUSPARSEProxy {

    public enum CUSPARSESpMVMatrixType {
        SPMV_MATRIX_TYPE_COO,
        SPMV_MATRIX_TYPE_CSR
    }

    private final int nArgsRaw = 11; // args for library function

    public CUSPARSEProxySpMV(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException, UnsupportedMessageException, ArityException {
        this.initializeNfi();
        if (rawArgs.length == nArgsRaw) {
            return rawArgs;
        } else {
            args = new Object[nArgsRaw];

            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();

            CUSPARSERegistry.CUSPARSEOperation opA = CUSPARSERegistry.CUSPARSEOperation.values()[(int)rawArgs[0]];
            Float alphaVal = (Float) rawArgs[1];
            SparseMatrix sparseMatrix = (SparseMatrix) rawArgs[2];
            AbstractArray vecXData = (AbstractArray) rawArgs[3];
            CUDADataType valueTypeVec = CUDADataType.values()[(int)rawArgs[4]];
            Float betaVal = (Float) rawArgs[5];
            AbstractArray vecYData = (AbstractArray) rawArgs[6];
            CUSPARSESpMVAlg alg = CUSPARSESpMVAlg.values()[(int)rawArgs[7]];

            final long cols = sparseMatrix.getRows();
            CUDADataType valueType = sparseMatrix.getDataType();
            UnsafeHelper.Integer64Object matDescr = sparseMatrix.getSpMatDescr();

            final boolean isComplex = sparseMatrix.getIsComplex();
            
            UnsafeHelper.Integer64Object vecXDesc = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object vecYDesc = UnsafeHelper.createInteger64Object();

            UnsafeHelper.Float32Object alpha = UnsafeHelper.createFloat32Object();
            UnsafeHelper.Float32Object beta = UnsafeHelper.createFloat32Object();

            alpha.setValue(alphaVal.floatValue());
            beta.setValue(betaVal.floatValue());
            INTEROP.execute(cusparseCreateDnVecFunctionNFI,
                                vecXDesc.getAddress(),
                                vecXData.getArraySize(),
                                vecXData,
                                CUSPARSERegistry.CUDADataType.fromGrCUDAType(vecXData.getElementType(), isComplex).ordinal());

            INTEROP.execute(cusparseCreateDnVecFunctionNFI,
                                vecYDesc.getAddress(),
                                vecYData.getArraySize(),
                                vecYData,
                                CUSPARSERegistry.CUDADataType.fromGrCUDAType(vecYData.getElementType(), isComplex).ordinal());
            /*
            // create buffer
            Object resultBufferSize = INTEROP.execute(cusparseSpMV_bufferSizeFunctionNFI, handle, opA.ordinal(), 
                    alpha.getAddress(),
                    matDescr.getValue(),
                    vecXDesc.getValue(),
                    beta.getAddress(),
                    vecYDesc.getValue(),
                    valueType.ordinal(), alg.ordinal(), bufferSize.getAddress());

            //sparseMatrix.getValues().getGrCUDAExecutionContext().getCudaRuntime().cudaDeviceSynchronize();

            long numElements;

            if (bufferSize.getValue() == 0) {
                numElements = 1;
            } else {
                numElements = bufferSize.getValue();
            }

            System.out.println(numElements);
            DeviceArray buffer = new DeviceArray(sparseMatrix.getValues().getGrCUDAExecutionContext(), numElements, Type.UINT8);
*/
            // format new arguments
            args[0] = opA.ordinal();
            args[1] = alpha.getAddress();
            args[2] = matDescr.getValue();
            args[3] = vecXDesc.getValue();
            args[4] = beta.getAddress();
            args[5] = vecYDesc.getValue();
            args[6] = valueType.ordinal();
            args[7] = alg.ordinal();
            args[8] = Long.valueOf(0);
            //additional arguments for dependency tracking
            args[9] = new ComputationArgumentWithValue(
                    "input_tracker", Type.NFI_POINTER, ComputationArgument.Kind.POINTER_IN,
                    vecXData);
            args[10] = new ComputationArgumentWithValue(
                    "output_tracker", Type.NFI_POINTER, ComputationArgument.Kind.POINTER_INOUT,
                    vecYData);
            
            return args;
        }
    }

    @Override
    public boolean requiresHandle() {
        return true;
    }
}
