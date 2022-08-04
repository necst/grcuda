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
import static com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry.CUSPARSESpMVAlg;
import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectInt;

import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DenseVector;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.SparseMatrix;
import com.nvidia.grcuda.runtime.array.SparseMatrixCSR;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import org.graalvm.polyglot.PolyglotException;

public class CUSPARSEProxySpGEMM extends CUSPARSEProxy {

    private final int nArgsRaw = 10; // args for library function

    public CUSPARSEProxySpGEMM(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException, UnsupportedMessageException, ArityException {
        this.initializeNfi();
        args = new Object[nArgsRaw];

        DeviceArray alpha = (DeviceArray) rawArgs[0];
        SparseMatrixCSR matA = (SparseMatrixCSR) rawArgs[1];
        SparseMatrixCSR matB = (SparseMatrixCSR) rawArgs[2];
        DeviceArray beta = (DeviceArray) rawArgs[3];
        SparseMatrixCSR matC = (SparseMatrixCSR) rawArgs[4];


        UnsafeHelper.Integer64Object spgemmDescr = UnsafeHelper.createInteger64Object();
        UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();


        bufferSize.setValue(1000);
        DeviceArray buffer1 = new DeviceArray(getContext().getGrCUDAExecutionContext(), 1000, Type.SINT8);
        DeviceArray buffer2 = new DeviceArray(getContext().getGrCUDAExecutionContext(), 1000, Type.SINT8);

        INTEROP.execute(cusparseSpGEMM_createDescrFunctionNFI, spgemmDescr.getAddress());
        cudaDeviceSynchronize();
        try {

        INTEROP.execute(
                cusparseSpGEMM_workEstimationFunctionNFI,
                (Long)handle,
                0,
                0,
                alpha,
                matA.getSpMatDescr().getValue(),
                matB.getSpMatDescr().getValue(),
                beta,
                matC.getSpMatDescr().getValue(),
                matA.getDataType().ordinal(),
                0,
                (Long)spgemmDescr.getValue(),
                bufferSize.getAddress(),
                buffer1
        );
        /*   INTEROP.execute(
                    cusparseSpGEMM_workEstimationFunctionNFI,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
            );*/
        } catch (UnsupportedTypeException e) {
            e.printStackTrace();
            for (Object o : e.getSuppliedValues()) {
                System.out.print(o);
                System.out.print(",\t");
            }
            System.out.println("\n");
            System.out.println(handle);
            System.out.println(matA.getSpMatDescr().getValue());
            System.out.println(matB.getSpMatDescr().getValue());
            System.out.println(matC.getSpMatDescr().getValue());
            System.out.println(spgemmDescr.getValue());

        }
        cudaDeviceSynchronize();

        INTEROP.execute(
                    cusparseSpGEMM_computeFunctionNFI,
                    handle,
                    0,
                    0,
                    alpha,
                    matA.getSpMatDescr().getValue(),
                    matB.getSpMatDescr().getValue(),
                    beta,
                    matC.getSpMatDescr().getValue(),
                    matA.getDataType().ordinal(),
                    0,
                    spgemmDescr.getValue(),
                    bufferSize.getAddress(),
                    buffer2
            );

        cudaDeviceSynchronize();
        System.out.println("-----------------------------------------------------------------------------------");

        args[0] = 0;
        args[1] = 0;
        args[2] = alpha;
        args[3] = matA.getSpMatDescr().getValue();
        args[4] = matB.getSpMatDescr().getValue();
        args[5] = beta;
        args[6] = matC.getSpMatDescr().getValue();
        args[7] = matA.getDataType().ordinal();
        args[8] = 0;
        args[9] = spgemmDescr.getValue();

        return args;
    }

    @Override
    public boolean requiresHandle() {
        return true;
    }
}
