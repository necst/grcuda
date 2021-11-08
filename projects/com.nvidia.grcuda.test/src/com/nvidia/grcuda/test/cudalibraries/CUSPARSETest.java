/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda.test.cudalibraries;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotException;
import org.graalvm.polyglot.Value;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;

@RunWith(Parameterized.class)
public class CUSPARSETest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                        {ExecutionPolicyEnum.SYNC.getName(), ExecutionPolicyEnum.ASYNC.getName()},
                        {true, false}
// {'S', 'D', 'C', 'Z'}
        }));
    }

    private final String policy;
    private final boolean inputPrefetch;
// private final char typeChar;

    public CUSPARSETest(String policy, boolean inputPrefetch) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
// this.typeChar = typeChar;
    }

    /**
     * SPARSE Level-1 Test.
     */

    @Test
    public void testSpMV() {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
                        true).build()) {

            int numDim = 1000;
            int numElements = numDim;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            Value bufferSize = cu.invokeMember("DeviceArray", "float", 1);
            Value dnVecXDescr = cu.invokeMember("DeviceArray", "long", 1);
            Value dnVecYDescr = cu.invokeMember("DeviceArray", "long", 1);
            Value spMatDescr = cu.invokeMember("DeviceArray", "long", 1);
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);

            // handle not use with the current specifications (see registry)
//            UnsafeHelper.Integer64Object cusparseHandle = UnsafeHelper.createInteger64Object();

            // variables as generic elements
//            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
//            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
//            UnsafeHelper.Integer64Object spMatDescr = UnsafeHelper.createInteger64Object();
//            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
//            UnsafeHelper.Float32Object alpha = UnsafeHelper.createFloat32Object();
//            UnsafeHelper.Float32Object beta = UnsafeHelper.createFloat32Object();
//            alpha.setValue(1.0F);
//            beta.setValue(0.0F);
//            bufferSize.setValue(0);

            // variables initialization
            alpha.setArrayElement(0, 1);
            beta.setArrayElement(0, 0);
            bufferSize.setArrayElement(0,0);

            // initial checks
            assertEquals(numElements, coordX.getArraySize());
            assertEquals(numElements, coordY.getArraySize());
            assertEquals(numElements, nnzVec.getArraySize());

            // populating arrays
            float edge_value = (float) Math.random();

            for (int i = 0; i < numElements; ++i) {
                coordX.setArrayElement(i, i);
                coordY.setArrayElement(i, i);
                nnzVec.setArrayElement(i, edge_value);
                dnVec.setArrayElement(i, 1.0);
                outVec.setArrayElement(i, 0.0);
            }

            // TODO: find a way to check whether the output of those functions is correct or not

            // cusparseCreateDnVec
            Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
            cusparseCreateDnVec.execute(dnVecXDescr.asNativePointer(), numElements, dnVec, CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal());
            cusparseCreateDnVec.execute(dnVecYDescr.asNativePointer(), numElements, outVec, CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal());

            // cusparseCreateCoo
            Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
            cusparseCreateCoo.execute(spMatDescr.asNativePointer(), numElements, numElements, numElements, coordX, coordY, nnzVec,
                            CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(), CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                        CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal());

            // cusparseSpMV_buffersize
            Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
            cusparseSpMV_bufferSize.execute(
//                    cusparseHandle.getValue(),
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    spMatDescr,
                    dnVecXDescr,
                    beta,
                    dnVecYDescr,
                    CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal(),
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal(),
                    bufferSize
            );

            // cusparseSpMV
            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");
            cusparseSpMV.execute(
                    // no need to pass the handle
//                    cusparseHandle.getValue(),
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    spMatDescr,
                    dnVecXDescr,
                    beta,
                    dnVecYDescr,
                    CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal(),
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal(),
                    bufferSize
            );
        }
    }
}
