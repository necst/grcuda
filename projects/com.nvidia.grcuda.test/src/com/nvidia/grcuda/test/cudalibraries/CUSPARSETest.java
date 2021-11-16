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
package com.nvidia.grcuda.test.cudalibraries;

import static com.nvidia.grcuda.functions.Function.expectInt;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.util.Arrays;
import java.util.Collection;

import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import org.graalvm.polyglot.Context;
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
                        {true, false},
        }));
    }

//    GrCUDAOptions.CuSPARSEEnabled
    private final String policy;
    private final boolean inputPrefetch;

    public CUSPARSETest(String policy, boolean inputPrefetch) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
    }

    /**
     * SPARSE SpMV function test with COO matrix.
     */

    @Test
    public void SpMV_COO(){
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option("grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(
                true).build()) {

            //option("grcuda.CuSPARSEEnabled", String.valueOf(true))
            int numElements = 1000;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);

            // variables initialization
            alpha.setArrayElement(0, 1);
            beta.setArrayElement(0, 0);

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


            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");

            // order of the arguments should be the following
            cusparseSpMV.execute(
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    numElements,
                    numElements,
                    numElements,
                    coordX,
                    coordY,
                    nnzVec,
                    CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
                    CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    dnVec,
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    beta,
                    outVec,
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal()
            );

            for (int i = 0; i < numElements; i++) {
                assertEquals(outVec.getArrayElement(i).asFloat(), edge_value, 1e-5);
            }
        }
    }

    /**
     * SPARSE SpMV function test with CSR matrix.
     */

//    @Test
//    public void SpMV_CSR(){
//        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option("grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(
//                true).build()) {
//
//            int numElements = 1000;
//
//            // creating context variables
//            Value cu = polyglot.eval("grcuda", "CU");
//
//            // creating variables for cusparse functions as DeviceArrays
//            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
//            Value beta = cu.invokeMember("DeviceArray", "float", 1);
//            Value coordX = cu.invokeMember("DeviceArray", "int", numElements + 1);
//            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
//            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
//            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
//            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
//
//            // variables initialization
//            alpha.setArrayElement(0, 1);
//            beta.setArrayElement(0, 0);
//
//            // populating arrays
//            float edge_value = (float) Math.random();
//
//            coordX.setArrayElement(0, 1);
//            for (int i = 0; i < numElements; ++i) {
//                coordX.setArrayElement(i + 1, i + 1);
//                coordY.setArrayElement(i, i);
//                nnzVec.setArrayElement(i, edge_value);
//                dnVec.setArrayElement(i, 1.0);
//                outVec.setArrayElement(i, 0.0);
//            }
//
//            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");
//
//            // order of the arguments should be the following
//            cusparseSpMV.execute(
//                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
//                    alpha,
//                    numElements,
//                    numElements,
//                    numElements,
//                    coordX,
//                    coordY,
//                    nnzVec,
//                    CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
//                    CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
//                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
//                    dnVec,
//                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
//                    beta,
//                    outVec,
//                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal()
//            );
//
//            for (int i = 0; i < numElements; i++) {
//                assertEquals(outVec.getArrayElement(i).asFloat(), edge_value, 1e-5);
//            }
//        }
//    }

    /**
     * SPARSE Sgemvi function test
     */

    @Test
    public void Sgemvi(){

        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option("grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(
                true).build()) {

            //option("grcuda.CuSPARSEEnabled", String.valueOf(true))
            int numElements = 1000;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value A = cu.invokeMember("DeviceArray", "", numElements, numElements);

            // variables initialization
            alpha.setArrayElement(0, 1);
            beta.setArrayElement(0, 0);

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


            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");

            // order of the arguments should be the following
//            transA, m, n, alpha, A, lda, x, xInd, beta, y, idxBase
            cusparseSpMV.execute(
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    numElements,
                    numElements,
                    numElements,
                    coordX,
                    coordY,
                    nnzVec,
                    CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
                    CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    dnVec,
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    beta,
                    outVec,
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal()
            );

            for (int i = 0; i < numElements; i++) {
                assertEquals(outVec.getArrayElement(i).asFloat(), edge_value, 1e-5);
            }
        }
    }

}
