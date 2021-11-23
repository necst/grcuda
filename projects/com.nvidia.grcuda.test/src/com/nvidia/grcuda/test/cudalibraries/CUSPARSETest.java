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

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;

@RunWith(Parameterized.class)
public class CUSPARSETest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                        {ExecutionPolicyEnum.SYNC.getName(), ExecutionPolicyEnum.ASYNC.getName()},
                        {true, false},
                        {'C', 'S'}
        }));
    }

// GrCUDAOptions.CuSPARSEEnabled
    private final String policy;
    private final boolean inputPrefetch;
    private final char type;

    public CUSPARSETest(String policy, boolean inputPrefetch, char type) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
        this.type = type;
    }

    /**
     * SPARSE SpMV function test with COO matrix.
     */

// @Test
// public void TestSpMV_COO() {
// try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy",
// this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option(
// "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {
//
// // option("grcuda.CuSPARSEEnabled", String.valueOf(true))
// int numElements = 1000;
//
// // creating context variables
// Value cu = polyglot.eval("grcuda", "CU");
//
// // creating variables for cusparse functions as DeviceArrays
// Value alpha = cu.invokeMember("DeviceArray", "float", 1);
// Value beta = cu.invokeMember("DeviceArray", "float", 1);
// Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
// Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
// Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
// Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
// Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
//
// // variables initialization
// alpha.setArrayElement(0, 1);
// beta.setArrayElement(0, 0);
//
// // initial checks
// assertEquals(numElements, coordX.getArraySize());
// assertEquals(numElements, coordY.getArraySize());
// assertEquals(numElements, nnzVec.getArraySize());
//
// // populating arrays
// float edgeValue = (float) Math.random();
//
// for (int i = 0; i < numElements; ++i) {
// coordX.setArrayElement(i, i);
// coordY.setArrayElement(i, i);
// nnzVec.setArrayElement(i, edgeValue);
// dnVec.setArrayElement(i, 1.0);
// outVec.setArrayElement(i, 0.0);
// }
//
// Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");
//
// // order of the arguments should be the following
// cusparseSpMV.execute(
// CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
// alpha,
// numElements,
// numElements,
// numElements,
// coordX,
// coordY,
// nnzVec,
// CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
// CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
// CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
// dnVec,
// CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
// beta,
// outVec,
// CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal());
//
// for (int i = 0; i < numElements; i++) {
// assertEquals(outVec.getArrayElement(i).asFloat(), edgeValue, 1e-5);
// }
// }
// }

    /**
     * SPARSE SpMV function test with CSR matrix.
     */

// @Test
// public void TestSpMV_CSR() {
// try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy",
// this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option(
// "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {
//
// int numElements = 10;
//
// // creating context variables
// Value cu = polyglot.eval("grcuda", "CU");
//
// boolean isComplex = this.type == 'Z' || this.type == 'C';
// boolean isDouble = this.type == 'D' || this.type == 'Z';
//
// String dataType = "float";
// int complexScale = 1;
// if (isComplex) {
// dataType = "double";
// complexScale = 2;
// }
//
// // creating variables for cusparse functions as DeviceArrays
// Value alpha = cu.invokeMember("DeviceArray", dataType, 1 * complexScale);
// Value beta = cu.invokeMember("DeviceArray", dataType, 1 * complexScale);
// Value coordX = cu.invokeMember("DeviceArray", "int", numElements + 1);
// Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
// Value nnzVec = cu.invokeMember("DeviceArray", dataType, numElements * complexScale);
// Value dnVec = cu.invokeMember("DeviceArray", dataType, numElements * complexScale);
// Value outVec = cu.invokeMember("DeviceArray", dataType, numElements * complexScale);
//
// // variables initialization
//
// alpha.setArrayElement(0, 1);
// beta.setArrayElement(0, 0);
// if (isComplex) {
// alpha.setArrayElement(1, 0);
// beta.setArrayElement(1, 0);
// }
//
// // populating arrays
// float edgeValue = (float) Math.random();
//
// coordX.setArrayElement(0, 1);
//
// for (int i = 0; i < numElements; ++i) {
// coordX.setArrayElement(i + 1, (i + 1));
// coordY.setArrayElement(i, i);
// nnzVec.setArrayElement(i * complexScale, edgeValue);
// dnVec.setArrayElement(i * complexScale, 1.0);
// outVec.setArrayElement(i * complexScale, 0.0);
// if (isComplex) {
// nnzVec.setArrayElement(i * complexScale + 1, 0.0);
// dnVec.setArrayElement(i * complexScale + 1, 0.0);
// outVec.setArrayElement(i * complexScale + 1, 0.0);
// }
// }
//
// Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");
//
// int cudaDataType = 0;
//
// switch (this.type) {
// case 'C':
// cudaDataType = CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal();
// break;
// case 'Z':
// cudaDataType = CUSPARSERegistry.cudaDataType.CUDA_C_64F.ordinal();
// break;
// case 'D':
// cudaDataType = CUSPARSERegistry.cudaDataType.CUDA_R_64F.ordinal();
// break;
// case 'S':
// cudaDataType = CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal();
// break;
//
// }
//
// System.out.println("edgevalue = " + edgeValue);
//
// // order of the arguments should be the following
// cusparseSpMV.execute(
// CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
// alpha,
// numElements,
// numElements,
// numElements,
// coordX,
// coordY,
// nnzVec,
// CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
// CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
// cudaDataType,
// dnVec,
// cudaDataType,
// beta,
// outVec,
// CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal());
//
// for (int i = 0; i < numElements*2; i+=2) {
// for (int j = 0; j < complexScale; j++) {
//
// if (isDouble){
//// assertEquals( edgeValue, outVec.getArrayElement(i + j).asDouble(),1e-5);
//
// } else {
//// assertEquals(edgeValue, outVec.getArrayElement(i + j).asFloat(), 1e-5);
// System.out.println(outVec.getArrayElement(i+j).asFloat());
// }
//
//
// }
// }
// }
// }

    /**
     * SPARSE SpMV function test with complex data type and COO matrix
     */

    @Test
    public void TestSpMV_COO() {
        try (Context polyglot = GrCUDATestUtil.buildTestContext()
                .option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch))
                .option("grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {

            // option("grcuda.CuSPARSEEnabled", String.valueOf(true))
            int numElements = 10;

            boolean isComplex = this.type == 'C' || this.type == 'Z';

            int complexScale = 1;

            if (isComplex) complexScale = 2;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            Value alpha = cu.invokeMember("DeviceArray", "float", 1 * complexScale);
            Value beta = cu.invokeMember("DeviceArray", "float", 1 * complexScale);
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements * complexScale);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements * complexScale);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements * complexScale);

            // variables initialization
            alpha.setArrayElement(0, 1);
            beta.setArrayElement(0, 0);

            if (isComplex) {
                alpha.setArrayElement(1, 0);
                beta.setArrayElement(1, 0);
            }

            // initial checks
//            assertEquals(numElements, coordX.getArraySize());
//            assertEquals(numElements, coordY.getArraySize());
//            assertEquals(numElements * complexScale, nnzVec.getArraySize());

            // populating arrays
            float edgeValue = (float) Math.random();

            for (int i = 0; i < numElements; i++) {
                coordX.setArrayElement(i, i * complexScale);
                coordY.setArrayElement(i, i * complexScale);
                nnzVec.setArrayElement(i * complexScale, edgeValue);
                dnVec.setArrayElement(i * complexScale, 1.0);
                outVec.setArrayElement(i * complexScale, 0.0);
                if (isComplex) {
                    outVec.setArrayElement(i * complexScale + 1, 0.0);
                    dnVec.setArrayElement(i * complexScale + 1, 0.0);
                    nnzVec.setArrayElement(i * complexScale + 1, 0.0);
                }
            }

            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");

            int cudaDataType = CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal();

            if (isComplex)
                cudaDataType = CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal();

            // order of the arguments should be the following
            cusparseSpMV.execute(
                            CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                            alpha,
                            numElements * complexScale,
                            numElements * complexScale,
                            numElements * complexScale,
                            coordX,
                            coordY,
                            nnzVec,
                            CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
                            CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                            cudaDataType,
                            dnVec,
                            cudaDataType,
                            beta,
                            outVec,
                            CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal());

            for (int i = 0; i < numElements*complexScale; i += complexScale) {
                System.out.println(outVec.getArrayElement(i).asFloat());
//                assertEquals(edgeValue, outVec.getArrayElement(i).asFloat(), 1e-5);
                if (isComplex) {
//                    assertEquals(0.0, outVec.getArrayElement(i + 1).asFloat(), 1e-5);
                    System.out.println(outVec.getArrayElement(i).asFloat());

                }
            }
        }
    }

    /**
     * SPARSE Sgemvi function test
     */
//
// @Test
// public void TestGemvi() {
//
// try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy",
// this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option(
// "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {
//
// int numElements = 10;
//
// String cudaType = ((this.type == 'D') || (this.type == 'Z')) ? "double" : "float";
// boolean isComplex = this.type == 'C' || this.type == 'Z';
// boolean isDouble = this.type == 'D' || this.type == 'Z';
// // creating context variables
// Value cu = polyglot.eval("grcuda", "CU");
//
// int complexScale = 1;
//
// if (isComplex)
// complexScale = 2;
//
// // creating variables for cusparse functions as DeviceArrays
// Value alpha = cu.invokeMember("DeviceArray", cudaType, 1 * complexScale);
// Value beta = cu.invokeMember("DeviceArray", cudaType, 1 * complexScale);
// int rows = numElements * complexScale; // m
// int cols = numElements * complexScale; // n
// int lda = numElements * complexScale; // leading dim of A
// int nnz = 2; // number of nnz
// Value spVec = cu.invokeMember("DeviceArray", cudaType, nnz * complexScale); // x
// Value outVec = cu.invokeMember("DeviceArray", cudaType, numElements * complexScale); // output
//
// Value matA = cu.invokeMember("DeviceArray", cudaType, numElements * numElements * complexScale);
// // variables initialization
// alpha.setArrayElement(0, 1);
// beta.setArrayElement(0, 0);
//
// if (isComplex) {
// alpha.setArrayElement(1, 0);
// beta.setArrayElement(1, 0);
// }
//
// Value xInd = cu.invokeMember("DeviceArray", "int", nnz); // must be the same
//
// // initialization of outVec not necessary
//
// float edgeValue = (float) Math.random();
//
// // fill sparse vector and related arguments
// for (int i = 0; i < nnz; i++) {
// int idxNnz = (int) (Math.random() * numElements); // to make sure indices are valid
// xInd.setArrayElement(i, idxNnz); // set indices vector
// for (int j = 0; j < complexScale; j++) {
// spVec.setArrayElement(i + j, j == 0 ? 1.0 : 0.0); // set '1' in the
// // corresponding positions of
// }
// }
//
// // fill dense matrix
// for (int i = 0; i < numElements; i++) {
// for (int j = 0; j < numElements; j++) {
// for (int k = 0; k < complexScale; k++) {
// matA.setArrayElement((i * numElements + j) * complexScale + k, k == 0 ? edgeValue : 0.0);
// }
// }
// }
//
// Value cusparseTgemvi = polyglot.eval("grcuda", "SPARSE::cusparse" + this.type + "gemvi");
//
// // order of the arguments should be the following
// // transA, m, n, alpha, A, lda, nnz, x, xInd, beta, y, idxBases
// cusparseTgemvi.execute(
// CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
// rows,
// cols,
// alpha,
// matA,
// lda,
// nnz,
// spVec,
// xInd,
// beta,
// outVec,
// CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
// this.type);
// float expectedResult = nnz * edgeValue;
// for (int i = 0; i < numElements; i++) {
// for (int j = 0; j < complexScale; j++) {
// if(isDouble){
// assertEquals(j == 0 ? expectedResult : 0.0, outVec.getArrayElement(i * complexScale +
// j).asDouble(), 1e-5);
// } else {
// assertEquals(j == 0 ? expectedResult : 0.0, outVec.getArrayElement(i * complexScale +
// j).asFloat(), 1e-5);
// }
// }
// }
// }
// }

    /**
     * Libraries Integration Test
     */

// @Test
// public void TestLibrariesIntegration() {
// // y = M x, z = M v
// // A = z + y, with axpy (a = 1)
// try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy",
// this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).option(
// "grcuda.CuSPARSEEnabled", String.valueOf(true)).allowAllAccess(true).build()) {
// // context creation
// Value cu = polyglot.eval("grcuda", "CU");
//
// // variables creation
// int numElements = 1000;
//
// Value alphaX = cu.invokeMember("DeviceArray", "float", 1);
// Value betaX = cu.invokeMember("DeviceArray", "float", 1);
// Value alphaV = cu.invokeMember("DeviceArray", "float", 1);
// Value betaV = cu.invokeMember("DeviceArray", "float", 1);
// Value coordXX = cu.invokeMember("DeviceArray", "int", numElements);
// Value coordYX = cu.invokeMember("DeviceArray", "int", numElements);
// Value coordXV = cu.invokeMember("DeviceArray", "int", numElements);
// Value coordYV = cu.invokeMember("DeviceArray", "int", numElements);
// Value nnzVecX = cu.invokeMember("DeviceArray", "float", numElements);
// Value nnzVecV = cu.invokeMember("DeviceArray", "float", numElements);
// Value dnVecZ = cu.invokeMember("DeviceArray", "float", numElements);
// Value outVecZ = cu.invokeMember("DeviceArray", "float", numElements);
// Value dnVecY = cu.invokeMember("DeviceArray", "float", numElements);
// Value outVecY = cu.invokeMember("DeviceArray", "float", numElements);
//
// alphaX.setArrayElement(0, 1);
// betaX.setArrayElement(0, 0);
// alphaV.setArrayElement(0, 1);
// betaV.setArrayElement(0, 0);
//
// // initial checks
// assertEquals(numElements, coordXX.getArraySize());
// assertEquals(numElements, coordYX.getArraySize());
// assertEquals(numElements, nnzVecX.getArraySize());
// assertEquals(numElements, coordXV.getArraySize());
// assertEquals(numElements, coordYV.getArraySize());
// assertEquals(numElements, nnzVecV.getArraySize());
//
// // initialization
//
// float edgeValueX = (float) Math.random();
//
// // y = M x
// for (int i = 0; i < numElements; i++) {
// coordXX.setArrayElement(i, i);
// coordYX.setArrayElement(i, i);
// nnzVecX.setArrayElement(i, edgeValueX);
// dnVecY.setArrayElement(i, 1.0);
// outVecY.setArrayElement(i, 0.0);
// }
//
// float edgeValueV = (float) Math.random();
//
// // z = M v
// for (int i = 0; i < numElements; i++) {
// coordXV.setArrayElement(i, i);
// coordYV.setArrayElement(i, i);
// nnzVecV.setArrayElement(i, edgeValueV);
// dnVecZ.setArrayElement(i, 1.0);
// outVecZ.setArrayElement(i, 0.0);
// }
//
// Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");
//
// cusparseSpMV.execute(
// CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
// alphaX,
// numElements,
// numElements,
// numElements,
// coordXX,
// coordYX,
// nnzVecX,
// CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
// CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
// CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
// dnVecY,
// CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
// betaX,
// outVecY,
// CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal());
//
// cusparseSpMV.execute(
// CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
// alphaV,
// numElements,
// numElements,
// numElements,
// coordXV,
// coordYV,
// nnzVecV,
// CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(),
// CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
// CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
// dnVecZ,
// CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
// betaV,
// outVecZ,
// CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal());
//
// Value saxpy = polyglot.eval("grcuda", "BLAS::cublas" + this.type + "axpy");
//
// saxpy.execute(numElements, alphaX, outVecY, 1, outVecZ, 1);
//
// for (int i = 1; i < numElements; i++) {
// assertEquals(outVecZ.getArrayElement(i).asFloat(), edgeValueV + edgeValueX, 1e-5);
// }
// }
// }

}
