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
            Value cu = polyglot.eval("grcuda", "CU");
            int numDim = 1000;
            int numElements = numDim;
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            Value bufferSize = cu.invokeMember("DeviceArray", "float", 1);
            alpha.setArrayElement(0, 1);
            beta.setArrayElement(0, 0);
            bufferSize.setArrayElement(0,0);
            // creating handle
            UnsafeHelper.Integer64Object cusparseHandle = UnsafeHelper.createInteger64Object();
            // creating COO matrix
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
            // creating descriptors
            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object spMatDescr = UnsafeHelper.createInteger64Object();
//            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
//            UnsafeHelper.Float32Object alpha = UnsafeHelper.createFloat32Object();
//            UnsafeHelper.Float32Object beta = UnsafeHelper.createFloat32Object();
//            alpha.setValue(1.0F);
//            beta.setValue(0.0F);


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

            // creating matrix and vectors descriptors (from cusparse)
            // TODO: find a way to assert that the output of those functions is correct
            Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
            System.out.println("POLIPO");
            cusparseCreateDnVec.execute(dnVecXDescr.getAddress(), numElements, dnVec, CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal());
            cusparseCreateDnVec.execute(dnVecYDescr.getAddress(), numElements, outVec, CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal());
            Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
            cusparseCreateCoo.execute(spMatDescr.getAddress(), numElements, numElements, numElements, coordX, coordY, nnzVec,
                            CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(), CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                        CUSPARSERegistry.cudaDataType.CUDA_C_32F.ordinal());
            Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
            try{
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
            } catch (PolyglotException e){
                e.printStackTrace();
                System.out.println(e.getMessage());
                System.out.println(e.getSourceLocation());
                System.out.println(e.isGuestException());
                System.out.println(e.isHostException());
            }


            
            

            // create all inputs for a function
// Value taxpy = polyglot.eval("grcuda", "SPARSE::cusparse" + typeChar + "axpy");
// taxpy.execute(numDim, alpha, coordX, 1, coordY, 1);
// assertOutputVectorIsCorrect(numElements, coordY, (Integer i) -> i);
        }
    }
//
// /**
// * BLAS Level-2 Test.
// */
// @Test
// public void testTgemv() {
// try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy",
// this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
// true).build()) {
// Value cu = polyglot.eval("grcuda", "CU");
// int numDim = 10;
// boolean isComplex = (typeChar == 'C') || (typeChar == 'Z');
// String cudaType = ((typeChar == 'D') || (typeChar == 'Z')) ? "double" : "float";
// int numElements = isComplex ? numDim * 2 : numDim;
// Value alpha = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
// Value beta = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
// alpha.setArrayElement(0, -1);
// beta.setArrayElement(0, 2);
// if (isComplex) {
// alpha.setArrayElement(1, 0);
// beta.setArrayElement(1, 0);
// }
//
// // complex types require two elements along 1st dimension (since column-major order)
// Value matrixA = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
// Value x = cu.invokeMember("DeviceArray", cudaType, numElements);
// Value y = cu.invokeMember("DeviceArray", cudaType, numElements);
//
// // set matrix
// // A: identity matrix
// for (int j = 0; j < numDim; j++) {
// for (int i = 0; i < numElements; i++) {
// // complex types require two elements along 1st dimension (since column-major
// // order)
// Value row = matrixA.getArrayElement(i);
// row.setArrayElement(j, ((!isComplex & (i == j)) || (isComplex && (i == (2 * j)))) ? 1.0 : 0.0);
// }
// }
//
// // set vectors
// // x = (1, 2, ..., numDim)
// // y = (1, 2, ..., numDim)
// for (int i = 0; i < numElements; i++) {
// x.setArrayElement(i, i);
// y.setArrayElement(i, i);
// }
// Value tgemv = polyglot.eval("grcuda", "BLAS::cublas" + typeChar + "gemv");
// final int cublasOpN = 0;
// tgemv.execute(cublasOpN, numDim, numDim,
// alpha,
// matrixA, numDim,
// x, 1,
// beta,
// y, 1);
// assertOutputVectorIsCorrect(numElements, y, (Integer i) -> i);
// }
// }
//
// /**
// * BLAS Level-3 Test.
// */
// @Test
// public void testTgemm() {
// try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy",
// this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
// true).build()) {
// Value cu = polyglot.eval("grcuda", "CU");
// int numDim = 10;
// boolean isComplex = (typeChar == 'C') || (typeChar == 'Z');
// String cudaType = ((typeChar == 'D') || (typeChar == 'Z')) ? "double" : "float";
// int numElements = isComplex ? numDim * 2 : numDim;
// Value alpha = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
// Value beta = cu.invokeMember("DeviceArray", cudaType, isComplex ? 2 : 1);
// alpha.setArrayElement(0, -1);
// beta.setArrayElement(0, 2);
// if (isComplex) {
// alpha.setArrayElement(1, 0);
// beta.setArrayElement(1, 0);
// }
//
// // complex types require two elements along 1st dimension (since column-major order)
// Value matrixA = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
// Value matrixB = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
// Value matrixC = cu.invokeMember("DeviceArray", cudaType, numElements, numDim, "F");
//
// // set matrix
// // A: identity matrix
// for (int j = 0; j < numDim; j++) {
// for (int i = 0; i < numElements; i++) {
// // complex types require two elements along 1st dimension (since column-major
// // order)
// Value row = matrixA.getArrayElement(i);
// row.setArrayElement(j, ((!isComplex & (i == j)) || (isComplex && (i == (2 * j)))) ? 1.0 : 0.0);
// }
// }
// // B == C
// for (int j = 0; j < numDim; j++) {
// for (int i = 0; i < numElements; i++) {
// Value row = matrixB.getArrayElement(i);
// row.setArrayElement(j, i + numElements * j);
// }
// }
// for (int j = 0; j < numDim; j++) {
// for (int i = 0; i < numElements; i++) {
// Value row = matrixC.getArrayElement(i);
// row.setArrayElement(j, i + numElements * j);
// }
// }
// Value tgemm = polyglot.eval("grcuda", "BLAS::cublas" + typeChar + "gemm");
// final int cublasOpN = 0;
// tgemm.execute(cublasOpN, cublasOpN, numDim, numDim, numDim,
// alpha,
// matrixA, numDim,
// matrixB, numDim,
// beta,
// matrixC, numDim);
// assertOutputMatrixIsCorrect(numDim, numElements, matrixC, (Integer i) -> i);
// }
// }
//
// /**
// * Validation function for vectors.
// */
// public static void assertOutputVectorIsCorrect(int len, Value deviceArray,
// Function<Integer, Integer> outFunc, char typeChar) {
// boolean hasDouble = (typeChar == 'D') || (typeChar == 'Z');
// for (int i = 0; i < len; i++) {
// if (hasDouble) {
// double expected = outFunc.apply(i);
// double actual = deviceArray.getArrayElement(i).asDouble();
// assertEquals(expected, actual, 1e-5);
// } else {
// float expected = outFunc.apply(i);
// float actual = deviceArray.getArrayElement(i).asFloat();
// assertEquals(expected, actual, 1e-5f);
// }
// }
// }
//
// private void assertOutputVectorIsCorrect(int len, Value deviceArray,
// Function<Integer, Integer> outFunc) {
// CUSPARSETest.assertOutputVectorIsCorrect(len, deviceArray, outFunc, this.typeChar);
// }
//
// /**
// * Validation function for matrix.
// */
// private void assertOutputMatrixIsCorrect(int numDim, int numElements, Value matrix,
// Function<Integer, Integer> outFunc) {
// boolean hasDouble = (typeChar == 'D') || (typeChar == 'Z');
// for (int j = 0; j < numDim; j++) {
// for (int i = 0; i < numElements; i++) {
// int idx = i + numElements * j;
// if (hasDouble) {
// double expected = outFunc.apply(idx);
// double actual = matrix.getArrayElement(i).getArrayElement(j).asDouble();
// assertEquals(expected, actual, 1e-5);
// } else {
// float expected = outFunc.apply(idx);
// float actual = matrix.getArrayElement(i).getArrayElement(j).asFloat();
// assertEquals(expected, actual, 1e-5f);
// }
// }
// }
}
