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
import static org.junit.Assert.assertNotEquals;

import java.util.Arrays;
import java.util.Collection;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;

@RunWith(Parameterized.class)
public class CUSPARSETest {

    // handle not used in these test, in accordance with the current specifications (see registry)

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                        {ExecutionPolicyEnum.SYNC.getName(), ExecutionPolicyEnum.ASYNC.getName()},
                        {true, false}
        }));
    }

    private final String policy;
    private final boolean inputPrefetch;

    public CUSPARSETest(String policy, boolean inputPrefetch) {
        this.policy = policy;
        this.inputPrefetch = inputPrefetch;
    }

    /**
     * SPARSE Helper Function Test.
     */

    @Test
    public void testCreateDnVec(){
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
                true).build()) {

            int numElements = 1000;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object(); //cu.invokeMember("DeviceArray", "long", 1);
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object(); // cu.invokeMember("DeviceArray", "long", 1);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value ctrlVec = cu.invokeMember("DeviceArray", "float", numElements);

            float edge_value;

            // vectors initialization
            for (int i = 0; i < numElements; ++i) {
                edge_value = (float) Math.random();
                dnVec.setArrayElement(i, edge_value);
                outVec.setArrayElement(i, edge_value);
                ctrlVec.setArrayElement(i, edge_value);
            }

            // cusparseCreateDnVec
            Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
            Value cusparseCreateDnVecXOutputValue = cusparseCreateDnVec.execute(dnVecXDescr.getAddress(), numElements, dnVec, CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());
            Value cusparseCreateDnVecYOutputValue = cusparseCreateDnVec.execute(dnVecYDescr.getAddress(), numElements, outVec, CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());

            assertEquals(cusparseCreateDnVecXOutputValue.asInt(), 0);
            assertEquals(cusparseCreateDnVecYOutputValue.asInt(), 0);

            // check input and output vectors' values' consistency with initialization ones
            for(int i = 0; i < numElements; i++){
                assertEquals(dnVec.getArrayElement(i).asFloat(), ctrlVec.getArrayElement(i).asFloat(), 1e-5);
                assertEquals(outVec.getArrayElement(i).asFloat(), ctrlVec.getArrayElement(i).asFloat(), 1e-5);
            }
        }
    }

    /**
     * SPARSE Helper Function Test
     */

    @Test
    public void testCreateCoo(){
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
                true).build()) {

            int numElements = 1000;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object spMatDescr = UnsafeHelper.createInteger64Object();
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value ctrlVec = cu.invokeMember("DeviceArray", "float", numElements);
            bufferSize.setValue(-1);

            // initial checks
            assertEquals(numElements, coordX.getArraySize());
            assertEquals(numElements, coordY.getArraySize());
            assertEquals(numElements, nnzVec.getArraySize());

            // populating arrays
            float edge_value;

            for (int i = 0; i < numElements; ++i) {
                edge_value = (float) Math.random();
                coordX.setArrayElement(i, i);
                coordY.setArrayElement(i, i);
                nnzVec.setArrayElement(i, edge_value);
                ctrlVec.setArrayElement(i, edge_value);
                dnVec.setArrayElement(i, 1.0);
                outVec.setArrayElement(i, 0.0);
            }

            // cusparseCreateDnVec
            Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
            Value cusparseCreateDnVecXOutputValue = cusparseCreateDnVec.execute(dnVecXDescr.getAddress(), numElements, dnVec, CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());
            Value cusparseCreateDnVecYOutputValue = cusparseCreateDnVec.execute(dnVecYDescr.getAddress(), numElements, outVec, CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());

            assertEquals(cusparseCreateDnVecXOutputValue.asInt(), 0);
            assertEquals(cusparseCreateDnVecYOutputValue.asInt(), 0);

            // cusparseCreateCoo
            // NB: use 64I with complex type
            Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
            Value cusparseCreateCooOutputValue = cusparseCreateCoo.execute(spMatDescr.getAddress(), numElements, numElements, numElements, coordX, coordY, nnzVec,
                    CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_64I.ordinal(), CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());

            // check matrix construction's consistency with the provided initialization values
            assertEquals(cusparseCreateCooOutputValue.asInt(), 0);
            for(int i = 0; i < numElements; i++){
                assertEquals(nnzVec.getArrayElement(i).asFloat(), ctrlVec.getArrayElement(i).asFloat(), 1e-5);
                assertEquals(coordX.getArrayElement(i).asLong(), i);
                assertEquals(coordY.getArrayElement(i).asLong(), i);
            }
        }
    }

    /**
     * SPARSE Helper Function Test
     */

    @Test
    public void testSpMV_bufferSize(){
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
                true).build()) {

            int numElements = 1000;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object spMatDescr = UnsafeHelper.createInteger64Object();
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
            bufferSize.setValue(-1);

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

            // cusparseCreateDnVec
            Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
            Value cusparseCreateDnVecXOutputValue = cusparseCreateDnVec.execute(dnVecXDescr.getAddress(), numElements, dnVec, CUSPARSERegistry.cudaDataType.CUDA_R_64F.ordinal());
            Value cusparseCreateDnVecYOutputValue = cusparseCreateDnVec.execute(dnVecYDescr.getAddress(), numElements, outVec, CUSPARSERegistry.cudaDataType.CUDA_R_64F.ordinal());

            assertEquals(cusparseCreateDnVecXOutputValue.asInt(), 0);
            assertEquals(cusparseCreateDnVecYOutputValue.asInt(), 0);

            // cusparseCreateCoo
            Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
            Value cusparseCreateCooOutputValue = cusparseCreateCoo.execute(spMatDescr.getAddress(), numElements, numElements, numElements, coordX, coordY, nnzVec,
                    CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_64I.ordinal(), CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_64F.ordinal());

            assertEquals(cusparseCreateCooOutputValue.asInt(), 0);

            // cusparseSpMV_buffersize
            Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
            cusparseSpMV_bufferSize.execute(
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    spMatDescr.getValue(),
                    dnVecXDescr.getValue(),
                    beta,
                    dnVecYDescr.getValue(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal(),
                    bufferSize.getAddress()
            );

            long bufferSizeValue = bufferSize.getValue();

            assertNotEquals(bufferSizeValue, -1);
        }
    }

    /**
     * Sparse Level-2 Test
     */

    @Test
    public void testSpMV() {
        try (Context polyglot = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", this.policy).option("grcuda.InputPrefetch", String.valueOf(this.inputPrefetch)).allowAllAccess(
                        true).build()) {

            int numElements = 1000;

            // creating context variables
            Value cu = polyglot.eval("grcuda", "CU");

            // creating variables for cusparse functions as DeviceArrays
            Value alpha = cu.invokeMember("DeviceArray", "float", 1);
            Value beta = cu.invokeMember("DeviceArray", "float", 1);
            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
            UnsafeHelper.Integer64Object spMatDescr = UnsafeHelper.createInteger64Object();
            Value coordX = cu.invokeMember("DeviceArray", "int", numElements);
            Value coordY = cu.invokeMember("DeviceArray", "int", numElements);
            Value nnzVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value dnVec = cu.invokeMember("DeviceArray", "float", numElements);
            Value outVec = cu.invokeMember("DeviceArray", "float", numElements);
            bufferSize.setValue(-1);

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

            // cusparseCreateDnVec
            Value cusparseCreateDnVec = polyglot.eval("grcuda", "SPARSE::cusparseCreateDnVec");
            Value cusparseCreateDnVecXOutputValue = cusparseCreateDnVec.execute(dnVecXDescr.getAddress(), numElements, dnVec, CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());
            Value cusparseCreateDnVecYOutputValue = cusparseCreateDnVec.execute(dnVecYDescr.getAddress(), numElements, outVec, CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());

            assertEquals(cusparseCreateDnVecXOutputValue.asInt(), 0);
            assertEquals(cusparseCreateDnVecYOutputValue.asInt(), 0);

            // cusparseCreateCoo
            Value cusparseCreateCoo = polyglot.eval("grcuda", "SPARSE::cusparseCreateCoo");
            Value cusparseCreateCooOutputValue = cusparseCreateCoo.execute(spMatDescr.getAddress(), numElements, numElements, numElements, coordX, coordY, nnzVec,
                    CUSPARSERegistry.cusparseIndexType_t.CUSPARSE_INDEX_32I.ordinal(), CUSPARSERegistry.cusparseIndexBase_t.CUSPARSE_INDEX_BASE_ZERO.ordinal(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal());

            assertEquals(cusparseCreateCooOutputValue.asInt(), 0);

            // cusparseSpMV_buffersize
            Value cusparseSpMV_bufferSize = polyglot.eval("grcuda", "SPARSE::cusparseSpMV_bufferSize");
            cusparseSpMV_bufferSize.execute(
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    spMatDescr.getValue(),
                    dnVecXDescr.getValue(),
                    beta,
                    dnVecYDescr.getValue(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal(),
                    bufferSize.getAddress()
            );

            long bufferSizeValue = bufferSize.getValue();

            assertNotEquals(bufferSizeValue, -1);

            if (bufferSizeValue == 0) {
                // DeviceArrays cannot have size < 1
                // But cusparseSpMV_buffersize CAN return a value of zero
                // for the size of the temporary buffer
                bufferSizeValue = 1;
            }
            // cusparseSpMV
            Value cusparseSpMV = polyglot.eval("grcuda", "SPARSE::cusparseSpMV");
            Value buffer = cu.invokeMember("DeviceArray", "float", bufferSizeValue);

            cusparseSpMV.execute(
                    CUSPARSERegistry.cusparseOperation_t.CUSPARSE_OPERATION_NON_TRANSPOSE.ordinal(),
                    alpha,
                    spMatDescr.getValue(),
                    dnVecXDescr.getValue(),
                    beta,
                    dnVecYDescr.getValue(),
                    CUSPARSERegistry.cudaDataType.CUDA_R_32F.ordinal(),
                    CUSPARSERegistry.cusparseSpMVAlg_t.CUSPARSE_SPMV_ALG_DEFAULT.ordinal(),
                    buffer
            );

            for (int i = 0; i < numElements; i++) {
                assertEquals(outVec.getArrayElement(i).asFloat(), edge_value, 1e-5);
            }
        }
    }
}
