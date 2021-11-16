package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import org.graalvm.polyglot.Value;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectInt;
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

            if ((v1.getArraySize() == v2.getArraySize()) && (v2.getArraySize() == values.getArraySize())) { // coo

// create context variable//bitwise non si può
                UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object cooMatDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
                CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[0])];
                DeviceArray alpha = (DeviceArray) rawArgs[1];
                long rows = expectLong(rawArgs[2]);
                long cols = expectLong(rawArgs[3]);
                long nnz = expectLong(rawArgs[4]);
                CUSPARSERegistry.cusparseIndexType_t cooIdxType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(rawArgs[8])];
                CUSPARSERegistry.cusparseIndexBase_t cooIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[9])];
                CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[10])];
                long size = cols;
                DeviceArray valuesX = (DeviceArray) rawArgs[11];
                CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[12])];
                DeviceArray beta = (DeviceArray) rawArgs[13];
                DeviceArray valuesY = (DeviceArray) rawArgs[14];
                CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cusparseSpMVAlg_t.values()[expectInt(rawArgs[15])];

                // create coo matrix descriptor
                try {
                    Object resultCoo = INTEROP.execute(cusparseCreateCooFunction, cooMatDescr.getAddress(), rows, cols, nnz, v1, v2, values, cooIdxType.ordinal(), cooIdxBase.ordinal(),
                                    valueType.ordinal());
// System.out.println("created coo: result " + resultCoo);
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                } // TODO: re-throw an exception if sth goes wrong

                // create dense vectors X and Y descriptors
                try {
                    Object resultX = INTEROP.execute(cusparseCreateDnVecFunction, dnVecXDescr.getAddress(), size, valuesX, valueTypeVec.ordinal());
                    Object resultY = INTEROP.execute(cusparseCreateDnVecFunction, dnVecYDescr.getAddress(), size, valuesY, valueTypeVec.ordinal());
// System.out.println("created dnvec: result " + resultX + resultY);
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                }

                // create buffer
                try {
                    Object resultBufferSize = INTEROP.execute(cusparseSpMV_bufferSizeFunction, handle, opA.ordinal(), alpha, cooMatDescr.getValue(), dnVecXDescr.getValue(), beta,
                                    dnVecYDescr.getValue(), valueType.ordinal(), alg.ordinal(), bufferSize.getAddress());
// System.out.println("created buffer: result " + resultBufferSize);
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

                // format new arguments for SpMV with COO format
                args[0] = opA.ordinal();
                args[1] = alpha;
                args[2] = cooMatDescr.getValue();
                args[3] = dnVecXDescr.getValue();
                args[4] = beta;
                args[5] = dnVecYDescr.getValue();
                args[6] = valueType.ordinal();
                args[7] = alg.ordinal();
                args[8] = buffer;
            } else { // csr
                UnsafeHelper.Integer64Object dnVecXDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object dnVecYDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object csrMatDescr = UnsafeHelper.createInteger64Object();
                UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();
                CUSPARSERegistry.cusparseOperation_t opA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[0])];
                long alpha = expectLong(rawArgs[1]);
                long rows = expectLong(rawArgs[2]);
                long cols = expectLong(rawArgs[3]);
                long nnz = expectLong(rawArgs[4]);
                long csrRowOffsets = expectLong(rawArgs[5]);
                long csrColIdx = expectLong(rawArgs[6]);
                long csrValues = expectLong(rawArgs[7]);
                CUSPARSERegistry.cusparseIndexType_t csrOffsetType = CUSPARSERegistry.cusparseIndexType_t.values()[expectInt(rawArgs[8])];
                CUSPARSERegistry.cusparseIndexType_t csrColIdxType = csrOffsetType; // all the same
                                                                                    // (for now)
                CUSPARSERegistry.cusparseIndexBase_t csrIdxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[9])];
                CUSPARSERegistry.cudaDataType valueType = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[10])];
                long size = cols;
                DeviceArray valuesX = (DeviceArray) rawArgs[11];
                CUSPARSERegistry.cudaDataType valueTypeVec = CUSPARSERegistry.cudaDataType.values()[expectInt(rawArgs[12])];
                long beta = expectLong(rawArgs[13]);
                DeviceArray valuesY = (DeviceArray) rawArgs[14];
                CUSPARSERegistry.cusparseSpMVAlg_t alg = CUSPARSERegistry.cusparseSpMVAlg_t.values()[expectInt(rawArgs[15])];

                // create csr matrix descriptor
                try {
                    Object resultCsr = INTEROP.execute(cusparseCreateCsrFunction, csrMatDescr.getAddress(), rows, cols, nnz, csrRowOffsets, csrColIdx, csrValues, csrOffsetType.ordinal(),
                                    csrColIdxType.ordinal(), csrIdxBase.ordinal(), valueType.ordinal());
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                } // TODO: re-throw an exception if sth goes wrong

                // create dense vectors X and Y descriptors
                try {
                    Object resultX = INTEROP.execute(cusparseCreateDnVecFunction, dnVecXDescr.getAddress(), size, valuesX, valueTypeVec.ordinal());
                    Object resultY = INTEROP.execute(cusparseCreateDnVecFunction, dnVecYDescr.getAddress(), size, valuesY, valueTypeVec.ordinal());
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                }

                // create buffer
                try {
                    Object resultBufferSize = INTEROP.execute(cusparseSpMV_bufferSizeFunction, handle, opA.ordinal(), alpha, csrMatDescr.getValue(), dnVecXDescr.getValue(), beta,
                                    dnVecYDescr.getValue(), valueType.ordinal(), alg.ordinal(), bufferSize.getAddress());
                } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                    e.printStackTrace();
                }

                // format new arguments for SpMV with CSR format
                args[0] = opA;
                args[1] = alpha;
                args[2] = csrMatDescr;
                args[3] = dnVecXDescr;
                args[4] = beta;
                args[5] = dnVecYDescr;
                args[6] = valueType;
                args[7] = alg;
                args[8] = bufferSize;
            }
            return args;
        }
    }
}
