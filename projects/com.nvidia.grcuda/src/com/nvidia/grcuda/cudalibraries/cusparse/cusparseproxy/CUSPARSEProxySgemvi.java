package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArray;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;

import com.nvidia.grcuda.cudalibraries.cusparse.CUSPARSERegistry;
import com.nvidia.grcuda.functions.ExternalFunctionFactory;
import com.nvidia.grcuda.runtime.UnsafeHelper;

import static com.nvidia.grcuda.functions.Function.INTEROP;
import static com.nvidia.grcuda.functions.Function.expectLong;
import static com.nvidia.grcuda.functions.Function.expectInt;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public class CUSPARSEProxySgemvi extends CUSPARSEProxy {

    private final int nArgsRaw = 13; // args for library function

    public CUSPARSEProxySgemvi(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs, long handle) throws UnsupportedTypeException {
        this.initializeNfi();
        if(rawArgs.length == nArgsRaw){
            return rawArgs;
        } else {

            System.out.println("entered format arguments");
            args = new Object[nArgsRaw];

            UnsafeHelper.Integer64Object bufferSize = UnsafeHelper.createInteger64Object();

            bufferSize.setValue(0); // does not work without initialization

            CUSPARSERegistry.cusparseOperation_t transA = CUSPARSERegistry.cusparseOperation_t.values()[expectInt(rawArgs[0])];
            int rows = expectInt(rawArgs[1]);
            int cols = expectInt(rawArgs[2]);
            DeviceArray alpha = (DeviceArray) rawArgs[3];
            MultiDimDeviceArray matA = (MultiDimDeviceArray) rawArgs[4];
            int lda =expectInt(rawArgs[5]);
            int nnz = expectInt(rawArgs[6]);
            DeviceArray x = (DeviceArray) rawArgs[7];
            DeviceArray xInd = (DeviceArray) rawArgs[8];
            DeviceArray beta = (DeviceArray) rawArgs[9];
            DeviceArray outVec= (DeviceArray) rawArgs[10];
            CUSPARSERegistry.cusparseIndexBase_t idxBase = CUSPARSERegistry.cusparseIndexBase_t.values()[expectInt(rawArgs[11])];

            System.out.println("arguments fetched");
            // create buffer
            try {
                Object resultBufferSize = INTEROP.execute(cusparseSgemvi_bufferSizeFunction, handle, transA.ordinal(), rows, cols, nnz, bufferSize.getAddress());
            } catch (ArityException | UnsupportedTypeException | UnsupportedMessageException e) {
                e.printStackTrace();
            }

            System.out.println("created buffer");

            long numElements;

            if (bufferSize.getValue() == 0) {
                numElements = 1;
            } else {
                numElements = (long) bufferSize.getValue();
            }

            System.out.println(bufferSize.getValue());

            DeviceArray buffer = new DeviceArray(alpha.getGrCUDAExecutionContext(), numElements, alpha.getElementType());

            args[0] = transA.ordinal();
            args[1] = rows;
            args[2] = cols;
            args[3] = alpha;
            args[4] = matA;
            args[5] = lda;
            args[6] = nnz;
            args[7] = x;
            args[8] = xInd;
            args[9] = beta;
            args[10] = outVec;
            args[11] = idxBase.ordinal();
            args[12] = buffer;

            return args;
        }
    }
}
