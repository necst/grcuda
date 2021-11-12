package com.nvidia.grcuda.cudalibraries.cusparse.cusparseproxy;

import com.nvidia.grcuda.functions.ExternalFunctionFactory;

public class CUSPARSEProxySpMV extends CUSPARSEProxy {

    private final int nArgsRaw = 10; // args for library function
    private final int nArgsSimplified = 17; // args to be proxied


    public CUSPARSEProxySpMV(ExternalFunctionFactory externalFunctionFactory) {
        super(externalFunctionFactory);
    }

    @Override
    public Object[] formatArguments(Object[] rawArgs) {
        if(rawArgs.length == nArgsRaw){
            return rawArgs;
        } else {
            // call functions to create arguments and descriptors
            args = new Object[nArgsRaw];
            return args;
        }
    }
}
