/*
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
package com.nvidia.grcuda;

import com.nvidia.grcuda.runtime.computation.dependency.DependencyPolicyEnum;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.RetrieveParentStreamPolicyEnum;
import com.oracle.truffle.api.TruffleLogger;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownKeyException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

@ExportLibrary(InteropLibrary.class)
public class GrCUDAOptionMap implements TruffleObject {

    private HashMap<OptionKey<?>, Object> optionKeyValueMap;
    private static final TruffleLogger LOGGER = TruffleLogger.getLogger(GrCUDALanguage.ID, "com.nvidia.grcuda.GrCUDAContext");

    public GrCUDAOptionMap(OptionValues options) {
        optionKeyValueMap = new HashMap<>();
        List<OptionKey<?>> allOptions = new ArrayList<>();
        options.getDescriptors().forEach(o -> allOptions.add(o.getKey()));

        allOptions.forEach(i -> optionKeyValueMap.put(i, options.get(i)));
        // stream retrieval policy;
        optionKeyValueMap.replace(GrCUDAOptions.RetrieveNewStreamPolicy, parseRetrieveStreamPolicy(options.get(GrCUDAOptions.RetrieveNewStreamPolicy)));
        // how streams are obtained from parent computations;
        optionKeyValueMap.replace(GrCUDAOptions.RetrieveParentStreamPolicy, parseParentStreamPolicy(options.get(GrCUDAOptions.RetrieveParentStreamPolicy)));
        // dependency computation policy;
        DependencyPolicyEnum dependencyPolicy = (DependencyPolicyEnum) optionKeyValueMap.replace(GrCUDAOptions.DependencyPolicy, parseDependencyPolicy(options.get(GrCUDAOptions.DependencyPolicy)));
        LOGGER.fine("using " + dependencyPolicy.getName() + " dependency policy");
        // execution policy;
        optionKeyValueMap.replace(GrCUDAOptions.ExecutionPolicy, parseExecutionPolicy(options.get(GrCUDAOptions.ExecutionPolicy)));
    }

    public HashMap<OptionKey<?>, Object> getOptions(){
        return optionKeyValueMap;
    }

    @ExportMessage
    public final boolean hasHashEntries(){
        return true;
    }

    @ExportMessage
    public final Object readHashValue(Object key) throws UnknownKeyException, UnsupportedMessageException {
        Object value;
        if (key instanceof OptionKey){
            value = optionKeyValueMap.get(key);
        }
        else {
            throw UnsupportedMessageException.create();
        }
        if (value == null) throw UnknownKeyException.create(key, new Throwable());
        return value;
    }

    @ExportMessage
    final long getHashSize() throws UnsupportedMessageException { return optionKeyValueMap.size(); }

    @ExportMessage
    final boolean isHashEntryReadable(Object key) {
        return (key instanceof OptionKey) && optionKeyValueMap.get(key) != null;
    }

    @ExportMessage
    final Object getHashEntriesIterator() throws UnsupportedMessageException { return optionKeyValueMap.entrySet().iterator(); }

    private static ExecutionPolicyEnum parseExecutionPolicy(String policyString) {
        if (policyString.equals(ExecutionPolicyEnum.SYNC.getName())) return ExecutionPolicyEnum.SYNC;
        else if (policyString.equals(ExecutionPolicyEnum.ASYNC.getName())) return ExecutionPolicyEnum.ASYNC;
        else {
            LOGGER.warning("unknown execution policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_EXECUTION_POLICY);
            return GrCUDAContext.DEFAULT_EXECUTION_POLICY;
        }
    }

    private static DependencyPolicyEnum parseDependencyPolicy(String policyString) {
        if (policyString.equals(DependencyPolicyEnum.WITH_CONST.getName())) return DependencyPolicyEnum.WITH_CONST;
        else if (policyString.equals(DependencyPolicyEnum.NO_CONST.getName())) return DependencyPolicyEnum.NO_CONST;
        else {
            LOGGER.warning("Warning: unknown dependency policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_DEPENDENCY_POLICY);
            return GrCUDAContext.DEFAULT_DEPENDENCY_POLICY;
        }
    }

    private static RetrieveNewStreamPolicyEnum parseRetrieveStreamPolicy(String policyString) {
        if (policyString.equals(RetrieveNewStreamPolicyEnum.FIFO.getName())) return RetrieveNewStreamPolicyEnum.FIFO;
        else if (policyString.equals(RetrieveNewStreamPolicyEnum.ALWAYS_NEW.getName())) return RetrieveNewStreamPolicyEnum.ALWAYS_NEW;
        else {
            LOGGER.warning("Warning: unknown new stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY);
            return GrCUDAContext.DEFAULT_RETRIEVE_STREAM_POLICY;
        }
    }

    private static RetrieveParentStreamPolicyEnum parseParentStreamPolicy(String policyString) {
        if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.DISJOINT.getName())) return RetrieveParentStreamPolicyEnum.DISJOINT;
        else if (Objects.equals(policyString, RetrieveParentStreamPolicyEnum.SAME_AS_PARENT.getName())) return RetrieveParentStreamPolicyEnum.SAME_AS_PARENT;
        else {
            LOGGER.warning("Warning: unknown parent stream retrieval policy=" + policyString + "; using default=" + GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY);
            return GrCUDAContext.DEFAULT_PARENT_STREAM_POLICY;
        }
    }

    public Boolean isCuBLASEnabled(){
        return (Boolean) optionKeyValueMap.get(GrCUDAOptions.CuBLASEnabled);
    }

    public String getCuBLASLibrary(){
        return (String) optionKeyValueMap.get(GrCUDAOptions.CuBLASLibrary);
    }

    public Boolean isCuMLEnabled(){
        return (Boolean) optionKeyValueMap.get(GrCUDAOptions.CuMLEnabled);
    }

    public String getCuMLLibrary(){
        return (String) optionKeyValueMap.get(GrCUDAOptions.CuMLLibrary);
    }

    public ExecutionPolicyEnum getExecutionPolicy(){
        return (ExecutionPolicyEnum) optionKeyValueMap.get(GrCUDAOptions.ExecutionPolicy);
    }

    public DependencyPolicyEnum getDependencyPolicy(){
        return (DependencyPolicyEnum) optionKeyValueMap.get(GrCUDAOptions.DependencyPolicy);
    }

    public RetrieveNewStreamPolicyEnum getRetrieveNewStreamPolicy(){
        return (RetrieveNewStreamPolicyEnum) optionKeyValueMap.get(GrCUDAOptions.RetrieveNewStreamPolicy);
    }

    public RetrieveParentStreamPolicyEnum getRetrieveParentStreamPolicy(){
        return (RetrieveParentStreamPolicyEnum) optionKeyValueMap.get(GrCUDAOptions.RetrieveParentStreamPolicy);
    }

    public Boolean isForceStreamAttach(){
        return (Boolean) optionKeyValueMap.get(GrCUDAOptions.ForceStreamAttach);
    }

    public Boolean isInputPrefetch(){
        return (Boolean) optionKeyValueMap.get(GrCUDAOptions.InputPrefetch);
    }

    public Boolean isEnableMultiGPU(){
        return (Boolean) optionKeyValueMap.get(GrCUDAOptions.EnableMultiGPU);
    }

    public Boolean isTensorRTEnabled(){
        return (Boolean) optionKeyValueMap.get(GrCUDAOptions.TensorRTEnabled);
    }

    public String getTensorRTLibrary(){
        return (String) optionKeyValueMap.get(GrCUDAOptions.TensorRTLibrary);
    }

}
