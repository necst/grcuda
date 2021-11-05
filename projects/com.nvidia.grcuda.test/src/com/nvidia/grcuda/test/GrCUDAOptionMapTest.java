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
package com.nvidia.grcuda.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.cudalibraries.cublas.CUBLASRegistry;
import com.nvidia.grcuda.cudalibraries.cuml.CUMLRegistry;
import com.nvidia.grcuda.cudalibraries.tensorrt.TensorRTRegistry;
import com.nvidia.grcuda.runtime.executioncontext.ExecutionPolicyEnum;
import com.nvidia.grcuda.test.util.GrCUDATestUtil;
import com.nvidia.grcuda.test.util.mock.OptionValuesMock;
import com.oracle.truffle.api.interop.UnknownKeyException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;


public class GrCUDAOptionMapTest {

    private GrCUDAOptionMap optionMap;
    private final HashMap<OptionKey<String>, String> unparsedOptions = new HashMap<>();
    private final HashMap<OptionKey<Boolean>, Boolean> options = new HashMap<>();

    public void initializeDefault(){
        options.put(GrCUDAOptions.CuBLASEnabled, true);
        options.put(GrCUDAOptions.CuMLEnabled, true);
        options.put(GrCUDAOptions.ForceStreamAttach, GrCUDAOptionMap.DEFAULT_FORCE_STREAM_ATTACH);
        options.put(GrCUDAOptions.InputPrefetch, false);
        options.put(GrCUDAOptions.EnableMultiGPU, false);
        options.put(GrCUDAOptions.TensorRTEnabled, false);
        unparsedOptions.put(GrCUDAOptions.CuBLASLibrary, CUBLASRegistry.DEFAULT_LIBRARY);
        unparsedOptions.put(GrCUDAOptions.CuMLLibrary, CUMLRegistry.DEFAULT_LIBRARY);
        unparsedOptions.put(GrCUDAOptions.ExecutionPolicy, GrCUDAOptionMap.DEFAULT_EXECUTION_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.DependencyPolicy, GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.RetrieveNewStreamPolicy, GrCUDAOptionMap.DEFAULT_RETRIEVE_STREAM_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.RetrieveParentStreamPolicy, GrCUDAOptionMap.DEFAULT_PARENT_STREAM_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.TensorRTLibrary, TensorRTRegistry.DEFAULT_LIBRARY);

        OptionValues optionValues = new OptionValuesMock();
        options.forEach(optionValues::set);
        unparsedOptions.forEach(optionValues::set);

        optionMap = GrCUDAOptionMap.getInstance(optionValues);
    }

    public void initializeNull(){
        unparsedOptions.put(GrCUDAOptions.ExecutionPolicy, GrCUDAOptionMap.DEFAULT_EXECUTION_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.DependencyPolicy, GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.RetrieveNewStreamPolicy, GrCUDAOptionMap.DEFAULT_RETRIEVE_STREAM_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.RetrieveParentStreamPolicy, GrCUDAOptionMap.DEFAULT_PARENT_STREAM_POLICY.getName());
        unparsedOptions.put(GrCUDAOptions.TensorRTLibrary, null);

        OptionValues optionValues = new OptionValuesMock();
        unparsedOptions.forEach(optionValues::set);

        optionMap = GrCUDAOptionMap.getInstance(optionValues);
    }

    @Test
    public void testGetOption(){
        initializeDefault();
        assertEquals(optionMap.isCuBLASEnabled(), true);
        assertEquals(optionMap.isForceStreamAttach(), false);
        assertEquals(optionMap.getCuBLASLibrary(), CUBLASRegistry.DEFAULT_LIBRARY);
        assertEquals(optionMap.getDependencyPolicy(), GrCUDAOptionMap.DEFAULT_DEPENDENCY_POLICY);
    }

    @Test(expected = UnknownKeyException.class)
    public void testReadUnknownKey() throws UnsupportedMessageException, UnknownKeyException {
        initializeNull();
        optionMap.readHashValue("NotPresent");
    }

    @Test(expected = UnsupportedMessageException.class)
    public void testReadUnsupportedMessage() throws UnsupportedMessageException, UnknownKeyException {
        initializeDefault();
        optionMap.readHashValue(null);
    }

    @Test
    public void testGetOptionsFunction() {
        try (Context ctx = GrCUDATestUtil.buildTestContext().option("grcuda.ExecutionPolicy", ExecutionPolicyEnum.ASYNC.getName()).build()) {
            // Obtain the options map;
            Value options = ctx.eval("grcuda", "getoptions").execute();
            // Check the we have a map;
            assertTrue(options.hasHashEntries());
            System.out.println(options.getHashSize());
            // Obtain some options;
            assertEquals(options.getHashValue("grcuda.ExecutionPolicy").asString(), ExecutionPolicyEnum.ASYNC.getName());
            assertEquals(options.getHashValue("grcuda.EnableMultiGPU").asBoolean(), GrCUDAOptionMap.DEFAULT_ENABLE_MULTIGPU);
        }
    }
}
