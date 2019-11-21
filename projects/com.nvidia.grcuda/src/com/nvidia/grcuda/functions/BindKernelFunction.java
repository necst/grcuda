/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.gpu.CUDARuntime;
import com.oracle.truffle.api.CompilerDirectives.TruffleBoundary;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;

public final class BindKernelFunction extends Function {

    private final CUDARuntime cudaRuntime;

    public BindKernelFunction(CUDARuntime cudaRuntime) {
        super("bindkernel", "");
        this.cudaRuntime = cudaRuntime;
    }

    private String parseCubinFile(Object cubinFile) throws UnsupportedTypeException {
        return expectString(cubinFile, "argument 0 of bindkernel must be string (cubin file)");
    }

    private String parseKernelName(Object kernelName) throws UnsupportedTypeException {
        return expectString(kernelName, "argument 1 of bindkernel must be string (kernel name)");
    }

    private String parseKernelSignature(Object kernelSignature) throws UnsupportedTypeException {
        return expectString(kernelSignature, "argument 2 of bindkernel must be string (signature of kernel)");
    }

    private String parseAddSizeParam(Object addSize) throws UnsupportedTypeException {
        return expectString(addSize, "argument 3 of bindkernel must be string with value True or False");
    }

    private Object parseStandardParameters(Object[] arguments) throws UnsupportedTypeException {
        String cubinFile = parseCubinFile(arguments[0]);
        String kernelName = parseKernelName(arguments[1]);
        String kernelSignature = parseKernelSignature(arguments[2]);
        return cudaRuntime.loadKernel(cubinFile, kernelName, kernelSignature);
    }

    @Override
    @TruffleBoundary
    public Object call(Object[] arguments) throws UnsupportedTypeException, ArityException {
        if (arguments.length == 3) {
            // Default case;
            checkArgumentLength(arguments, 3);
            return parseStandardParameters(arguments);
        } else if (arguments.length == 4) {
            // If specified, add the array length parameters to the kernel signature;
            checkArgumentLength(arguments, 4);
            String addSizesStr = parseAddSizeParam(arguments[3]);
            boolean addSizes = Boolean.parseBoolean(addSizesStr);
            if (addSizes) {
                String cubinFile = parseKernelName(arguments[0]);
                String kernelName = parseKernelName(arguments[1]);
                String kernelSignature = parseKernelSignature(arguments[2]);
                String[] parameters = kernelSignature.trim().replaceAll("\\s+","").split(",");
                StringBuilder newSignatureBuilder = new StringBuilder();
                newSignatureBuilder.append(kernelSignature);

                int numOfPointers = 0;
                boolean sizes_array_added = false;
                for (String p : parameters) {
                    // Add a "pointer" parameter if any pointer argument is found, used to pass the size;
                    if (p.equals("pointer")) {
                        if (!sizes_array_added) {
                            newSignatureBuilder.append(",pointer");
                            sizes_array_added = true;
                        }
                        numOfPointers++;
                    }
                }
                return cudaRuntime.loadKernelWithSizes(cubinFile, kernelName, newSignatureBuilder.toString(), numOfPointers);
            } else {
                // If the 4th parameter is false, fallback to the standard kernel loading;
                return parseStandardParameters(arguments);
            }
        } else {
            // The bindkernel function actually supports 3 or 4 arguments, but ArityException doesn't
            // allow to specify >1 expected value;
            throw ArityException.create(4, arguments.length);
        }
    }
}
