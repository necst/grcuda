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
package com.nvidia.grcuda.runtime.computation;

import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.runtime.array.AbstractArray;
import com.nvidia.grcuda.runtime.ConfiguredKernel;
import com.nvidia.grcuda.runtime.Kernel;
import com.nvidia.grcuda.runtime.KernelArguments;
import com.nvidia.grcuda.runtime.KernelConfig;
import com.nvidia.grcuda.runtime.executioncontext.AsyncGrCUDAExecutionContext;
import com.nvidia.grcuda.runtime.stream.CUDAStream;
import com.nvidia.grcuda.runtime.stream.DefaultStream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.File;

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Class used to store data about the single execution of a {@link KernelExecution}.
 */
public class KernelExecutionObserver {
    private final KernelConfig config;
    private final Kernel kernel;
    private final KernelArguments args;

    public KernelExecutionObserver(KernelConfig config, Kernel kernel, KernelArguments args) {
        this.config = config;
        this.kernel = kernel;
        this.args = args;
    }

    public void update(float time) {
        // Save information on .csv;
        List<String[]> kernelInformations = new ArrayList<>();
        List<Long> deviceArraySize;
        List<Integer> integerValue;
        List<Float> floatValue;
        List<Double> doubleValue;
        List<String> signature;

        //Get lengths of arrays
        deviceArraySize = this.args.getKernelDeviceArraySize();
        String deviceArraySizeForFile = "[";
        for (long i : deviceArraySize) {
            deviceArraySizeForFile += Long.toString(i);
            deviceArraySizeForFile += ";";
        }
        deviceArraySizeForFile += "]";

        //Get values of 'int' variables
        integerValue = this.args.getKernelIntegerValue();
        String integerValueForFile = "[";
        for (int i : integerValue) {
            integerValueForFile += Integer.toString(i);
            integerValueForFile += ";";
        }
        integerValueForFile += "]";

        //Get values of 'float' variables
        floatValue = this.args.getKernelFloatValue();
        String floatValueForFile = "[";
        for (float i : floatValue) {
            floatValueForFile += Float.toString(i);
            floatValueForFile += ";";
        }
        floatValueForFile += "]";

        //Get values of 'double' variables
        doubleValue = this.args.getKernelDoubleValue();
        String doubleValueForFile = "[";
        for (double i : doubleValue) {
            doubleValueForFile += Double.toString(i);
            doubleValueForFile += ";";
        }
        doubleValueForFile += "]";

        //Get signature
        signature = this.args.getKernelSignature();
        signature = signature.stream().map(x -> (x.split("\\.")[x.split("\\.").length - 1])).collect(Collectors.toList());
        String signatureForHash = " (";
        for (int i = 0; i < signature.size(); i++) {
            signatureForHash += signature.get(i);
            if (i < signature.size() - 1) signatureForHash += ", ";
        }
        signatureForHash += ")";

        // Get Kernel_ID
        String hashtext;
        String id = "";
        try {
            id = this.kernel.getKernelName() + signatureForHash;
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] messageDigest = md.digest(id.getBytes());
            BigInteger no = new BigInteger(1, messageDigest);
            hashtext = no.toString(16);
            while (hashtext.length() < 32) {
                hashtext = "0" + hashtext;
            }
        } catch (NoSuchAlgorithmException e) {
            hashtext = this.kernel.getKernelName();
        }

        // Find the location where to save the data
        File f = new File("");
        String end = "grcuda/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/data/";
        String path = "";
        for (String dir : f.getAbsolutePath().split("/")) {
            if (dir.equals("grcuda")) {
                path += end;
                break;
            }
            path += (dir + "/");
        }

        Path p = Paths.get(path);
        if (!Files.isDirectory(p)) {
            new File(path).mkdirs();
        }
        path += (hashtext + ".csv");

        //List with name, grid dimensions, block dimensions, time, lengths of arrays, values of variables (int, float, double)
        kernelInformations.add(new String[]
                {id, Integer.toString(this.config.getGridSizeX()),
                        Integer.toString(this.config.getGridSizeY()), Integer.toString(this.config.getGridSizeZ()),
                        Integer.toString(this.config.getBlockSizeX()), Integer.toString(this.config.getBlockSizeY()),
                        Integer.toString(this.config.getBlockSizeZ()), Float.toString(time),
                        deviceArraySizeForFile, integerValueForFile, floatValueForFile, doubleValueForFile});
        this.givenDataArrayWhenConvertToCSVThenOutputCreated(kernelInformations, path);
    }

    // Useful methods for writing .csv files without external libraries
    private String convertToCSV(String[] data) {
        return Stream.of(data)
                .map(this::escapeSpecialCharacters)
                .collect(Collectors.joining(","));
    }

    private String escapeSpecialCharacters(String data) {
        String escapedData = data.replaceAll("\\R", " ");
        if (data.contains(",") || data.contains("\"") || data.contains("'")) {
            data = data.replace("\"", "\"\"");
            escapedData = "\"" + data + "\"";
        }
        return escapedData;
    }

    private void givenDataArrayWhenConvertToCSVThenOutputCreated(List<String[]> dataLines, String fileName) {
        try {
            FileWriter csvOutputFile = new FileWriter(fileName, true);
            PrintWriter pw = new PrintWriter(csvOutputFile);
            dataLines.stream()
                    .map(this::convertToCSV)
                    .forEach(pw::write);
            pw.write("\n");
            pw.close();
        } catch (Exception e) {
            e.getStackTrace();
        }
    }
}