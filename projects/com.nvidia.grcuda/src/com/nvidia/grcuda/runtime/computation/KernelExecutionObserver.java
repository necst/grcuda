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


import com.nvidia.grcuda.runtime.Kernel;
import com.nvidia.grcuda.runtime.KernelArguments;
import com.nvidia.grcuda.runtime.KernelConfig;
import com.nvidia.grcuda.runtime.stream.Utilities;
//import com.nvidia.grcuda.runtime.stream.trainingmodel.TrainModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.Map;
import java.util.HashMap;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.File;

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.pmml4s.model.Model;

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

    // Save information on .csv;
    public void update(float time) {
        List<Object> values;
        int count = 1;

        //Get values of numeric variables
        values = this.args.getKernelValues();
        String[] line = new String[values.size() + 1];
        line[0] = Float.toString(time);
        for (Object i : values) {
            line[count] = i.toString();
            count++;
        }

        // Find the location where to save the data
        String path = absolutePath("grcuda/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/data/");

        // Get Kernel_ID
        String id = this.kernel.getKernelName() + signatureForHash();

        // Hash
        String hashtext = hash(id);

        // Save kernel id and hashtext for developing, TODO: delete
        if (!Files.exists(Paths.get(path + "names.csv"))) {
            this.printLineToCsv(new String[]{"id", "hashtext"}, path + "names.csv");
        }
        this.printLineToCsv(new String[]{id, hashtext}, path + "names.csv");

        // Complete path
        Path p = Paths.get(path);
        if (!Files.isDirectory(p)) {
            new File(path).mkdirs();
        }
        path += (hashtext + ".csv");

        //If file doesn't exist add head
        if (!Files.exists(Paths.get(path))) {
            String[] head = new String[values.size() + 1];
            head[0] = "Time";
            for (int i = 0; i < values.size(); i++) {
                head[i + 1] = Integer.toString(i);
            }
            this.printLineToCsv(head, path);
        }
        //Add line with time, values of numeric variables (in the order of signature)
        this.printLineToCsv(line, path);

    }

    //TEST accuracy of models
    public double testModel() {
        List<Object> values;
        Map<String, Double> map = new HashMap<String, Double>();
        int count = 0;

        // Find the location where the model is
        String path = absolutePath("grcuda/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/trainingmodel/trainedmodels/");
        // Get Kernel_ID
        String id = this.kernel.getKernelName() + signatureForHash();
        // Hash
        String hashtext = hash(id);
        // Complete path
        path += (hashtext + ".pmml");
        // Get model
        if (Files.exists(Paths.get(path))) {
            Model model = Model.fromFile(path);
            //Get values of numeric variables and create map
            values = this.args.getKernelValues();
            for (Object i : values) {
                map.put(Integer.toString(count), Double.parseDouble(i.toString()));
                count++;
            }
            // Test model
            double predicted = getRegressionValue(map, model);
            return predicted;
        } else {
            System.out.println(path);
            System.out.println("Model doesn't exist");
            return -1;
        }
    }

    /*public void printPrediction(float time) {
        ArrayList<Double> values = new ArrayList<>();
        TrainModel trainModel;

        // Find the location where the model is
        String path = Utilities.getPath() + "grcuda/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/trainingmodel/trainedmodels/";
        
        // Get Kernel_ID
        String id = this.kernel.getKernelName() + signatureForHash();
        
        // Hash
        String hashSiganture = hash(id);

        // Complete path
        path += (hashSiganture + ".model");
        
        // Get model
        if (Files.exists(Paths.get(path))) {
            trainModel = new TrainModel(hashSiganture);

            //Get values of numeric variables and create map
            for (Object i : this.args.getKernelValues()) {
                values.add(Double.parseDouble(i.toString()));
            }

            // Test model
            double predicted = trainModel.predictionTime(values);
            System.out.println("Predicted: " + predicted + "  \t" + "Actual: " + time);
        } else {
            System.out.println(path);
            System.out.println("Model doesn't exist");
        }
    }*/

    public Double getRegressionValue(Map<String, Double> values, Model model) {
        Object[] valuesMap = Arrays.stream(model.inputNames())
                .map(values::get)
                .toArray();

        Object[] result = model.predict(valuesMap);
        return (Double) result[0];
    }

    //Return signature, grid and block sizes for hash
    String signatureForHash() {
        List<String> signature;
        List<String> sizes = new ArrayList<>();
        //Get signature, grid size and block size
        signature = this.args.getKernelSignature();
        signature = signature.stream().map(x -> (x.split("\\.")[x.split("\\.").length - 1])).collect(Collectors.toList());
        String signatureForHash = " (";
        for (int i = 0; i < signature.size(); i++) {
            signatureForHash += signature.get(i);
            if (i < signature.size() - 1) signatureForHash += ", ";
        }
        signatureForHash += ")";

        sizes.add(Integer.toString(this.config.getGridSizeX()));
        sizes.add(Integer.toString(this.config.getGridSizeY()));
        sizes.add(Integer.toString(this.config.getGridSizeZ()));
        sizes.add(Integer.toString(this.config.getBlockSizeX()));
        sizes.add(Integer.toString(this.config.getBlockSizeY()));
        sizes.add(Integer.toString(this.config.getBlockSizeZ()));
        sizes = sizes.stream().map(x -> (x.split("\\.")[x.split("\\.").length - 1])).collect(Collectors.toList());
        signatureForHash += " (";
        for (int i = 0; i < sizes.size(); i++) {
            signatureForHash += sizes.get(i);
            if (i < sizes.size() - 1) signatureForHash += ", ";
        }
        signatureForHash += ")";
        return signatureForHash;
    }

    String absolutePath(String end) {
        File f = new File("");
        String path = "";
        for (String dir : f.getAbsolutePath().split("/")) {
            if (dir.equals("grcuda")) {
                path += end;
                break;
            }
            path += (dir + "/");
        }
        return path;
    }

    String hash(String id) {
        String hashtext = "";
        try {
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
        return hashtext;
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

    private void printLineToCsv(String[] line, String fileName) {
        try {
            FileWriter csvOutputFile = new FileWriter(fileName, true);
            PrintWriter pw = new PrintWriter(csvOutputFile);
            pw.write(convertToCSV(line));
            pw.write("\n");
            pw.close();
        } catch (Exception e) {
            e.getStackTrace();
        }
    }
}
