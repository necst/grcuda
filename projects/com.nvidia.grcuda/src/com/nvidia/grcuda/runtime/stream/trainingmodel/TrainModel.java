/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
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

package com.nvidia.grcuda.runtime.stream.trainingmodel;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.nvidia.grcuda.runtime.stream.Utilities;

import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class TrainModel {
    final private RandomForest rf;
    final private String kernelName;
    final private String pathFromGrcudaToStream = "grcuda/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/";
    final private String modelPath = "trainingmodel/trainedmodels/";
    final private String dataPath = "data/";
    final private String dataTrainClean = "clean_train_";

    // Create a model
    // TODO: add the possibility to set up the model
    public TrainModel(String kernelName, int numTree) {
        String path = Utilities.getPath() + this.pathFromGrcudaToStream;
        this.kernelName = kernelName;
        try {
            // Read the CSV file for cleaning the dataset
            Scanner scanner = new Scanner(new File(path + this.dataPath +this.kernelName + ".csv"));
            JsonObject jsonFile = new JsonObject();
            String[] line;
            String key, sep;
            String head = scanner.nextLine();
            while (scanner.hasNextLine()) {
                key = "";
                sep = "-";
                line = scanner.nextLine().split(",");
                for (String s : line) {
                    if (s.equals(line[0])) continue;
                    if (s.equals(line[line.length - 1])) sep = "";
                    key += (s + sep);
                }
                if (!jsonFile.keySet().contains(key)) {
                    JsonArray jsonArray = new JsonArray();
                    jsonArray.add(Double.parseDouble(line[0]));
                    jsonFile.add(key, jsonArray);
                } else {
                    ((JsonArray) jsonFile.get(key)).add(Double.parseDouble(line[0]));
                }
            }
            scanner.close();

            // Evaluate the median of the time in dataset
            JsonObject jsonFile_final = new JsonObject();
            for (String k : jsonFile.keySet()) {
                JsonArray ja = (JsonArray) jsonFile.get(k);
                ArrayList<Double> ar = new ArrayList<>();
                ja.forEach(x -> ar.add(x.getAsDouble()));
                jsonFile_final.addProperty(k, median(ar));
            }
            FileWriter output_train = new FileWriter(path + dataPath + dataTrainClean + kernelName + ".csv");
            output_train.write(head + "\n");
            for (String k : jsonFile_final.keySet()) {
                output_train.write(jsonFile_final.get(k) + "," + k.replace('-', ',') + "\n");
            }
            output_train.close();

            /*
            // Code to obtain the test dataset to check the model (Comment the code upper)
            FileWriter output_train = new FileWriter(path + "clean_train_" + kernelName + ".csv");
            output_train.write(head + "\n");
            FileWriter output_test = new FileWriter(path + "clean_test_" + kernelName + ".csv");
            output_test.write(head + "\n");
            int i = 0;
            int threshold = (int) (jsonFile_final.size() * 0.85);
            for (String k : jsonFile_final.keySet()) {
                if (i < threshold) {
                    output_train.write(jsonFile_final.get(k) + "," + k.replace('-', ',') + "\n");
                } else {
                    output_test.write(jsonFile_final.get(k) + "," + k.replace('-', ',') + "\n");
                }
                i++;
            }
            output_train.close();
            output_test.close();

            // Load CSV Test
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path + "clean_test_" + kernelName + ".csv"));
            Instances data = loader.getDataSet();

            // Save ARFF Test
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(path + "clean_test_" + kernelName + ".arff"));
            saver.writeBatch();

            // Load Test
            inputFile = new File(path + "clean_test_" + kernelName + ".arff"); //Test corpus file
            atf.setFile(inputFile);
            Instances instancesTest = atf.getDataSet(); // Read in the test file
            instancesTest.setClassIndex(0); //Setting the line number of the categorized attribute (No. 0 of the first action), instancesTest.numAttributes() can get the total number of attributes.
             */

            // Load CSV Train
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path + this.dataPath + this.dataTrainClean + this.kernelName + ".csv"));
            Instances data = loader.getDataSet();

            // Save ARFF Train
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(path + this.dataPath + this.dataTrainClean + this.kernelName + ".arff"));
            saver.writeBatch();

            // Load Train
            File inputFile = new File(path + this.dataPath + this.dataTrainClean + this.kernelName + ".arff"); //Training corpus file
            ArffLoader atf = new ArffLoader();
            atf.setFile(inputFile);
            Instances instancesTrain = atf.getDataSet(); // Read in training documents
            instancesTrain.setClassIndex(0);

            this.rf = new RandomForest();
            this.rf.buildClassifier(instancesTrain); //train

            // Preservation model
            SerializationHelper.write(path + this.modelPath + this.kernelName + ".model", this.rf);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // Load the model
    public TrainModel(String fileName) {
        this.kernelName = fileName;
        String path = Utilities.getPath() + this.pathFromGrcudaToStream + this.modelPath;
        try {
            this.rf = (RandomForest) SerializationHelper.read(path + fileName + ".model");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public double predictionTime(ArrayList<Double> params) {
        String name = "clean_train_" + this.kernelName;

        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("Time"));
        for (int j = 1; j < params.size() + 1; j++) attributes.add(new Attribute(String.valueOf(j)));

        Instance instance = new DenseInstance(params.size() + 1);
        int i = 1;
        for (Double el : params) {
            instance.setValue(i, el);
            i++;
        }

        Instances instances = new Instances(name, attributes, 1);
        instances.setClassIndex(0);
        instances.add(0, instance);

        try {
            return this.rf.classifyInstance(instances.instance(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public RandomForest getModel() {
        return this.rf;
    }

    private double median(ArrayList<Double> values) {
        // sort array
        Collections.sort(values);
        double median;
        // get count of scores
        int totalElements = values.size();
        // check if total number of scores is even
        if (totalElements % 2 == 0) {
            double sumOfMiddleElements = values.get(totalElements / 2) + values.get(totalElements / 2 - 1);
            // calculate average of middle elements
            median = (sumOfMiddleElements) / 2;
        } else {
            // get the middle element
            median = values.get(values.size() / 2);
        }
        return median;
    }
}

