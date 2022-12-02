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

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class RetrainModel {
    // Check for system.platform, replace python for Windows
    String CMD = "cd $GRCUDA_HOME/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/trainingmodel/ && python3 main.py";

    public RetrainModel() {
    }

    public void retrainModel(ArrayList<String> kernels) {
        if(kernels!=null){
            try {
                // TODO: fix requirements
                // CMD for requirements: pip3 install -r requirements.txt
                String kernelsString = "";
                for (String k : kernels) kernelsString += (" " + k);
                CMD += kernelsString;
                ProcessBuilder builtCmd = new ProcessBuilder("bash", "-c", CMD);
                builtCmd.redirectErrorStream(true);
                Process p = builtCmd.start();

                // Print output code
                BufferedReader output_reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String output = "";
                while ((output = output_reader.readLine()) != null) {
                    System.out.println(output);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("Training models impossible: no data...");
        }
    }
}
