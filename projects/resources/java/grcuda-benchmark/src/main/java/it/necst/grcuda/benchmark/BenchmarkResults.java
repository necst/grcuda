/*
 * Copyright (c) 2022 NECSTLab, Politecnico di Milano. All rights reserved.
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

package it.necst.grcuda.benchmark;


import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This class stores all the results coming from a benchmark.
 * It is mainly composed of a linked list containing various BenchmarkRecord, those records are reporting information on the single phases in the benchmark (like timings etc).
 */
public class BenchmarkResults {
    List<BenchmarkRecord> phases = new LinkedList<>();
    final String currentBenchmark;
    final String currentSetupId;
    public float gpu_result; // here we store the result of the gpu computation (if a reduction is done)
    public float cpu_result; // here we store the result of the cpu computation (if a reduction is done)
    

    BenchmarkResults(String benchmarkName, String setupId){
        this.currentBenchmark = benchmarkName;
        this.currentSetupId = setupId;
    }

    public void addPhase(String phaseName, Long phaseDuration, Integer iteration) {
        phases.add(new BenchmarkRecord(phaseName, currentBenchmark, currentSetupId, phaseDuration, iteration));
    }
    
    public void addPhase(BenchmarkRecord record) {
        phases.add(record);
    }
   
    public BenchmarkResults filter(String phaseName){
        BenchmarkResults filtered = new BenchmarkResults(this.currentBenchmark, this.currentSetupId);
        phases.stream().filter(phase -> phase.phaseName.equals(phaseName)).forEach(filtered::addPhase);
        return filtered;
    }

    @Override
    public String toString() {
        return phases
                .stream()
                .map(BenchmarkRecord::toString)
                .collect(Collectors.joining("\n"));
    }
}

class BenchmarkRecord {
    final String phaseName;         // the phase of the benchmark that this class is representing
    final String currentSetupId;    // an identifier indicating the current setup of the benchmark
    final String benchmarkName;     // the name of the benchmark
    final Long executionTime;       // the execution time of the current phase
    final Integer currentIteration; // the current iteration number


    public BenchmarkRecord(String phaseName, String benchmarkName, String setupId, Long executionTime,  Integer currentInteration) {
        this.phaseName = phaseName;
        this.currentSetupId = setupId;
        this.benchmarkName = benchmarkName;
        this.executionTime = executionTime;
        this.currentIteration = currentInteration;
    }

    @Override
    public String toString() {
        return "BenchmarkRecord{" +
                "phaseName='" + phaseName + "'" +
                ", setupId='" + currentSetupId + "'" +
                ", benchmarkName='" + benchmarkName + "'" +
                ", executionTime='" + executionTime + "'" +
                ", iteration='" + currentIteration + "'" +
                '}';
    }

}