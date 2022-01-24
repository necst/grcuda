package com.nvidia.grcuda.benchmark;


import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class BenchmarkResults {

    private static class BenchmarkRecord {

        private String phaseName;
        private String benchmarkName;
        private Long executionTime;


        public BenchmarkRecord(String phaseName, String benchmarkName, Long executionTime) {
            this.phaseName = phaseName;
            this.benchmarkName = benchmarkName;
            this.executionTime = executionTime;
        }

        public String getPhase() {
            return phaseName;
        }

        public String getBenchmark() {
            return benchmarkName;
        }

        public Long getExecutionTime() {
            return executionTime;
        }

        @Override
        public String toString() {
            return "BenchmarkRecord{" +
                    "phaseName='" + phaseName + '\'' +
                    ", benchmarkName='" + benchmarkName + '\'' +
                    ", executionTime=" + executionTime +
                    '}';
        }

    }

    private class Pair<V1,V2> {
        private V1 v1;
        private V2 v2;

        Pair(V1 v1, V2 v2){
            this.v1 = v1;
            this.v2 = v2;
        }

        public V1 left(){
            return v1;
        }

        public V2 right(){
            return v2;
        }

    }

    private List<BenchmarkRecord> phases = new LinkedList<>();

    private static String currentBenchmark;

    public void addPhase(String phaseName, Long phaseDuration) {
        phases.add(new BenchmarkRecord(phaseName, currentBenchmark, phaseDuration));
    }

    public static void setBenchmark(String benchmark) {
        currentBenchmark = benchmark;
    }

    public void addPhase(BenchmarkRecord record) {
        phases.add(record);
    }

    public BenchmarkResults filter(String phaseName){
        BenchmarkResults filtered = new BenchmarkResults();
        phases.stream().filter(phase -> phase.getPhase().equals(phaseName)).forEach(filtered::addPhase);
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
