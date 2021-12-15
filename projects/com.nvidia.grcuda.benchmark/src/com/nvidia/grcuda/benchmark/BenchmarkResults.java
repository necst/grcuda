package com.nvidia.grcuda.benchmark;


import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class BenchmarkResults {

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

    private List<Pair<String, Long>> phases = new LinkedList<>();

    public void addPhase(String phaseName, Long phaseDuration) {
        phases.add(new Pair<>(phaseName, phaseDuration));
    }


    public void addPhase(Pair<String, Long> phase) {
        phases.add(phase);
    }

    public BenchmarkResults filter(String phaseName){
        BenchmarkResults filtered = new BenchmarkResults();
        phases.stream().filter(phase -> phase.left().equals(phaseName)).forEach(filtered::addPhase);
        return filtered;
    }

    @Override
    public String toString() {
        return phases
                .stream()
                .map(entry -> entry.left() + " -> " + entry.right())
                .collect(Collectors.joining("\n"));
    }
}
