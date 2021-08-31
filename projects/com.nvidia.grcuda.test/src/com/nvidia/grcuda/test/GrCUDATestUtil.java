package com.nvidia.grcuda.test;

import org.graalvm.polyglot.Context;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class GrCUDATestUtil {
    public static Collection<Object[]> crossProduct(List<Object[]> sets) {
        int solutions = 1;
        List<Object[]> combinations = new ArrayList<>();
        for (Object[] objects : sets) {
            solutions *= objects.length;
        }
        for(int i = 0; i < solutions; i++) {
            int j = 1;
            List<Object> current = new ArrayList<>();
            for(Object[] set : sets) {
                current.add(set[(i / j) % set.length]);
                j *= set.length;
            }
            combinations.add(current.toArray(new Object[0]));
        }
        return combinations;
    }

    /**
     * Return a list of {@link GrCUDATestOptionsStruct}, where each elemenent is a combination of input policy options.
     * Useful to perform tests that cover all cases;
     * @return the cross-product of all options
     */
    public static Collection<Object[]> getAllOptionCombinations() {
        Collection<Object[]> options = GrCUDATestUtil.crossProduct(Arrays.asList(new Object[][]{
                {"sync", "default"},        // ExecutionPolicy
                {true, false},              // InputPrefetch
                {"fifo", "always-new"},     // RetrieveNewStreamPolicy
                {"default", "disjoint"},    // RetrieveParentStreamPolicy
                {"default", "with-const"},  // DependencyPolicy
                {true, false},              // ForceStreamAttach
        }));
        List<Object[]> combinations = new ArrayList<>();
        options.forEach(optionArray -> {
            List<Object> current = new ArrayList<>();
            current.add(new GrCUDATestOptionsStruct(
                    (String) optionArray[0], (boolean) optionArray[1],
                    (String) optionArray[2], (String) optionArray[3],
                    (String) optionArray[4], (boolean) optionArray[5]));
            combinations.add(current.toArray(new Object[0]));
        });
        return combinations;
    }

    public static Context createContextFromOptions(GrCUDATestOptionsStruct options) {
        return Context.newBuilder().allowExperimentalOptions(true)
                .option("grcuda.ExecutionPolicy", options.policy)
                .option("grcuda.InputPrefetch", String.valueOf(options.inputPrefetch))
                .option("grcuda.RetrieveNewStreamPolicy", options.retrieveNewStreamPolicy)
                .option("grcuda.RetrieveParentStreamPolicy", options.retrieveParentStreamPolicy)
                .option("grcuda.DependencyPolicy", options.dependencyPolicy)
                .option("grcuda.ForceStreamAttach", String.valueOf(options.forceStreamAttach))
                .allowAllAccess(true).build();
    }
}


