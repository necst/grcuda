package com.nvidia.grcuda.test.util.mock;

import org.graalvm.options.OptionDescriptor;
import org.graalvm.options.OptionDescriptors;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;

import java.util.HashMap;
import java.util.Map;

public class OptionValuesMock implements OptionValues {


    private final Map<OptionKey<?>, Object> values;
    private final Map<OptionKey<?>, String> unparsedValues;

    public OptionValuesMock() {
        this.values =  new HashMap<>();
        this.unparsedValues =  new HashMap<>();
    }


    @Override
    public OptionDescriptors getDescriptors() {
        return null;
    }

    @Override
    public <T> void set(OptionKey<T> optionKey, T value) {
        this.values.put(optionKey, value);
    }

    @Override
    public <T> T get(OptionKey<T> optionKey) {
        return (T) this.values.get(optionKey);
    }

    @Override
    public boolean hasBeenSet(OptionKey<?> optionKey) {
        return values.containsKey(optionKey) || unparsedValues.containsKey(optionKey);
    }

    @Override
    public boolean hasSetOptions() {
        return OptionValues.super.hasSetOptions();
    }

}
