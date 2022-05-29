package it.necst.grcuda.benchmark;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;
import it.necst.grcuda.benchmark.bench.single_gpu.B5M;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Ignore;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.text.SimpleDateFormat;
import java.util.*;


/*
    TODO:
        1) missing  create_block_size_list()
        2) missing execute_grcuda_benchmark()
        3) the function written in runAll_gtx1660_super() can be abstracted to be used with all the gpus
 */
public class TestBenchmarks {
    private static final String PATH = System.getenv("GRCUDA_HOME")+"/projects/resources/java/grcuda-benchmark/src/test/java/it/necst/grcuda/benchmark";

    /*
        This method reflects the pattern of benchmark_wrapper.py present in the python suite.
        //TODO: Proper refactoring should be done to generate the set of tests needed from the json file
     */
    private void iterateAllPossibleConfig(Config parsedConfig, GPUs currentGPUmodel) throws ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        String BANDWIDTH_MATRIX = System.getenv("GRCUDA_HOME")+"/projects/resources/connection_graph/datasets/connection_graph_2_v100.csv";

        ArrayList<String> dp, nsp, psp, cdp;
        ArrayList<Integer> ng, block_sizes;
        Integer nb; // number of blocks
        Integer blockSize1D, blockSize2D;
        int num_iter = parsedConfig.num_iter;
        boolean time_phases = false; // TODO: add value to json and to the Config class
        boolean debug = true;  // TODO: add value to json and to the Config class
        boolean mock = true; // TODO: FOR NOW SIMULATE ONLY THE COMMAND
        String output_date = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss", Locale.ITALIAN).format(new Date());

        Benchmark benchToRun;
        for(String bench : parsedConfig.benchmarks){ // given bench X from the set of all the benchmarks iterate over the number of elements associated with that benchmark
            ArrayList<Integer> sizes = parsedConfig.num_elem.get(bench);
            if(sizes == null) continue; //skip everything if no sizes are specified for the current bench
            for(Integer curr_size : sizes){ // given a specific input size iterate over the various execution policies
                for(String policy : parsedConfig.exec_policies){
                    if(policy.equals("sync")){
                        dp = new ArrayList<>(List.of(parsedConfig.dependency_policies.get(0)));
                        nsp = new ArrayList<>(List.of(parsedConfig.new_stream_policies.get(0)));
                        psp = new ArrayList<>(List.of(parsedConfig.parent_stream_policies.get(0)));
                        cdp = new ArrayList<>(List.of(parsedConfig.choose_device_policies.get(0)));
                        ng = new ArrayList<>(List.of(1));
                    }
                    else{
                        dp = parsedConfig.dependency_policies;
                        nsp = parsedConfig.new_stream_policies;
                        psp = parsedConfig.parent_stream_policies;
                        cdp = parsedConfig.choose_device_policies;
                        ng = parsedConfig.num_gpus;
                    }
                    for(int num_gpu : ng){
                        if(policy.equals("async") && num_gpu == 1){
                            dp = new ArrayList<>(List.of(parsedConfig.dependency_policies.get(0)));
                            nsp = new ArrayList<>(List.of(parsedConfig.new_stream_policies.get(0)));
                            psp = new ArrayList<>(List.of(parsedConfig.parent_stream_policies.get(0)));
                            cdp = new ArrayList<>(List.of(parsedConfig.choose_device_policies.get(0)));
                        }
                        else{
                            dp = parsedConfig.dependency_policies;
                            nsp = parsedConfig.new_stream_policies;
                            psp = parsedConfig.parent_stream_policies;
                            cdp = parsedConfig.choose_device_policies;
                        }
                        for(String m : parsedConfig.memory_advise){
                            for(Boolean p : parsedConfig.prefetch ){
                                for(Boolean s : parsedConfig.stream_attach){
                                    for(Boolean t : parsedConfig.time_computation){ // select the correct connection graph
                                        if(currentGPUmodel == GPUs.V100){
                                            BANDWIDTH_MATRIX = System.getenv("GRCUDA_HOME")
                                                    + "/projects/resources/connection_graph/datasets"
                                                    +"/connection_graph_" + num_gpu + "_v100.csv";
                                        }
                                        else if (currentGPUmodel == GPUs.A100) {
                                            BANDWIDTH_MATRIX = System.getenv("GRCUDA_HOME")
                                                    + "/projects/resources/connection_graph/datasets"
                                                    +"/connection_graph_8_a100.csv";
                                        }
                                        for(String dependency_policy : dp){
                                            for(String new_stream_policy : nsp){
                                                for(String parent_stream_policy : psp){
                                                    for(String choose_device_policy : cdp){
                                                        //TODO: replace BenchmarkConfig init with constructor call
                                                        BenchmarkConfig config = new BenchmarkConfig();

                                                        nb = parsedConfig.numBlocks.get(bench);
                                                        if(nb != null) config.numBlocks = nb;

                                                        blockSize1D = parsedConfig.block_size1d.get(bench);
                                                        if(blockSize1D != null) config.blockSize1D = blockSize1D;

                                                        blockSize2D = parsedConfig.block_size2d.get(bench);
                                                        if(blockSize2D != null) config.blockSize2D = blockSize2D;

                                                        config.debug = parsedConfig.debug;
                                                        config.benchmarkName = bench;
                                                        config.size = curr_size;
                                                        config.numGpus = num_gpu;
                                                        config.executionPolicy = policy;
                                                        config.dependencyPolicy = dependency_policy;
                                                        config.retrieveNewStreamPolicy = new_stream_policy;
                                                        config.retrieveParentStreamPolicy = parent_stream_policy;
                                                        config.deviceSelectionPolicy = choose_device_policy;
                                                        config.inputPrefetch = p;
                                                        config.totIter = num_iter;
                                                        config.forceStreamAttach = s;
                                                        config.memAdvisePolicy = m;
                                                        config.bandwidthMatrix = BANDWIDTH_MATRIX;
                                                        config.enableComputationTimers =t;

                                                        System.out.println(config);
                                                        benchToRun = createBench(config);
                                                        benchToRun.run();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private Benchmark createBench(BenchmarkConfig config) throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        // Courtesy of https://stackoverflow.com/questions/7495785/java-how-to-instantiate-a-class-from-string

        Class currBenchClass = Class.forName("it.necst.grcuda.benchmark.bench"+".single_gpu."+config.benchmarkName);

        Class[] types = {BenchmarkConfig.class};
        Constructor constructor = currBenchClass.getConstructor(types);

        Object[] parameters = {config};

        return (Benchmark) constructor.newInstance(parameters);

    }

    @Ignore
    @Test
    public void runAll_gtx1660_super() throws FileNotFoundException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        GPUs gpuModel = GPUs.GTX1660_SUPER;
        // TODO: given that we are following the selection procedure of the benchmarks in python
        //      we need to be sure that we have inserted the values in the same order like "dependencies_policies"

        // get the configuration for the selected GPU into a Config class
        String CONFIG_PATH = PATH + "/config_GTX1660_super.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);
        System.out.println(gson.toJson(parsedConfig)); // print the current configuration

        iterateAllPossibleConfig(parsedConfig, gpuModel);

        /*
            TODO:
                0) skip the test if the gpu is not the one present in the system
                1) parse the json config file
                2) sequentially run all the specified benchmarks configurations like in python code
         */
    }

    @Test
    public void runAll_gtx960_multi() throws FileNotFoundException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        GPUs gpuModel = GPUs.GTX960;
        // TODO: given that we are following the selection procedure of the benchmarks in python
        //      we need to be sure that we have inserted the values in the same order like "dependencies_policies"

        // get the configuration for the selected GPU into a Config class
        String CONFIG_PATH = PATH + "/config_GTX960.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);
        System.out.println(gson.toJson(parsedConfig)); // print the current configuration

        iterateAllPossibleConfig(parsedConfig, gpuModel);

        /*
            TODO:
                0) skip the test if the gpu is not the one present in the system
                1) parse the json config file
                2) sequentially run all the specified benchmarks configurations like in python code
         */
    }

}

enum GPUs {
    GTX1660_SUPER,
    A100,
    V100,
    GTX960
}

/**
 * Used to map/parse the json config files to a class
 */
class Config {
    int num_iter;
    int heap_size;

    boolean reInit = false;
    boolean randomInit;
    boolean cpuValidation;
    boolean debug;

    ArrayList<String> benchmarks;
    ArrayList<String> exec_policies;
    ArrayList<String> dependency_policies;
    ArrayList<String> new_stream_policies;
    ArrayList<String> parent_stream_policies;
    ArrayList<String> choose_device_policies;
    ArrayList<String> memory_advise;

    ArrayList<Boolean> prefetch;
    ArrayList<Boolean> stream_attach;
    ArrayList<Boolean> time_computation;

    ArrayList<Integer> num_gpus;

    HashMap<String, ArrayList<Integer>> num_elem;
    HashMap<String, Integer> numBlocks;
    HashMap<String, Integer> block_size1d;
    HashMap<String, Integer> block_size2d;

    @Override
    public String toString() {
        return "Config{" +
                "num_iter=" + num_iter +
                ", heap_size=" + heap_size +
                ", reInit=" + reInit +
                ", randomInit=" + randomInit +
                ", cpuValidation=" + cpuValidation +
                ", benchmarks=" + benchmarks +
                ", exec_policies=" + exec_policies +
                ", dependency_policies=" + dependency_policies +
                ", new_stream_policies=" + new_stream_policies +
                ", parent_stream_policies=" + parent_stream_policies +
                ", choose_device_policies=" + choose_device_policies +
                ", memory_advise=" + memory_advise +
                ", prefetch=" + prefetch +
                ", stream_attach=" + stream_attach +
                ", time_computation=" + time_computation +
                ", num_gpus=" + num_gpus +
                ", num_elem=" + num_elem +
                ", numBlocks=" + numBlocks +
                ", block_size1d=" + block_size1d +
                ", block_size2d=" + block_size2d +
                '}';
    }
}


