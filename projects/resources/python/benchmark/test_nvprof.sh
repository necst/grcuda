#B1

nvprof --export-profile prof_multi_gpu_b1_%p.nvvp --profile-from-start off --print-gpu-trace --profile-child-processes --csv --log-file prof_multi_gpu_b1_%p.csv graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -d -i 3 -n 1000000000 -b b1 --nvprof --no_cpu_validation;

#B5
nvprof --export-profile prof_multi_gpu_b5_%p.nvvp --profile-from-start off --print-gpu-trace --profile-child-processes --csv --log-file prof_multi_gpu_b5_%p.csv graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -d -i 3 -n 100000000 -b b5 --nvprof --no_cpu_validation;

#B6
nvprof --export-profile prof_multi_gpu_b6_%p.nvvp --profile-from-start off --print-gpu-trace --profile-child-processes --csv --log-file prof_multi_gpu_b6_%p.csv graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -d -i 3 -n 6000000 -b b6 --nvprof --no_cpu_validation;

#B7
nvprof --export-profile prof_multi_gpu_b7_%p.nvvp --profile-from-start off --print-gpu-trace --profile-child-processes --csv --log-file prof_multi_gpu_b7_%p.csv graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -d -i 3 -n 20000000 -b b7 --nvprof --no_cpu_validation;

#B8
nvprof --export-profile prof_multi_gpu_b8_%p.nvvp --profile-from-start off --print-gpu-trace --profile-child-processes --csv --log-file prof_multi_gpu_b8_%p.csv graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -d -i 3 -n 12000 -b b8 --nvprof --no_cpu_validation;

#B10
nvprof --export-profile prof_multi_gpu_b10_%p.nvvp --profile-from-start off --print-gpu-trace --profile-child-processes --csv --log-file prof_multi_gpu_b10_%p.csv graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -d -i 3 -n 10000 -b b10 --nvprof --no_cpu_validation;
