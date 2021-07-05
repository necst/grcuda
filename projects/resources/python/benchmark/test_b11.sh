graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=fifo  --grcuda.RetrieveParentStreamPolicy=stream_aware --grcuda.DependencyPolicy=with-const  benchmark_main.py -g 1024 --block_size_1d 32 -i 3 -d -n 40000 -b b11 --no_cpu_validation;


graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=fifo  --grcuda.RetrieveParentStreamPolicy=stream_aware --grcuda.DependencyPolicy=with-const  benchmark_main.py -g 1024 --block_size_1d 32 -i 3 -d -n 40000 -b b11 --no_cpu_validation;
