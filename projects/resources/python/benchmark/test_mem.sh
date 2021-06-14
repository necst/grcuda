#always-new, data_aware, with-const
#benchmark B1 
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new  --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware benchmark_main.py -g 512 --block_size_1d 256  -d -i 3 -n 1000000000 -b b1 --no_cpu_validation;
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 1000000000 -b b1 --no_cpu_validation;

#benchmark B5
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 100000000 -b b5 --no_cpu_validation;
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 100000000 -b b5 --no_cpu_validation;

#benchmark B6
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 6000000 -b b6 --no_cpu_validation;
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 6000000 -b b6 --no_cpu_validation;

#benchmark b7
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 20000000 -b b7 --no_cpu_validation;
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 20000000 -b b7 --no_cpu_validation;

#benchmark b8
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 12000 -b b8 --no_cpu_validation;
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 12000 -b b8 --no_cpu_validation;

#benchmark b10
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=2 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 10000 -b b10 --no_cpu_validation;
graalpython --jvm --polyglot --grcuda.NumberOfGPUs=1 --experimental-options --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.InputPrefetch --grcuda.RetrieveParentStreamPolicy=disjoint_data_aware  benchmark_main.py -g 512 --block_size_1d 256 -d -i 3 -n 10000 -b b10 --no_cpu_validation;
