sudo /usr/local/cuda/bin/nsys profile -f true -o test.qdrep /home/users/alberto.parravicini/Documents/graalpython_venv/bin/graalpython --jvm --polyglot --grcuda.ExecutionPolicy=default benchmark_main.py -i 10 -n 10000  --reinit false --realloc false  -b b1 --no_cpu_validation  -d; sudo /usr/local/cuda/bin/nsys stats --report gputrace --format csv test.qdrep

nsys profile -f true -o test.qdrep bin/b -n 10000000 -t 10 -p default -r g 176 -d; nsys stats --report gputrace --format csv test.qdrep
