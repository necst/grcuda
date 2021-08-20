# GPU Flag;
# GTX 960 (Maxwell)
GPU_FLAG=sm_50
# GTX 1660 Super (Volta)
# GPU_FLAG=sm_70

# Use NVCC;
CXX=nvcc
FLAGS = -std=c++11 -O3 -arch=${GPU_FLAG}

BIN_FOLDER=build
FILES=main.cu image_pipeline.cu

.PHONY: all 

all:
	$(CXX) $(FILES) $(FLAGS) -o $(BIN_FOLDER)/image_pipeline;
