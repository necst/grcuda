# Use NVCC;
CXX=nvcc
FLAGS = -std=c++11 -O3 -arch=sm_70
FLAG_NVML = -lnvidia-ml
# Use Clang;
# CXX=$(CLANG_DIR)/clang++
# FLAGS = --cuda-gpu-arch=sm_70 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -std=c++11 -O3

BIN_FOLDER=bin
FILES= main.cu benchmark.cu b1.cu b5.cu b6.cu b7.cu b8.cu b10.cu b11.cu b15.cu b16.cu b17.cu b18.cu b20.cu b21.cu b22.cu
#FILES= main.cu benchmark.cu b22.cu
.PHONY: full all b1 b5 b6 b7 b8 b10 b11 b15 b16 b17 b18 b20 b21 b22
#.PHONY: full all b22

#data_transfer: data_transfer.cu
#	$(CXX) -o data_transfer data_transfer.cu $(FLAG_NVML)
#LU: LU_v2.cu
#	$(CXX) -o LU LU_v2.cu

full:
	$(CXX) $(FILES) $(FLAGS) -o $(BIN_FOLDER)/b;

all: \
    full \
	b1 \
	b5 \
	b6 \
	b7 \
	b8 \
	b10 \
	b11 \
    b15 \
    b16 \
    b17 \
    b18 \
    b20 \
	b21 \
	b22 \

b1: b1*
	$(CXX) old/b1_default.cu $(FLAGS) -o $(BIN_FOLDER)/b1_default;
	$(CXX) old/b1_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b1_sync

b5: b5*
	$(CXX) old/b5_default.cu $(FLAGS) -o $(BIN_FOLDER)/b5_default;
	$(CXX) old/b5_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b5_sync

b6: b6*
	$(CXX) old/b6_default.cu $(FLAGS) -o $(BIN_FOLDER)/b6_default;
	$(CXX) old/b6_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b6_sync

b7: b7*
	$(CXX) old/b7_default.cu $(FLAGS) -o $(BIN_FOLDER)/b7_default;
	$(CXX) old/b7_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b7_sync

b8: b8*
	$(CXX) old/b8_default.cu $(FLAGS) -o $(BIN_FOLDER)/b8_default;
	$(CXX) old/b8_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b8_sync

b10: b10*
	$(CXX) old/b10_default.cu $(FLAGS) -o $(BIN_FOLDER)/b10_default;
	$(CXX) old/b10_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b10_sync
b11: b11*
	$(CXX) b11.cu $(FLAGS) -o $(BIN_FOLDER)/b11;

b15: b15*
	$(CXX) b15.cu $(FLAGS) -o $(BIN_FOLDER)/b15;

b16: b16*
	$(CXX) b16.cu $(FLAGS) -o $(BIN_FOLDER)/b16;

b17: b17*
	$(CXX) b17.cu $(FLAGS) -o $(BIN_FOLDER)/b17;

b18: b18*
	$(CXX) b18.cu $(FLAGS) -o $(BIN_FOLDER)/b18;

b20: b20*
	$(CXX) b20.cu $(FLAGS) -o $(BIN_FOLDER)/b20;

b21: b21*
	$(CXX) b21.cu $(FLAGS) -o $(BIN_FOLDER)/b21;

b22: b22*
	$(CXX) b22.cu $(FLAGS) -o $(BIN_FOLDER)/b22;
