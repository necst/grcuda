FLAGS = -std=c++11 -O3 -use_fast_math -arch=sm_70

BIN_FOLDER=bin/original

.PHONY: all axpy dot conv hospot mc mmul autocov backprop bfs pr
all: \
	axpy \
	dot \
	conv \
	hotspot \
	mc \
	mmul \
	autocov \
	backprop \
	bfs \
	pr

axpy: axpy/*
	nvcc axpy/axpy.cu $(FLAGS) -o $(BIN_FOLDER)/axpy

dot: dot_product/*
	nvcc dot_product/dot_product.cu $(FLAGS) -o $(BIN_FOLDER)/dot

conv: convolution/*
	nvcc convolution/convolution.cu $(FLAGS) -o $(BIN_FOLDER)/conv

hotspot: hotspot/*
	nvcc hotspot/hotspot.cu $(FLAGS) -o $(BIN_FOLDER)/hotspot

mc: montecarlo/*
	nvcc -lcurand montecarlo/MonteCarlo_gold.cpp montecarlo/MonteCarlo_kernel.cu montecarlo/MonteCarloMultiGPU.cpp $(FLAGS) -o $(BIN_FOLDER)/mc

mmul: mmul/*
	nvcc mmul/mmul.cu $(FLAGS) -o $(BIN_FOLDER)/mmul

autocov: autocov/*
	nvcc autocov/autocov.cu $(FLAGS) -o $(BIN_FOLDER)/autocov

backprop: backprop/*
	nvcc backprop/backprop.cu $(FLAGS) -o $(BIN_FOLDER)/backprop

bfs: bfs/*
	nvcc bfs/bfs.cu $(FLAGS) -o $(BIN_FOLDER)/bfs	

pr: pr/*
	nvcc pr/pr.cu $(FLAGS) -o $(BIN_FOLDER)/pr	
