example.exe : example.cu jacobi.o
	nvcc example.cu jacobi.o -o example.exe

jacobi.o : jacobi.cu jacobi.cuh
	nvcc -dc jacobi.cu

.PHONY: cleangraphs
cleangraphs:
	rm graphs/*.svg

