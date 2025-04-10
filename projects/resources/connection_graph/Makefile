# Copyright (c) 2021, 2022, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Use NVCC.
# Set the appropriate GPU architecture, check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
CXX=nvcc
FLAGS = -O3 -std=c++11 -arch=sm_70

# (Experimental) Use Clang;
# CXX=$(CLANG_DIR)/clang++
# FLAGS = --cuda-gpu-arch=sm_70 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -std=c++11 -O3

BIN_FOLDER=bin
FILES=connection_graph.cu
.PHONY: all clean

all:
	mkdir -p $(BIN_FOLDER);
	$(CXX) $(FILES) $(FLAGS) -o $(BIN_FOLDER)/connection_graph;

clean:
	rm $(BIN_FOLDER)/*;
