// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <string>
#include <iostream>
#include <ctime>    // For time()
#include <cstdlib>  // For srand()
#include "options.hpp"
#include "benchmark.cuh"

#include "single_gpu/b1.cuh"
#include "single_gpu/b5.cuh"
#include "single_gpu/b6.cuh"
#include "single_gpu/b7.cuh"
#include "single_gpu/b8.cuh"
#include "single_gpu/b10.cuh"
#include "multi_gpu/b1.cuh"
#include "multi_gpu/b5.cuh"
#include "multi_gpu/b6.cuh"
#include "multi_gpu/b9.cuh"
#include "multi_gpu/b11.cuh"
#include "multi_gpu/b12.cuh"
#include "multi_gpu/b13.cuh"

int main(int argc, char *argv[])
{
    // srand(time(0));
    srand(12);
    
    Options options = Options(argc, argv);
    BenchmarkEnum benchmark_choice = options.benchmark_choice;
    Benchmark *b;

    switch (benchmark_choice)
    {
    case BenchmarkEnum::B1:
        b = new Benchmark1(options);
        break;
    case BenchmarkEnum::B5:
        b = new Benchmark5(options);
        break;
    case BenchmarkEnum::B6:
        b = new Benchmark6(options);
        break;
    case BenchmarkEnum::B7:
        b = new Benchmark7(options);
        break;
    case BenchmarkEnum::B8:
        b = new Benchmark8(options);
        break;
    case BenchmarkEnum::B10:
        b = new Benchmark10(options);
        break;
    case BenchmarkEnum::B1M:
        b = new Benchmark1M(options);
        break;
    case BenchmarkEnum::B5M:
        b = new Benchmark5M(options);
        break;
    case BenchmarkEnum::B6M:
        b = new Benchmark6M(options);
        break;
    case BenchmarkEnum::B9M:
        b = new Benchmark9M(options);
        break;
    case BenchmarkEnum::B11M:
        b = new Benchmark11M(options);
        break;
    case BenchmarkEnum::B12M:
        b = new Benchmark12M(options);
        break;
    case BenchmarkEnum::B13M:
        b = new Benchmark13M(options);
        break;
    default:
        break;
    }
    if (b != nullptr) {
        b->run();
    } else {
        std::cout << "ERROR = benchmark is null" << std::endl;
    }
}