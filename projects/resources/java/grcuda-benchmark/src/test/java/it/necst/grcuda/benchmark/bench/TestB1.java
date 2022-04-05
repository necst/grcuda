/*
 * Copyright (c) 2022 NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package it.necst.grcuda.benchmark.bench;

import org.junit.Test;

import it.necst.grcuda.benchmark.Benchmark;
import it.necst.grcuda.benchmark.BenchmarkConfig;
import it.necst.grcuda.benchmark.bench.single_gpu.B1;

/**
 * Test class for B1
 */
public class TestB1 
{
    @Test
    public void version_1()
    {
        BenchmarkConfig currentConfig = new BenchmarkConfig();
        currentConfig.name = "B1";
        currentConfig.setupId = "version_1";
        currentConfig.threadsPerBlock = 64;
        currentConfig.iterations = 100;
        currentConfig.blocks = 8;
        currentConfig.testSize = 100;
        currentConfig.reInit = false;
        currentConfig.randomInit = false;
        currentConfig.cpuValidate = true;
        
        Benchmark b1 = new B1(currentConfig);

        b1.run();

        b1.saveResults();
    }

    @Test
    public void version_2()
    {
        BenchmarkConfig currentConfig = new BenchmarkConfig();
        currentConfig.name = "B1";
        currentConfig.setupId = "version_2";
        currentConfig.threadsPerBlock = 64;
        currentConfig.iterations = 100;
        currentConfig.blocks = 8;
        currentConfig.testSize = 100;
        currentConfig.reInit = false;
        currentConfig.randomInit = false;
        currentConfig.cpuValidate = true;
        
        Benchmark b1 = new B1(currentConfig);

        b1.run();
        b1.saveResults();
    }
}
