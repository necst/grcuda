/* Copyright (c) 1993-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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
const System = Java.type("java.lang.System");
const N = 1e6;
const cu = Polyglot.eval('grcuda', 'CU')
const cv = require("opencv4nodejs");

const x = cu.DeviceArray('int', N);
const y = [] 
for (let i = 0; i < N; i++) {
	y[i] = i;
}

// console.log("copy from device array to js")
// for (n = 0; n < 30; n++) {
// 	const start = System.nanoTime();
// 	for (let i = 0; i < N; i++) {
// 		y[i] = x[i];
// 	}
// 	const end = System.nanoTime();
// 	console.log("--copy - js=" + ((end - start) / 1e6) + " ms")

// 	const start2 = System.nanoTime();
// 	x.copyTo(y, N);
// 	const end2 = System.nanoTime();
// 	console.log("--copy - grcuda=" + ((end2 - start2) / 1e6) + " ms")
// }

// console.log("copy to device array from js")
// for (n = 0; n < 30; n++) {
// 	const start = System.nanoTime();
// 	for (let i = 0; i < N; i++) {
// 		x[i] = y[i];
// 	}
// 	const end = System.nanoTime();
// 	console.log("--copy - js=" + ((end - start) / 1e6) + " ms")

// 	const start2 = System.nanoTime();
// 	x.copyFrom(y, N);
// 	const end2 = System.nanoTime();
// 	console.log("--copy - grcuda=" + ((end2 - start2) / 1e6) + " ms")
// }

const img = cv.imread("img_in/lena.jpg", cv.IMREAD_GRAYSCALE).resize(10, 10);
console.log("copy to image from js from device array")
const b = img.getData();
const size = img.rows;

for (let i = 0; i < size; i++) {
	for (let j = 0; j < size; j++) {
		console.log(b[i * size + j]);
	}
}

for (n = 0; n < 30; n++) {
	const start = System.nanoTime();
	for (let i = 0; i < size; i++) {
		for (let j = 0; j < size; j++) {
			x[i * size + j] = b[i * size + j];
		}
	}
	const end = System.nanoTime();
	console.log("--copy - image - js=" + ((end - start) / 1e6) + " ms")
}

for (let i = 0; i < size; i++) {
	for (let j = 0; j < size; j++) {
		console.log(x[i * size + j]);
	}
}