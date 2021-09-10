/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda.test;

import static org.junit.Assert.assertEquals;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.Test;
import com.nvidia.grcuda.gpu.LittleEndianNativeArrayView;
import com.nvidia.grcuda.gpu.OffheapMemory;

public class DeviceArrayCopyFunctionTest {

    @Test
    public void testDeviceArrayCopyFromOffheapMemory() {
        final int numElements = 1000;
        final int numBytesPerInt = 4;
        final int numBytes = numElements * numBytesPerInt;
        try (OffheapMemory hostMemory = new OffheapMemory(numBytes)) {
            // create off-heap host memory of integers: [1, 2, 3, 4, ..., 1000]
            LittleEndianNativeArrayView hostArray = hostMemory.getLittleEndianView();
            for (int i = 0; i < numElements; ++i) {
                hostArray.setInt(i, i + 1);
            }
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                // create DeviceArray and copy content from off-heap host memory into it
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                Value deviceArray = createDeviceArray.execute("int", numElements);
                deviceArray.invokeMember("copyFrom", hostMemory.getPointer(), numElements);

                // Verify content of device array
                for (int i = 0; i < numElements; ++i) {
                    assertEquals(i + 1, deviceArray.getArrayElement(i).asInt());
                }
            }
        }
    }

    @Test
    public void testDeviceArrayCopyToOffheapMemory() {
        final int numElements = 1000;
        final int numBytesPerInt = 4;
        final int numBytes = numElements * numBytesPerInt;
        try (OffheapMemory hostMemory = new OffheapMemory(numBytes)) {
            // create off-heap host memory of integers and initialize all elements to zero.
            LittleEndianNativeArrayView hostArray = hostMemory.getLittleEndianView();
            for (int i = 0; i < numElements; ++i) {
                hostArray.setInt(i, i);
            }
            try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
                // create DeviceArray and set its content [1, 2, 3, 4, ..., 1000]
                Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
                Value deviceArray = createDeviceArray.execute("int", numElements);
                for (int i = 0; i < numElements; ++i) {
                    deviceArray.setArrayElement(i, i + 1);
                }
                // copy content of device array to off-heap host memory
                deviceArray.invokeMember("copyTo", hostMemory.getPointer(), numElements);

                // Verify content of device array
                for (int i = 0; i < numElements; ++i) {
                    assertEquals(i + 1, hostArray.getInt(i));
                }
            }
        }
    }

    @Test
    public void testDeviceArrayCopyFromDeviceArray() {
        final int numElements = 1000;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                sourceDeviceArray.setArrayElement(i, i + 1);
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                destinationDeviceArray.setArrayElement(i, 0);
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements);
            // Verify content of device array
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i + 1, destinationDeviceArray.getArrayElement(i).asInt());
            }
        }
    }

    @Test
    public void testDeviceArrayCopyToDeviceArray() {
        final int numElements = 1000;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                sourceDeviceArray.setArrayElement(i, i + 1);
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements);
            for (int i = 0; i < numElements; ++i) {
                destinationDeviceArray.setArrayElement(i, 0);
            }
            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements);
            // Verify content of device array
            for (int i = 0; i < numElements; ++i) {
                assertEquals(i + 1, destinationDeviceArray.getArrayElement(i).asInt());
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArray() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArray() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);

            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayRow() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // Set each rows 3 to j;
            for (int j = 0; j < numElements2; ++j) {
                sourceDeviceArray.getArrayElement(3).setArrayElement(j, j);
            }

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements2);

            sourceDeviceArray.getArrayElement(3).invokeMember("copyTo", destinationDeviceArray, sourceDeviceArray.getArrayElement(3).getArraySize());
            // Verify content of device array
            for (int j = 0; j < numElements2; ++j) {
                assertEquals(j, destinationDeviceArray.getArrayElement(j).asInt());
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArrayRow() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }

            // create destination device array.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements2);
            // Set each value to j;
            for (int j = 0; j < numElements2; ++j) {
                destinationDeviceArray.setArrayElement(j, 42 + j);
            }

            sourceDeviceArray.getArrayElement(3).invokeMember("copyFrom", destinationDeviceArray, sourceDeviceArray.getArrayElement(3).getArraySize());
            // Verify content of device array
            for (int j = 0; j < numElements2; ++j) {
                assertEquals(42 + j, sourceDeviceArray.getArrayElement(3).getArrayElement(j).asInt());
            }
            // Everything else is unmodified;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    if (i != 3) {
                        assertEquals(i * numElements2 + j, sourceDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                    }
                }
            }
        }
    }

    // FIXME: memcpy on single dimensions of column-major arrays doesn't work, as memory is not contiguous;

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArrayF() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(i * numElements2 + j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayF() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");

            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(i * numElements2 + j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArrayC() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");

            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");

            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }
            destinationDeviceArray.invokeMember("copyFrom", sourceDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(i * numElements2 + j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayC() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // create device array initialize its elements.
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");
            // Set each row to j;
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, j);
                }
            }
            // create destination device array initialize its elements to zero.
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "C");

            sourceDeviceArray.invokeMember("copyTo", destinationDeviceArray, numElements1 * numElements2);
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals(j, destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayRowC() {
        final int numElements1 = 5;
        final int numElements2 = 7;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // Create device array initialize its elements;
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // Initialize elements with unique values.
            // Values are still written as (row, col), even if the storage is "C";
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }
            // Initialize destination array with unique values, to ensure that it's modified correctly;
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    destinationDeviceArray.getArrayElement(i).setArrayElement(j,  -(i * numElements2 + j));
                }
            }

            // This copies the 4th row of the source array into the 4th row of the destination array;
            sourceDeviceArray.getArrayElement(3).invokeMember("copyTo", destinationDeviceArray.getArrayElement(3), sourceDeviceArray.getArrayElement(3).getArraySize());

            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals((i == 3 ? 1 : -1) *(i * numElements2 + j), destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyFromDeviceArrayRowC() {
        final int numElements1 = 10;
        final int numElements2 = 25;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // Create device array initialize its elements;
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            // Initialize elements with unique values.
            // Values are still written as (row, col), even if the storage is "C";
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }
            // Initialize destination array with unique values, to ensure that it's modified correctly;
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2);
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    destinationDeviceArray.getArrayElement(i).setArrayElement(j,  -(i * numElements2 + j));
                }
            }

            sourceDeviceArray.getArrayElement(3).invokeMember("copyFrom", destinationDeviceArray.getArrayElement(3), destinationDeviceArray.getArrayElement(3).getArraySize());
            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals((i == 3 ? -1 : 1) * (i * numElements2 + j), sourceDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }

    @Test
    public void testMultiDimDeviceArrayCopyToDeviceArrayRowF() {
        final int numElements1 = 5;
        final int numElements2 = 7;
        try (Context ctx = GrCUDATestUtil.buildTestContext().build()) {
            Value createDeviceArray = ctx.eval("grcuda", "DeviceArray");
            // Create device array initialize its elements;
            Value sourceDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
            // Initialize elements with unique values.
            // Values are still written as (row, col), even if the storage is "C";
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    sourceDeviceArray.getArrayElement(i).setArrayElement(j, i * numElements2 + j);
                }
            }
            // Initialize destination array with unique values, to ensure that it's modified correctly;
            Value destinationDeviceArray = createDeviceArray.execute("int", numElements1, numElements2, "F");
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    destinationDeviceArray.getArrayElement(i).setArrayElement(j,  -(i * numElements2 + j));
                }
            }
            System.out.println("source");
            for (int i = 0; i < numElements1; ++i) {
                System.out.print(i + "=[");
                for (int j = 0; j < numElements2; ++j) {
                    System.out.print(sourceDeviceArray.getArrayElement(i).getArrayElement(j) + ", ");
                }
                System.out.println("]");
            }
            System.out.println("destination");
            for (int i = 0; i < numElements1; ++i) {
                System.out.print(i + "=[");
                for (int j = 0; j < numElements2; ++j) {
                    System.out.print(destinationDeviceArray.getArrayElement(i).getArrayElement(j) + ", ");
                }
                System.out.println("]");
            }

            // This copies the 4th column of the source array into the 4th column of the destination array;
            sourceDeviceArray.getArrayElement(3).invokeMember("copyTo", destinationDeviceArray.getArrayElement(3), sourceDeviceArray.getArrayElement(3).getArraySize());

            System.out.println("destination-after");
            for (int i = 0; i < numElements1; ++i) {
                System.out.print(i + "=[");
                for (int j = 0; j < numElements2; ++j) {
                    System.out.print(destinationDeviceArray.getArrayElement(i).getArrayElement(j) + ", ");
                }
                System.out.println("]");
            }

            // Verify content of device array
            for (int i = 0; i < numElements1; ++i) {
                for (int j = 0; j < numElements2; ++j) {
                    assertEquals((i == 3 ? 1 : -1) *(i * numElements2 + j), destinationDeviceArray.getArrayElement(i).getArrayElement(j).asInt());
                }
            }
        }
    }
}
