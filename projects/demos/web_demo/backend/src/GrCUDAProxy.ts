import WebSocket from 'ws'
import {
  _sleep,
  _getDelayJitter,
  _intervalToMs,
  _gaussianKernel,
  loadImage,
  storeImage,
} from './utils'

import {
  KERNEL_LARGE_DIAMETER,
  KERNEL_LARGE_VARIANCE,
  KERNEL_SMALL_DIAMETER,
  KERNEL_SMALL_VARIANCE,
  KERNEL_UNSHARPEN_DIAMETER,
  KERNEL_UNSHARPEN_VARIANCE,
  UNSHARPEN_AMOUNT,
  NUM_BLOCKS,
  THREADS_1D,
  THREADS_2D,
  DEBUG,
  RESIZED_IMG_WIDTH,
  MOCK_OPTIONS,
  COMPUTATION_MODES,
  CONFIG_OPTIONS,
  BW
} from './options';


import * as ck from "./CudaKernels"

// Load OpenCV;
import cv from "opencv4nodejs";
// Load function to write to file;
import fs from "fs";
// Load GrCUDA;
//@ts-ignore
const cu = Polyglot.eval("grcuda", "CU")

//@ts-ignore
const cudaFree = Polyglot.eval('grcuda', 'cudaFree')


// Use Java System to measure time;
//@ts-ignore
const System = Java.type("java.lang.System");

// Build the CUDA kernels;
const GAUSSIAN_BLUR_KERNEL = cu.buildkernel(ck.GAUSSIAN_BLUR, "gaussian_blur", "const pointer, pointer, sint32, sint32, const pointer, sint32")
const SOBEL_KERNEL = cu.buildkernel(ck.SOBEL, "sobel", "pointer, pointer, sint32, sint32")
const EXTEND_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "extend", "pointer, const pointer, const pointer, sint32, sint32")
const MAXIMUM_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "maximum", "const pointer, pointer, sint32")
const MINIMUM_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "minimum", "const pointer, pointer, sint32")
const UNSHARPEN_KERNEL = cu.buildkernel(ck.UNSHARPEN, "unsharpen", "const pointer, const pointer, pointer, float, sint32")
const COMBINE_KERNEL = cu.buildkernel(ck.COMBINE, "combine", "const pointer, const pointer, const pointer, pointer, sint32")
const COMBINE_KERNEL_2 = cu.buildkernel(ck.COMBINE_2, "combine_2", "const pointer, const pointer, const pointer, pointer, sint32")

export class GrCUDAProxy {
  private ws: WebSocket
  private computationType: string
  private imagesToSend: { [id: string]: Array<string> } = {}


  constructor(ws: WebSocket) {
    this.ws = ws
  }

  /*
   * Begins the computation using the mode specified
   * by `computationType`
   * @param computationType {string}
   * @returns `void`
   */
  public async beginComputation(computationType: string) {
    this.computationType = computationType
    console.log("beginning computation ", computationType)

    COMPUTATION_MODES.forEach(cm => this.imagesToSend[cm] = [])

    if (computationType == "sync" || computationType == "race-sync") {
      await this.computeSync(computationType.toString())
      return
    }
    if (computationType == "async" || computationType == "race-async") {
      await this.computeAsync(computationType.toString())
      return
    }
    if (computationType == "cuda-native" || computationType == "race-cuda-native") {
      await this.computeNative(computationType.toString())
      return
    }

    throw new Error(`Could not recognize computation type: ${computationType}`)

  }

  private communicateAll(imageId: number, computationType: string) {

    const {
      MAX_PHOTOS,
    } = CONFIG_OPTIONS

    this.communicateProgress(imageId / MAX_PHOTOS * 100, computationType)
    this.communicateImageProcessed(imageId, computationType)
  }

  async processImageBW(img: cv.Mat) {
    return new cv.Mat(Buffer.from(await this.processImage(img.getData(), img.rows)), img.rows, img.cols, cv.CV_8UC1);
  }

  async processImageColor(img: cv.Mat) {
    // Possibly not the most efficient way to do this,
    // we should process the 3 channels concurrently, and avoid creation of temporary cv.Mat;
    const channels = img.splitChannels();
    for (let c = 0; c < 3; c++) {
      channels[c] = new cv.Mat(Buffer.from(await this.processImage(channels[c].getData(), img.rows)), img.rows, img.cols, cv.CV_8UC1);;
    }
    return new cv.Mat(channels);
  }


  private async processImage(img: Buffer, size: number, debug: boolean = DEBUG) {
    // Allocate image data;
    const image = cu.DeviceArray("int", size, size);
    const image2 = cu.DeviceArray("float", size, size);
    const image3 = cu.DeviceArray("int", size, size);

    const kernel_small = cu.DeviceArray("float", KERNEL_SMALL_DIAMETER, KERNEL_SMALL_DIAMETER);
    const kernel_large = cu.DeviceArray("float", KERNEL_LARGE_DIAMETER, KERNEL_LARGE_DIAMETER);
    const kernel_unsharpen = cu.DeviceArray("float", KERNEL_UNSHARPEN_DIAMETER, KERNEL_UNSHARPEN_DIAMETER);

    const maximum_1 = cu.DeviceArray("float", 1);
    const minimum_1 = cu.DeviceArray("float", 1);
    const maximum_2 = cu.DeviceArray("float", 1);
    const minimum_2 = cu.DeviceArray("float", 1);

    const mask_small = cu.DeviceArray("float", size, size);
    const mask_large = cu.DeviceArray("float", size, size);
    const image_unsharpen = cu.DeviceArray("float", size, size);

    const blurred_small = cu.DeviceArray("float", size, size);
    const blurred_large = cu.DeviceArray("float", size, size);
    const blurred_unsharpen = cu.DeviceArray("float", size, size);

    // Fill the image data;
    const s1 = System.nanoTime();
    image.copyFrom(img, size * size);
    const e1 = System.nanoTime();
    if (DEBUG)
      console.log("--img to device array=" + _intervalToMs(s1, e1) + " ms");

    const start = System.nanoTime();

    // Create Gaussian kernels;
    _gaussianKernel(kernel_small, KERNEL_SMALL_DIAMETER, KERNEL_SMALL_VARIANCE);
    _gaussianKernel(kernel_large, KERNEL_LARGE_DIAMETER, KERNEL_LARGE_VARIANCE);
    _gaussianKernel(kernel_unsharpen, KERNEL_UNSHARPEN_DIAMETER, KERNEL_UNSHARPEN_VARIANCE);

    // Main GPU computation;
    // Blur - Small;
    GAUSSIAN_BLUR_KERNEL([NUM_BLOCKS, NUM_BLOCKS],
      [THREADS_2D, THREADS_2D],
      4 * KERNEL_SMALL_DIAMETER * KERNEL_SMALL_DIAMETER
    )
      (image, blurred_small, size, size, kernel_small, KERNEL_SMALL_DIAMETER);

    // Blur - Large;
    GAUSSIAN_BLUR_KERNEL([NUM_BLOCKS, NUM_BLOCKS],
      [THREADS_2D, THREADS_2D],
      4 * KERNEL_LARGE_DIAMETER * KERNEL_LARGE_DIAMETER
    )
      (image, blurred_large, size, size, kernel_large, KERNEL_LARGE_DIAMETER);

    // Blur - Unsharpen;
    GAUSSIAN_BLUR_KERNEL([NUM_BLOCKS, NUM_BLOCKS],
      [THREADS_2D, THREADS_2D],
      4 * KERNEL_UNSHARPEN_DIAMETER * KERNEL_UNSHARPEN_DIAMETER
    )
      (image, blurred_unsharpen, size, size, kernel_unsharpen, KERNEL_UNSHARPEN_DIAMETER);

    // Sobel filter (edge detection);
    SOBEL_KERNEL(
      [NUM_BLOCKS, NUM_BLOCKS],
      [THREADS_2D, THREADS_2D]
    )
      (blurred_small, mask_small, size, size);

    SOBEL_KERNEL(
      [NUM_BLOCKS, NUM_BLOCKS],
      [THREADS_2D, THREADS_2D]
    )
      (blurred_large, mask_large, size, size);
    // Ensure that the output of Sobel is in [0, 1];
    MAXIMUM_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (mask_small, maximum_1, size * size);

    MINIMUM_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (mask_small, minimum_1, size * size);

    EXTEND_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (mask_small, minimum_1, maximum_1, size * size, 1);

    // Extend large edge detection mask, and normalize it;
    MAXIMUM_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (mask_large, maximum_2, size * size);

    MINIMUM_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (mask_large, minimum_2, size * size);

    EXTEND_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (mask_large, minimum_2, maximum_2, size * size, 5);

    // Unsharpen;
    UNSHARPEN_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (image, blurred_unsharpen, image_unsharpen, UNSHARPEN_AMOUNT, size * size);

    // Combine results;
    COMBINE_KERNEL(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (image_unsharpen, blurred_large, mask_large, image2, size * size);

    COMBINE_KERNEL_2(
      NUM_BLOCKS * 2,
      THREADS_1D
    )
      (image2, blurred_small, mask_small, image3, size * size);

    // Store the image data.
    const end = System.nanoTime();
    if (DEBUG)
      console.log("--cuda time=" + _intervalToMs(start, end) + " ms");
    const s2 = System.nanoTime();
    image3.copyTo(img, size * size);
    const e2 = System.nanoTime();
    if (DEBUG)
      console.log("--device array to image=" + _intervalToMs(s2, e2) + " ms");

    image.free()
    image2.free()
    image3.free()


    return img;
  }

  /*
   * Compute the GrCUDA kernel using Async mode
   * @returns `void`
   */
  private async computeAsync(computationType: string) {
    console.log("Computing using mode Async")

    const {
      MAX_PHOTOS,
    } = CONFIG_OPTIONS

    for (let imageId = 0; imageId < MAX_PHOTOS; ++imageId) {
      try {
        const imageName = ("0000" + imageId).slice(-4)
        const begin = System.nanoTime();
        const image = await loadImage(imageName)
        const processedImage = BW ? await this.processImageBW(image) : await this.processImageColor(image)
        storeImage(processedImage, imageName)
        this.communicateAll(imageId, computationType)
        const end = System.nanoTime();
        console.log(`One image took ${_intervalToMs(begin, end)}`)
      } catch (e) {
        console.log(e)
        continue
      }
    }




  }

  /*
   * Compute the GrCUDA kernel using Sync mode
   * @returns `void`
   */
  private async computeSync(computationType: string) {
    console.log("Computing using mode Sync")
    await this.mockCompute(computationType)
  }

  /*
   * Compute the GrCUDA kernel using native 
   * CUDA code by `exec`ing the kernel via 
   * a shell
   * @returns `void`
   */
  private async computeNative(computationType: string) {
    console.log("Computing using mode Native")
    await this.mockCompute(computationType)
  }

  /* Mock the computation of the kernels 
   * inside GrCUDA.
   * Sends a `progress` message every time an image is computed
   * and a `image` message every time BATCH_SIZE images have been computed
   */
  private async mockCompute(computationType: string) {

    const {
      DELAY
    } = MOCK_OPTIONS

    const {
      MAX_PHOTOS,
    } = CONFIG_OPTIONS

    let delay_jitter = _getDelayJitter(computationType)

    for (let imageId = 0; imageId < MAX_PHOTOS; ++imageId) {
      // This does mock the actual computation that will happen 
      // in the CUDA realm
      await _sleep(DELAY + Math.random() * delay_jitter)
      this.communicateAll(imageId, computationType)
    }
    this.communicateAll(MAX_PHOTOS, computationType)


  }

  private communicateProgress(data: number, computationType: string) {
    const {
      MAX_PHOTOS
    } = CONFIG_OPTIONS

    this.ws.send(JSON.stringify({
      type: "progress",
      data: data,
      computationType
    }))
  }

  private communicateImageProcessed(imageId: number, computationType: string) {
    const {
      SEND_BATCH_SIZE,
      MAX_PHOTOS
    } = CONFIG_OPTIONS

    this.imagesToSend[computationType].push(`./images/thumb/${("0000" + imageId).slice(-4)}.jpg`)

    if ((imageId !== 0 && !(imageId % SEND_BATCH_SIZE) || imageId === MAX_PHOTOS - 1)) {

      this.ws.send(JSON.stringify({
        type: "image",
        images: this.imagesToSend[computationType],
        computationType
      }))

      this.imagesToSend[computationType] = []

    }
  }

}

