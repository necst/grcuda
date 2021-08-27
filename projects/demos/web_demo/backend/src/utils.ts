import cv from "opencv4nodejs"
import fs from "fs"

import {
  RESIZED_IMG_WIDTH,
  BW,
  RESIZED_IMG_WIDTH_OUT_LARGE,
  RESIZED_IMG_WIDTH_OUT_SMALL, 
  IMAGE_IN_DIRECTORY, 
  IMAGE_OUT_BIG_DIRECTORY, 
  IMAGE_OUT_SMALL_DIRECTORY, 
  MOCK_OPTIONS
} from "./options"



export const _sleep = (ms: number) => {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
} 

export const _gaussianKernel = (buffer: any, diameter: number, sigma: number) => {
  const mean = diameter / 2;
  let sum_tmp = 0;
  for (let x = 0; x < diameter; x++) {
      for (let y = 0; y < diameter; y++) {
          const val = Math.exp(-0.5 * (Math.pow(x - mean, 2) + Math.pow(y - mean, 2)) / Math.pow(sigma, 2));
          buffer[x][y] = val;
          sum_tmp += val;
      }
  }
  // Normalize;
  for (let x = 0; x < diameter; x++) {
      for (let y = 0; y < diameter; y++) {
          buffer[x][y] /= sum_tmp;
      }
  }
}

export const _getDelayJitter = (computationType: string) => {

  const {
    DELAY_JITTER_ASYNC,
    DELAY_JITTER_SYNC,
    DELAY_JITTER_NATIVE
  } = MOCK_OPTIONS

  switch(computationType) {
    case "sync": {
      return DELAY_JITTER_SYNC
    }
    case "async": {
      return DELAY_JITTER_ASYNC
    }
    case "cuda-native": {
      return DELAY_JITTER_NATIVE
    }
    case "race-sync": {
      return DELAY_JITTER_SYNC
    }
    case "race-async": {
      return DELAY_JITTER_ASYNC
    }
    case "race-cuda-native": {
      return DELAY_JITTER_NATIVE
    }
  }

}

export async function loadImage(imgName: string | number, resizeWidth = RESIZED_IMG_WIDTH, resizeHeight = RESIZED_IMG_WIDTH, fileFormat=".jpg") {
  const imagePath = `${IMAGE_IN_DIRECTORY}/${imgName}${fileFormat}`
  console.log(imagePath)
  const image = await cv.imreadAsync(imagePath, BW ? cv.IMREAD_GRAYSCALE : cv.IMREAD_COLOR)
  return image.resize(resizeWidth, resizeWidth);
}

export async function storeImageInner(img: cv.Mat, imgName: string | number, resolution: number, kind: string, imgFormat: string = ".jpg") {
  const imgResized = img.resize(resolution, resolution);
  const buffer = await cv.imencodeAsync('.jpg', imgResized, [cv.IMWRITE_JPEG_QUALITY, 80])
  const writeDirectory = kind === "full_res" ? IMAGE_OUT_BIG_DIRECTORY : IMAGE_OUT_SMALL_DIRECTORY
  fs.writeFileSync(`${writeDirectory}/${imgName}${imgFormat}`, buffer);
}

// Store the output of the image processing into 2 images,
// with low and high resolution;
export async function storeImage(img: cv.Mat, imgName: string | number, resizedImageWidthLarge = RESIZED_IMG_WIDTH_OUT_LARGE, resizedImageWidthSmall=RESIZED_IMG_WIDTH_OUT_SMALL) {
  storeImageInner(img, imgName, resizedImageWidthLarge, "full_res");
  storeImageInner(img, imgName, resizedImageWidthSmall, "thumb");
}

export function _intervalToMs(start: number, end: number) {
  return (end - start) / 1e6;
}