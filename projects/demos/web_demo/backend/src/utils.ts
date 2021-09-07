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
  MOCK_OPTIONS,
  CDEPTH,
  FACTOR
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
  const image = await cv.imreadAsync(imagePath, BW ? cv.IMREAD_GRAYSCALE : cv.IMREAD_COLOR)
  //return image.resize(resizeWidth, resizeWidth);
  return image
}

export async function storeImageInner(img: cv.Mat, imgName: string | number, resolution: number, kind: string, imgFormat: string = ".jpg", blackAndWhite: boolean = BW) {
  const imgResized = img.resize(resolution, resolution);
  const buffer = await cv.imencodeAsync('.jpg', imgResized, [cv.IMWRITE_JPEG_QUALITY, 80])
  const writeDirectory = kind === "full_res" ? IMAGE_OUT_BIG_DIRECTORY : IMAGE_OUT_SMALL_DIRECTORY
  fs.writeFileSync(`${writeDirectory}/${imgName}${imgFormat}`, buffer);
}

// Store the output of the image processing into 2 images,
// with low and high resolution;
export async function storeImage(img: cv.Mat, imgName: string | number, resizedImageWidthLarge = RESIZED_IMG_WIDTH_OUT_LARGE, resizedImageWidthSmall=RESIZED_IMG_WIDTH_OUT_SMALL, blackAndWhite: boolean = BW) {
  storeImageInner(img, imgName, resizedImageWidthLarge, "full_res", ".jpg", blackAndWhite);
  storeImageInner(img, imgName, resizedImageWidthSmall, "thumb", ".jpg", blackAndWhite);
}

export function _intervalToMs(start: number, end: number) {
  return (end - start) / 1e6;
}

function lut_r(lut: any) {
  for (let i = 0; i < CDEPTH; i++) {
      let x = i / CDEPTH;
      if (i < CDEPTH / 2) {
          lut[i] = Math.min(CDEPTH - 1, 255 * (0.8 * (1 / (1 + Math.exp(-x + 0.5) * 7 * FACTOR)) + 0.2)) >> 0;
      } else {
          lut[i] = Math.min(CDEPTH - 1, 255 * (1 / (1 + Math.exp((-x + 0.5) * 7 * FACTOR)))) >> 0;
      }
  }
}

function lut_g(lut: any) {
  for (let i = 0; i < CDEPTH; i++) {
      let x = i / CDEPTH;
      let y = 0;
      if (i < CDEPTH / 2) {
          y = 0.8 * (1 / (1 + Math.exp(-x + 0.5) * 10 * FACTOR)) + 0.2;
      } else {
          y = 1 / (1 + Math.exp((-x + 0.5) * 9 * FACTOR));
      }
      lut[i] = Math.min(CDEPTH - 1, 255 * Math.pow(y, 1.4)) >> 0;
  }
}

function lut_b(lut: any) {
  for (let i = 0; i < CDEPTH; i++) {
      let x = i / CDEPTH;
      let y = 0;
      if (i < CDEPTH / 2) {
          y = 0.7 * (1 / (1 + Math.exp(-x + 0.5) * 10 * FACTOR)) + 0.3;
      } else {
          y = 1 / (1 + Math.exp((-x + 0.5) * 10 * FACTOR));
      }
      lut[i] = Math.min(CDEPTH - 1, 255 * Math.pow(y, 1.6)) >> 0;
  }
}

export const copyFrom = (arrayFrom: any, arrayTo: any) => {
  let i = arrayTo.length
  while(i--) arrayTo[i] = arrayFrom[i];
}

export const LUT = [lut_r, lut_g, lut_b]; // VELVIA 50

// export const LUT = [lut_b, lut_r, lut_g]; // FUJIFILM SUPERIA 400

// export const LUT = [ lut_r, lut_b, lut_g]; // BAD

// export const LUT = [ lut_g, lut_r,  lut_b]; 