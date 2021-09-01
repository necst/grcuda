// Use Java System to measure time;
const System = Java.type("java.lang.System");
// Load OpenCV;
const cv = require("opencv4nodejs");
// Load function to write to file;
const fs = require("fs");
// Load GrCUDA;
const cu = Polyglot.eval("grcuda", "CU");
// Load CUDA kernels;
const ck = require("./cuda_kernels.js");

/////////////////////////////
/////////////////////////////

// Convert images to black and white;
const BW = false;
// Edge width (in pixel) of input images.
// If a loaded image has lower width than this, it is rescaled;
const RESIZED_IMG_WIDTH = 512;
// Edge width (in pixel) of output images.
// We store processed images in 2 variants: small and large;
const RESIZED_IMG_WIDTH_OUT_SMALL = 40;
const RESIZED_IMG_WIDTH_OUT_LARGE = RESIZED_IMG_WIDTH;

// Build the CUDA kernels;
const GAUSSIAN_BLUR_KERNEL = cu.buildkernel(ck.GAUSSIAN_BLUR, "gaussian_blur", "const pointer, pointer, sint32, sint32, const pointer, sint32");
const SOBEL_KERNEL = cu.buildkernel(ck.SOBEL, "sobel", "pointer, pointer, sint32, sint32");
const EXTEND_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "extend", "pointer, const pointer, const pointer, sint32, sint32");
const MAXIMUM_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "maximum", "const pointer, pointer, sint32");
const MINIMUM_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "minimum", "const pointer, pointer, sint32");
const UNSHARPEN_KERNEL = cu.buildkernel(ck.UNSHARPEN, "unsharpen", "const pointer, const pointer, pointer, float, sint32");
const COMBINE_KERNEL = cu.buildkernel(ck.COMBINE, "combine", "const pointer, const pointer, const pointer, pointer, sint32");
const COMBINE_KERNEL_LUT = cu.buildkernel(ck.COMBINE_2, "combine_lut", "const pointer, const pointer, const pointer, pointer, sint32, pointer");

// Constant parameters used in the image processing;
const KERNEL_SMALL_DIAMETER = 3;
const KERNEL_SMALL_VARIANCE = 0.1;
const KERNEL_LARGE_DIAMETER = 7;
const KERNEL_LARGE_VARIANCE = 20;
const KERNEL_UNSHARPEN_DIAMETER = 3;
const KERNEL_UNSHARPEN_VARIANCE = 5;
const UNSHARPEN_AMOUNT = 30;
const CDEPTH = 256;
const FACTOR = 0.8
// CUDA parameters;
const BLOCKS = 6;
const THREADS_1D = 32;
const THREADS_2D = 8;

/////////////////////////////
// Utility functions ////////
/////////////////////////////

function intervalToMs(start, end) {
    return (end - start) / 1e6;
}

function gaussian_kernel(buffer, diameter, sigma) {
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

function lut_r(lut) {
    for (let i = 0; i < CDEPTH; i++) {
        x = i / CDEPTH;
        if (i < CDEPTH / 2) {
            lut[i] = Math.min(CDEPTH - 1, 255 * (0.8 * (1 / (1 + Math.exp(-x + 0.5) * 7 * FACTOR)) + 0.2)) >> 0;
        } else {
            lut[i] = Math.min(CDEPTH - 1, 255 * (1 / (1 + Math.exp((-x + 0.5) * 7 * FACTOR)))) >> 0;
        }
    }
}

function lut_g(lut) {
    for (let i = 0; i < CDEPTH; i++) {
        x = i / CDEPTH;
        y = 0;
        if (i < CDEPTH / 2) {
            y = 0.8 * (1 / (1 + Math.exp(-x + 0.5) * 10 * FACTOR)) + 0.2;
        } else {
            y = 1 / (1 + Math.exp((-x + 0.5) * 9 * FACTOR));
        }
        lut[i] = Math.min(CDEPTH - 1, 255 * Math.pow(y, 1.4)) >> 0;
    }
}

function lut_b(lut) {
    for (let i = 0; i < CDEPTH; i++) {
        x = i / CDEPTH;
        y = 0;
        if (i < CDEPTH / 2) {
            y = 0.7 * (1 / (1 + Math.exp(-x + 0.5) * 10 * FACTOR)) + 0.3;
        } else {
            y = 1 / (1 + Math.exp((-x + 0.5) * 10 * FACTOR));
        }
        lut[i] = Math.min(CDEPTH - 1, 255 * Math.pow(y, 1.6)) >> 0;
    }
}

const LUT = [lut_r, lut_g, lut_b];

async function storeImageInner(img, imgName, resolution, kind) {
    const imgResized = img.resize(resolution, resolution);
    const buffer = await cv.imencodeAsync('.jpg', imgResized, [cv.IMWRITE_JPEG_QUALITY, 80])
    fs.writeFileSync("img_out/" + imgName + "_" + kind + ".jpg", buffer);
}

/////////////////////////////
// Main computations ////////
/////////////////////////////

// Load and preprocess an image, return it as a matrix;
async function loadImage(imgName) {
    return cv.imreadAsync("img_in/" + imgName + ".jpg", BW ? cv.IMREAD_GRAYSCALE : cv.IMREAD_COLOR)
        .then(img => {
            // Resize input;
            return img.resize(RESIZED_IMG_WIDTH, RESIZED_IMG_WIDTH);
        });
}

// Main processing of the image;
async function processImage(img, size, channel) {
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

    const lut = cu.DeviceArray("int", CDEPTH);  

    // Initialize the right LUT;
    LUT[channel](lut);

    // Fill the image data;
    const s1 = System.nanoTime();
    image.copyFrom(img, size * size);
    const e1 = System.nanoTime();
    console.log("--img to device array=" + intervalToMs(s1, e1) + " ms");

    const start = System.nanoTime();

    // Create Gaussian kernels;
    gaussian_kernel(kernel_small, KERNEL_SMALL_DIAMETER, KERNEL_SMALL_VARIANCE);
    gaussian_kernel(kernel_large, KERNEL_LARGE_DIAMETER, KERNEL_LARGE_VARIANCE);
    gaussian_kernel(kernel_unsharpen, KERNEL_UNSHARPEN_DIAMETER, KERNEL_UNSHARPEN_VARIANCE);

    // Main GPU computation;
    // Blur - Small;
    GAUSSIAN_BLUR_KERNEL([BLOCKS, BLOCKS], [THREADS_2D, THREADS_2D], 4 * KERNEL_SMALL_DIAMETER * KERNEL_SMALL_DIAMETER)(
        image, blurred_small, size, size, kernel_small, KERNEL_SMALL_DIAMETER);
    // Blur - Large;
    GAUSSIAN_BLUR_KERNEL([BLOCKS, BLOCKS], [THREADS_2D, THREADS_2D], 4 * KERNEL_LARGE_DIAMETER * KERNEL_LARGE_DIAMETER)(
        image, blurred_large, size, size, kernel_large, KERNEL_LARGE_DIAMETER);
    // Blur - Unsharpen;
    GAUSSIAN_BLUR_KERNEL([BLOCKS, BLOCKS], [THREADS_2D, THREADS_2D], 4 * KERNEL_UNSHARPEN_DIAMETER * KERNEL_UNSHARPEN_DIAMETER)(
        image, blurred_unsharpen, size, size, kernel_unsharpen, KERNEL_UNSHARPEN_DIAMETER);
    // Sobel filter (edge detection);
    SOBEL_KERNEL([BLOCKS, BLOCKS], [THREADS_2D, THREADS_2D])(
        blurred_small, mask_small, size, size);
    SOBEL_KERNEL([BLOCKS, BLOCKS], [THREADS_2D, THREADS_2D])(
        blurred_large, mask_large, size, size);
    // Ensure that the output of Sobel is in [0, 1];
    MAXIMUM_KERNEL(BLOCKS * 2, THREADS_1D)(mask_small, maximum_1, size * size);
    MINIMUM_KERNEL(BLOCKS * 2, THREADS_1D)(mask_small, minimum_1, size * size);
    EXTEND_KERNEL(BLOCKS * 2, THREADS_1D)(mask_small, minimum_1, maximum_1, size * size, 1);
    // Extend large edge detection mask, and normalize it;
    MAXIMUM_KERNEL(BLOCKS * 2, THREADS_1D)(mask_large, maximum_2, size * size);
    MINIMUM_KERNEL(BLOCKS * 2, THREADS_1D)(mask_large, minimum_2, size * size);
    EXTEND_KERNEL(BLOCKS * 2, THREADS_1D)(mask_large, minimum_2, maximum_2, size * size, 5);
    // Unsharpen;
    UNSHARPEN_KERNEL(BLOCKS * 2, THREADS_1D)(
        image, blurred_unsharpen, image_unsharpen, UNSHARPEN_AMOUNT, size * size);
    // Combine results;
    COMBINE_KERNEL(BLOCKS * 2, THREADS_1D)(
        image_unsharpen, blurred_large, mask_large, image2, size * size);
    COMBINE_KERNEL_LUT(BLOCKS * 2, THREADS_1D)(
        image2, blurred_small, mask_small, image3, size * size, lut);

    // Store the image data.
    const tmp = image3[0][0];
    const end = System.nanoTime();
    console.log("--cuda time=" + intervalToMs(start, end) + " ms");
    const s2 = System.nanoTime();
    image3.copyTo(img, size * size);
    const e2 = System.nanoTime();
    console.log("--device array to image=" + intervalToMs(s2, e2) + " ms");
    return img;
}

async function processImageBW(img) {
    return new cv.Mat(Buffer.from(await processImage(img.getData(), img.rows, 0)), img.rows, img.cols, cv.CV_8UC1);
}

async function processImageColor(img) {
    // Possibly not the most efficient way to do this,
    // we should process the 3 channels concurrently, and avoid creation of temporary cv.Mat;
    const channels = img.splitChannels();
    for (let c = 0; c < 3; c++) {
        channels[c] = new cv.Mat(Buffer.from(await processImage(channels[c].getData(), img.rows, c)), img.rows, img.cols, cv.CV_8UC1);;
    }
    return new cv.Mat(channels);  
}

// Store the output of the image processing into 2 images,
// with low and high resolution;
async function storeImage(img, imgName) {
    storeImageInner(img, imgName, RESIZED_IMG_WIDTH_OUT_LARGE, "large");
    storeImageInner(img, imgName, RESIZED_IMG_WIDTH_OUT_SMALL, "small");
}

// Main function, it loads an image, process it with our pipeline, writes it to a file;
async function imagePipeline(imgName, count) {
    try {
        // Load image;
        const start = System.nanoTime();
        let img = await loadImage(imgName);
        const endLoad = System.nanoTime();
        // Process image;
        if (BW) img = await processImageBW(img);
        else img = await processImageColor(img);
        const endProcess = System.nanoTime();
        // Store image;
        await storeImage(img, imgName + "_" + count)
        const endStore = System.nanoTime();
        console.log("- total time=" + intervalToMs(start, endStore) + ", load=" + intervalToMs(start, endLoad) + ", processing=" + intervalToMs(endLoad, endProcess) + ", store=" + intervalToMs(endProcess, endStore));
    } catch (err) {
        console.error(err);
    }
}

async function main() {
    // This will be some kind of server endpoint;
    for (let i = 0; i < 20; i++) {
        // Use await for serial execution, otherwise it processes multiple images in parallel.
        // Performance looks identical though;
        await imagePipeline("astro1", i);
    }
}

/////////////////////////////
/////////////////////////////

// Begin the computation;
main();

