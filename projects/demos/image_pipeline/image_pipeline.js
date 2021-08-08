// Use Java System to measure time;
const System = Java.type("java.lang.System");
// Load OpenCV;
const cv = require("opencv4nodejs");
// Load function to write to file;
const fs = require("fs");
// Load GrCUDA;
const cu = Polyglot.eval("grcuda", "CU")
// Load CUDA kernels;
const ck = require("./cuda_kernels.js");
const { assert } = require("console");

/////////////////////////////
/////////////////////////////

// Convert images to black and white;
const BW = false;
// Edge width (in pixel) of input images.
// If a loaded image has lower width than this, it is rescaled;
const RESIZED_IMG_WIDTH = 1024;
// Edge width (in pixel) of output images.
// We store processed images in 2 variants: small and large;
const RESIZED_IMG_WIDTH_OUT_SMALL = 40;
const RESIZED_IMG_WIDTH_OUT_LARGE = RESIZED_IMG_WIDTH;

// Build the CUDA kernels;
const GAUSSIAN_BLUR_KERNEL = cu.buildkernel(ck.GAUSSIAN_BLUR, "gaussian_blur", "const pointer, pointer, sint32, sint32, const pointer, sint32")
const SOBEL_KERNEL = cu.buildkernel(ck.SOBEL, "sobel", "pointer, pointer, sint32, sint32")
const EXTEND_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "extend", "pointer, const pointer, const pointer, sint32, sint32")
const MAXIMUM_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "maximum", "const pointer, pointer, sint32")
const MINIMUM_KERNEL = cu.buildkernel(ck.EXTEND_MASK, "minimum", "const pointer, pointer, sint32")
const UNSHARPEN_KERNEL = cu.buildkernel(ck.UNSHARPEN, "unsharpen", "pointer, pointer, pointer, float, sint32")
const COMBINE_KERNEL = cu.buildkernel(ck.COMBINE, "combine", "const pointer, const pointer, const pointer, pointer, sint32")
const RESET_KERNEL = cu.buildkernel(ck.RESET, "reset", "pointer, sint32")

// Constant parameters used in the image processing;
const KERNEL_SMALL_DIAMETER = 3
const KERNEL_SMALL_VARIANCE = 0.1
const KERNEL_LARGE_DIAMETER = 5
const KERNEL_LARGE_VARIANCE = 20
const KERNEL_UNSHARPEN_DIAMETER = 3
const KERNEL_UNSHARPEN_VARIANCE = 5
const UNSHARPEN_AMOUNT = 2
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

async function storeImageInner(img, imgName, resolution, kind) {
    const imgResized = img.resize(resolution, resolution);
    const buffer = await cv.imencodeAsync('.jpg', imgResized, [cv.IMWRITE_JPEG_QUALITY, 40])
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
async function processImage(img, channel) {

    assert(img.rows === img.cols);
    const size = img.rows;

    // Allocate image data;
    const image = cu.DeviceArray("float", size, size);
    const image2 = cu.DeviceArray("float", size, size);
    const image3 = cu.DeviceArray("float", size, size);

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
    // FIXME: use memcpy to speed up the process. We cannot access channels in this way though;
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            if (BW) image[i][j] = img.at(i, j) / 255;
            else image[i][j] = img.atRaw(i, j)[channel] / 255;
        }
    }

    const start = System.nanoTime();

    // Create Gaussian kernels;
    gaussian_kernel(kernel_small, KERNEL_SMALL_DIAMETER, KERNEL_SMALL_VARIANCE);
    gaussian_kernel(kernel_large, KERNEL_LARGE_DIAMETER, KERNEL_LARGE_VARIANCE);
    gaussian_kernel(kernel_unsharpen, KERNEL_UNSHARPEN_DIAMETER, KERNEL_UNSHARPEN_VARIANCE);

    // Main GPU computation;
    // Blur - Small;
    GAUSSIAN_BLUR_KERNEL((BLOCKS, BLOCKS), (THREADS_2D, THREADS_2D), 4 * KERNEL_SMALL_DIAMETER * KERNEL_SMALL_DIAMETER)(
        image, blurred_small, size, size, kernel_small, KERNEL_SMALL_DIAMETER);
    // Blur - Large;
    GAUSSIAN_BLUR_KERNEL((BLOCKS, BLOCKS), (THREADS_2D, THREADS_2D), 4 * KERNEL_LARGE_DIAMETER * KERNEL_LARGE_DIAMETER)(
        image, blurred_large, size, size, kernel_large, KERNEL_LARGE_DIAMETER);
    // Blur - Unsharpen;
    GAUSSIAN_BLUR_KERNEL((BLOCKS, BLOCKS), (THREADS_2D, THREADS_2D), 4 * KERNEL_UNSHARPEN_DIAMETER * KERNEL_UNSHARPEN_DIAMETER)(
        image, blurred_unsharpen, size, size, kernel_unsharpen, KERNEL_UNSHARPEN_DIAMETER);
    // Sobel filter (edge detection);
    SOBEL_KERNEL((BLOCKS, BLOCKS), (THREADS_2D, THREADS_2D))(
        blurred_small, mask_small, size, size);
    SOBEL_KERNEL((BLOCKS, BLOCKS), (THREADS_2D, THREADS_2D))(
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
    COMBINE_KERNEL(BLOCKS * 2, THREADS_1D)(
        image2, blurred_small, mask_small, image3, size * size);

    // Store the image data.
    // FIXME: use memcpy to speed up the process. We cannot access channels in this way though;
    const tmp = image3[0][0];
    const end = System.nanoTime();
    console.log("--cuda time=" + intervalToMs(start, end) + " ms");
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            if (BW) {
                img.set(i, j, image3[i][j] * 255);
            } else {
                const newPixel = img.atRaw(i, j);
                newPixel[channel] = image3[i][j] * 255
                img.set(i, j, newPixel);
            }
        }
    }
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
        const start = System.nanoTime();
        const img = await loadImage(imgName);
        const endLoad = System.nanoTime();
        for (let channel = 0; channel < 3; channel++) {
            await processImage(img);
        }
        const endProcess = System.nanoTime();
        await storeImage(img, imgName + "_" + count)
        const endStore = System.nanoTime();
        console.log("- total time=" + intervalToMs(start, endStore) + ", load=" + intervalToMs(start, endLoad) + ", processing=" + intervalToMs(endLoad, endProcess) + ", store=" + intervalToMs(endProcess, endStore));
    } catch (err) {
        console.error(err);
    }
}

async function main() {
    // This will be some kind of server endpoint;
    for (let i = 0; i < 1; i++) {
        // Use await for serial execution, otherwise it processes multiple images in parallel.
        // Performance looks identical though;
        await imagePipeline("lena", i);
    }
}

/////////////////////////////
/////////////////////////////

// Begin the computation;
main();

