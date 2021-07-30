// Use Java System to measure time;
const System = Java.type("java.lang.System");
// Load OpenCV;
const cv = require('./opencv.js');
// Create Jimp context to load images;
const Jimp = require('jimp');



// Convert images to black and white;
const BW = true;
// Edge width (in pixel) of input images.
// If a loaded image has lower width than this, it is rescaled;
const RESIDED_IMG_WIDTH = 4000;
// Edge width (in pixel) of output images.
// We store processed images in 2 variants: small and large;
const RESIDED_IMG_WIDTH_OUT_SMALL = 40;
const RESIDED_IMG_WIDTH_OUT_LARGE = RESIDED_IMG_WIDTH;

// Load and preprocess an image, return it as a matrix;
async function loadImage(imgName) {
    return Jimp.read("img_in/" + imgName + ".jpg")
        .then(img => {
            return cv.matFromImageData(img.bitmap);
            // // Resize input;
            // img = img.resize(RESIDED_IMG_WIDTH_OUT_SMALL, RESIDED_IMG_WIDTH_OUT_SMALL);
            // // Convert to B&W if necessary;
            // return BW ? img.greyscale() : img;
        });
}

// Main processing of the image;
async function processImage(img) {
    // console.log(img.rows);
    // console.log(img.cols);
    // console.log(img);
    // for (let i = 0; i < img.rows; i++) {
    //     console.log(img.data32F[i]);
    // }
    return img;
}

// Store the output of the image processing into 2 images,
// with low and high resolution;
async function storeImage(img, imgName) {
    // Create a Jimp image from the matrix;
    const out = new Jimp({
        width: img.cols,
        height: img.rows,
        data: Buffer.from(img.data)
    })
    out.resize(RESIDED_IMG_WIDTH_OUT_LARGE, RESIDED_IMG_WIDTH_OUT_LARGE).write("img_out/" + imgName + ".jpeg");
    out.resize(RESIDED_IMG_WIDTH_OUT_SMALL, RESIDED_IMG_WIDTH_OUT_SMALL).write("img_out/" + imgName + "_small.jpeg");
    // Clean the OpenCV buffer;
    img.delete();
}

// Main function, it loads an image, process it with our pipeline, writes it to a file;
async function imagePipeline(imgName) {
    try {
        const start = System.nanoTime();
        const img = await loadImage(imgName);
        const endLoad = System.nanoTime();
        const imgProcessed = await processImage(img);
        const endProcess = System.nanoTime();
        storeImage(imgProcessed, imgName);
        const endStore = System.nanoTime();
        console.log("- total time=" + ((endStore - start) / 10e6) + ", load=" + ((endLoad - start) / 10e6) + ", processing=" + ((endProcess - endLoad) / 10e6) + ", store=" + ((endStore - endProcess) / 10e6));
    } catch (err) {
        console.error(err);
    }
}
// console.log(Java.type('java.lang.Math').PI);
// This will be some kind of server endpoint;
for (let i = 0; i < 20; i++) {
    imagePipeline("lena");
}

