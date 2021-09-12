const System = Java.type("java.lang.System");
const cu = Polyglot.eval('grcuda', 'CU')
const { assert } = require("console");
const cv = require("opencv4nodejs");

function intervalToMs(start, end) {
    return (end - start) / 1e6;
}

function mean(x) {
	return x.reduce((a, b) => (a + b)) / x.length;
}

const R = 512;
const N = R * R;
const NUM_TESTS = 120;
const DEBUG = true;

// Create the device array;
const deviceArray = cu.DeviceArray('int', N);

// Load the image;
const img = cv.imread("img_in/lena.jpg", cv.IMREAD_GRAYSCALE).resize(R, R);
const img_buffer = img.getData();

const direction = process.argv[2];
const dirText = direction == "to" ? "grcuda->img" : "img->grcuda";

// Initialize device array;
for (let i = 0; i < N; i++) {
	deviceArray[i] = 1;
}

//////////////////////////
// COPY TO DEVICE ARRAY //
//////////////////////////

// Copy using for loop;
function copy_for(from, to) {
	const start = System.nanoTime();
	for (let i = 0; i < N; i++) {
		to[i] = from[i];
	}
	const end = System.nanoTime();
	const time = intervalToMs(start, end);
	if (DEBUG) console.log("-- copy "+ dirText + "- forloop=" + time + " ms")
	return time
}

// Copy using copyFrom;
function copy_grcuda_from(from, to) {
	const start = System.nanoTime();
	to.copyFrom(from)
	const end = System.nanoTime();
	const time = intervalToMs(start, end);
	if (DEBUG) console.log("-- copy "+ dirText + "- grcuda=" + time + " ms")
	return time
}
function copy_grcuda_to(from, to) {
	const start = System.nanoTime();
	from.copyFrom(to)
	const end = System.nanoTime();
	const time = intervalToMs(start, end);
	if (DEBUG) console.log("-- copy "+ dirText + "- grcuda=" + time + " ms")
	return time
}

// Copy using while;
function copy_while(from, to) {
	const start = System.nanoTime();
	let i = from.length;
	while(i--) to[i] = from[i];
	const end = System.nanoTime();
	const time = intervalToMs(start, end);
	if (DEBUG) console.log("-- copy "+ dirText + "- while=" + time + " ms")
	return time
}

// Copy using map;
function copy_map(from, to) {
	if (direction == "to") return undefined;
	const start = System.nanoTime();
	let i = 0;
	from.forEach(a => {
		to[i++] = a;
	});
	const end = System.nanoTime();
	const time = intervalToMs(start, end);
	if (DEBUG) console.log("-- copy "+ dirText + "- map=" + time + " ms")
	return time
}

let from = img_buffer;
let to = deviceArray;
let copy_grcuda = copy_grcuda_from
if (direction == "to") {
	from = deviceArray;
	to = img_buffer;
	copy_grcuda = copy_grcuda_to
}
const types = ["for", "grcuda", "while", "map"];
const functions = [copy_for, copy_grcuda, copy_while, copy_map];
const averageTimes = [];

// Test all functions
for (let t = 0; t < types.length; t++) {
	times = []
	for (let n = 0; n < NUM_TESTS; n++) {
		times.push(functions[t](from, to));
	}
	const averageTime = mean(times);
	if (DEBUG) console.log(types[t] + "=" + averageTime + " ms");
	averageTimes.push(averageTime);

	// Check that the output is correct;
	for (let n = 0; n < N; n++) {
		assert(from[n] == to[n]);
	}
}

console.log("---- RESULTS ----")
for (let t = 0; t < types.length; t++) {
	console.log("--" + types[t] + "=" + averageTimes[t] + " ms");
}

///////////////////////////////
// OLD, WITHOUT IMAGE BUFFER //
///////////////////////////////

// const y = [] 
// for (let i = 0; i < N; i++) {
// 	y[i] = i;
// }

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