/**
 * Fills an array, elementwise, with a given function.
 * Similar to Array.fill, but the fillFn gets called every time.
 * @param array
 * @param fillFn
 */
const fillWith = (array, fillFn) => {
    for (let i = 0; i < array.length; ++i) {
        array[i] = fillFn()
    }
}

const fillWithConst = (array, fill) => {
    for (let i = 0; i < array.length; ++i) {
        array[i] = fill
    }
}

const printArray = (array) => {
    for (let i = 0; i < array.length; ++i) {
        process.stdout.write(array[i] + " ")
    }
    console.log("array lenght: " + array.length)
}

const dotProduct = (v1, v2) => {
    let acc = 0.0;

    for (let i = 0; i < v1.length; ++i) {
        acc += v1[i] * v2[i]
    }
    return acc
}

/**
 * Takes the i-th element of every subarray in the input array
 * @param arrays
 * @param i
 * @returns {*}
 */
const take = (arrays, i) => {
    return arrays.map(array => array[i])
}


/**
 * Normalize the input array
 * @param array
 */
const arrayNormalize = (array) => {

    let accumulator = 0.0
    for (let i = 0; i < array.length; ++i) {
        let tmpValue = array[i]
        accumulator += tmpValue * tmpValue
    }

    accumulator = Math.sqrt(accumulator)

    for (let i = 0; i < array.length; ++i) {
        array[i] /= accumulator
    }

    return array
}


/**
 * Set every element of the input array to a specific value
 * @param array
 * @param length
 * @param value
 */
const arrayMemset = (array, length, value) => {
    for (let i = 0; i < length; ++i) {
        array[i] = value
    }
}


/**
 * Copies an array into another.
 * The function written as follows is faster than the
 * `DeviceArray.copyFrom` or `DeviceArray.copyTo` function in GrCUDA
 * @param from
 * @param to
 */
const copy = (from, to) => {

    let length = from.length
    while (length--) {
        to[length] = from[length]
    }

}

/**
 * Abuses javascript spread operator to overcome the fact that cu.buildkernel returns an HOF...
 * @param kernel
 * @param kernelConfig
 * @param kernelArguments
 */
const launchKernel = (kernel, kernelConfig, kernelArguments) => {
    kernel(...kernelConfig)(...kernelArguments)
}


/**
 * Creates a zero-filled matrix with the given dimensions
 * @param {number} dimensionX
 * @param {number} dimensionY
 * @returns {*[*[]]} a matrix
 */
const createMatrix = (dimensionX, dimensionY) => {

    let matrix = []
    for (let i = 0; i < dimensionX; ++i) {
        matrix.push(new Array(dimensionY).fill(0))
    }

    return matrix
}

matmul = (matrix, vector) => {
    return matrix.map(row => {
        return row.reduce((acc, cur, idx) => acc + cur * vector[idx])
    })
}

const norm = (arr) => {
    let acc = 0; 
    for(let i = 0; i < arr.length; ++i){
        acc += arr[i] * arr[i]
    }
    return Math.sqrt(acc) 
}


const spmvCoo = (cooMatrix, vIn, N, nnz) => {
    const {
        x, y, val
    } = cooMatrix
    // console.log("BEGIN -> cooMatrix")
    // console.log(cooMatrix)
    // console.log("END   -> cooMatrix")

    // console.log("BEGIN -> cooMatrix.x")
    // console.log(cooMatrix.x)
    // console.log("END   -> cooMatrix.x")

    // console.log("BEGIN -> cooMatrix.y")
    // console.log(cooMatrix.y)
    // console.log("END   -> cooMatrix.y")

    // console.log("BEGIN -> cooMatrix.val")
    // console.log(cooMatrix.val)
    // console.log("END   -> cooMatrix.val")


    vOut = new Array(N).fill(0)
    for (let i = 0; i < nnz; i++) {
        vOut[y[i]] += vIn[x[i]] * val[i];
    }
    return vOut
}

module.exports = {
    fillWith,
    fillWithConst,
    arrayNormalize,
    arrayMemset,
    copy,
    take,
    launchKernel,
    createMatrix, 
    dotProduct, 
    matmul, 
    spmvCoo,
    printArray
}