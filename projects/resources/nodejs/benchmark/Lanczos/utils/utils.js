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


module.exports = {
    fillWith,
    arrayNormalize,
    arrayMemset,
    copy,
    take,
    launchKernel,
    createMatrix
}