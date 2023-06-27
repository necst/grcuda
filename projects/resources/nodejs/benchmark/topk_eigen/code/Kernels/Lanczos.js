const {
    fillWith,
    arrayNormalize,
    launchKernel,
    take,
    dotProduct,
    matmul,
    spmvCoo
} = require("../utils/utils.js")
//console.log = () => {}
const kernelsDef = require("../utils/kernels.js");

const {
    arrayMemset,
    copy
} = require("../utils/utils");

const {
    BLOCKS, THREADS_PER_BLOCK, DEFAULT_KERNEL_CONFIG
} = require("../utils/options")

Math.arrayMax = (array) => array.reduce((cur, next) => Math.max(cur, next))

class Lanczos {

    cooMatrix = {}
    N = 0
    M = 0
    nnz = 0
    cu = undefined
    kernels = {}
    reorthogonalize = undefined
    numPartitions = 0
    matrixPartitions = []
    matrixName = ""
    iteration = 0

    /**
     * Creates an object that handles the whole computation
     * @param cooMatrix Input matrix, in sparse coordinate format
     * @param cu Instance of GrCUDA
     * @param {number} numPartitions number of (balanced) partitions
     * @param {boolean} reorthogonalize whether to reorthogonalize the lanczos vectors or not
     */
    constructor(cooMatrix, numEigen, cu, numPartitions, reorthogonalize, matrixName, iteration, types = { input: "float", middle: "float", output: "float" }) {
        this.cooMatrix = cooMatrix
        this.N = Math.arrayMax(cooMatrix.x)
        this.M = Math.arrayMax(cooMatrix.y)
        this.nnz = cooMatrix.val.length
        this.cu = cu
        this.numEigen = numEigen
        this.reorthogonalize = reorthogonalize
        this.numPartitions = numPartitions
        this.matrixName = matrixName
        this.iteration = iteration

        this.inputType = (types.input == "__half" || types.input == "__nv_bfloat16") ? "float" : types.input
        this.middleType = types.middle
        this.outputType = types.output

        this.arrayInputType = types.input

        //console.log(this.inputType, this.outputType)

    }


    /**
     * Private method to build _somewhat_ balanced partitions wrt the number of nonzeros
     * After determining the optimal partition size, it advances the iterator until the reference y index does change
     * This is done in order to not interrupt the current partition at a given y index
     * @private
     */
    _createPartitions = () => {
        const nnzPerPartition = ((this.nnz + this.numPartitions) / this.numPartitions) >> 0
        let fromIndex = 0
        let toIndex = nnzPerPartition
        let indexValue = this.cooMatrix.y[toIndex]

        for (let i = 0; i < this.numPartitions - 1; ++i) {
            // advance the next index until the values differ
            while (indexValue === this.cooMatrix.y[toIndex]) {
                toIndex++
            }

            // get the y value that corresponds to that index
            const offset = (fromIndex === 0) ? fromIndex : (this.cooMatrix.y[fromIndex] - 1)

            let cooPartition = this._assignPartition(fromIndex, toIndex, offset)
            this.matrixPartitions.push(cooPartition)

            fromIndex = toIndex
            toIndex += nnzPerPartition
            indexValue = this.cooMatrix.y[toIndex]
        }

        // Last partition gets everything else
        const offset = this.cooMatrix.y[fromIndex]
        let cooPartition = this._assignPartition(fromIndex, this.nnz, offset)
        this.matrixPartitions.push(cooPartition)

    }

    /**
     *
     * @param {number} fromIndex begin nnz index
     * @param {number} toIndex end nnz index
     * @param {number} offset offset by which to displace the y vector of a given partition
     * @returns {{val: *[], size: number, x: *[], y: *[], end: number, begin: number, N: number}} the partition
     * @private
     */
    _assignPartition(fromIndex, toIndex, offset) {
        // Now, the partition from lastIndex to nextIndex contains the somewhat balanced partition
        let cooPartition = { x: [], y: [], val: [], size: -1, begin: -1, end: -1, N: -1 }
        cooPartition.size = toIndex - fromIndex
        cooPartition.begin = fromIndex
        cooPartition.end = toIndex
        cooPartition.x = []//new Array(cooPartition.size).fill(0)
        cooPartition.y = []//new Array(cooPartition.size).fill(0)
        cooPartition.val = []//new Array(cooPartition.size).fill(0)


        for (let i = fromIndex; i < toIndex; ++i) {
            cooPartition.x.push(this.cooMatrix.x[i])
            cooPartition.y.push(this.cooMatrix.y[i] - offset)
            cooPartition.val.push(this.cooMatrix.val[i])
        }

        cooPartition.N = cooPartition.y[cooPartition.y.length - 1]

        return cooPartition
    }

    /**
     * Builds the input, intermediate and output vectors on the host side
     * @returns {Lanczos} the instance itself for chaining
     */
    build = () => {

        this._createPartitions()

        this.vecsIn = []
        this.vecsOutSpmv = []

        this.matrixPartitions.forEach(partition => {
            this.vecsIn.push(new Array(this.N).fill(0))
            this.vecsOutSpmv.push(new Array(partition.N).fill(0))
        })


        fillWith(this.vecsIn[0], Math.random)
        arrayNormalize(this.vecsIn[0])
        for (let i = 1; i < this.numPartitions; ++i) {
            copy(this.vecsIn[0], this.vecsIn[i])
        }

        return this
    }


    /**
     * Creates and transfers the host buffers to GPU memory
     * @returns {Lanczos} the instance itself for chaining
     */
    transferToGPU = () => {

        this._createBuffers();

        this.matrixPartitions.forEach((partition, i) => {
            copy(this.vecsIn[i], this.deviceVecIn[i])
            //copy(this.vecsOutSpmv[i], this.deviceVecOutSpmv[i])
            copy(partition.x, this.deviceCooX[i])
            copy(partition.y, this.deviceCooY[i])
            copy(partition.val, this.deviceCooVal[i])
            arrayMemset(this.deviceIntermediateDotProductValues[i], BLOCKS, 0.0)
        })

        return this
    }

    /**
     * Creates the buffer on GPU memory.
     * @private
     */
    _createBuffers = () => {
        console.log("buffers")
        this.deviceVecIn = this.matrixPartitions.map(unused => this.cu.DeviceArray(this.inputType, this.N))
        this.deviceVecInLP = this.matrixPartitions.map(unused => this.cu.DeviceArray(this.inputType, Math.round(this.N / 2)))
        this.deviceIntermediateDotProductValues = this.matrixPartitions.map(unused => this.cu.DeviceArray(this.outputType, BLOCKS))
        this.alphaIntermediate = this.matrixPartitions.map(unused => this.cu.DeviceArray(this.outputType, 1))
        this.alphaIntermediateHP = this.matrixPartitions.map(unused => this.cu.DeviceArray(this.outputType, 1))
        this.betaIntermediate = this.matrixPartitions.map(unused => this.cu.DeviceArray(this.outputType, 1))
        this.deviceVecOutSpmv = this.matrixPartitions.map(partition => this.cu.DeviceArray(this.inputType, (this.arrayInputType != this.inputType) ? (Math.round(partition.N / 2) + 1) : partition.N))
        this.deviceCooX = this.matrixPartitions.map(partition => this.cu.DeviceArray("int", partition.size))
        this.deviceCooY = this.matrixPartitions.map(partition => this.cu.DeviceArray("int", partition.size))
        this.deviceCooVal = this.matrixPartitions.map(partition => this.cu.DeviceArray(this.inputType, partition.size))
        this.deviceCooValLP = this.matrixPartitions.map(partition => this.cu.DeviceArray(this.inputType, Math.round(partition.size / 2) + 1))
        this.deviceVecNext = this.matrixPartitions.map(partition => this.cu.DeviceArray(this.outputType, partition.N))
        this.deviceNormalizeOut = this.matrixPartitions.map(partition => this.cu.DeviceArray(this.outputType, partition.N))
        this.deviceLanczosVectors = this.matrixPartitions.map(partition => this.cu.DeviceArray(this.outputType, partition.N * this.numEigen))
        console.log(" endbuffers")

        this.alpha = 0.0 //this.cu.DeviceArray(this.outputType, 1)
        this.beta = 0.0 //this.cu.DeviceArray(this.outputType, 1)
    }

    /**
     * @unused
     * @returns {Lanczos}
     */
    transferFromGPU = () => {
        return this
    }


    /**
     * Creates the "1D" kernels (i.e. that operate on a single partition)
     * @private
     */
    _buildKernels = () => {
        console.log("buildkernels")
        const SPMV = this.cu.buildkernel(kernelsDef.SPMV(this.inputType, this.middleType, this.inputType), "spmv", "const pointer, const pointer, const pointer, const pointer, pointer, const sint32")
        const DOT_PRODUCT = this.cu.buildkernel(kernelsDef.DOT_PRODUCT(this.inputType, this.middleType, this.outputType), "dot_product", "const pointer, const pointer, pointer, const sint32, const sint32")
        const DOT_PRODUCT_HP = this.cu.buildkernel(kernelsDef.DOT_PRODUCT(this.outputType, this.outputType, this.outputType), "dot_product", "const pointer, const pointer, pointer, const sint32, const sint32")
        const L2_NORM = this.cu.buildkernel(kernelsDef.L2_NORM(this.outputType, this.middleType, this.outputType), "l2_norm", "const pointer, pointer, const sint32, const sint32")
        const AXPB_XTENDED = this.cu.buildkernel(kernelsDef.AXPB_XTENDED(this.inputType, this.middleType, this.outputType), "axpb_xtended", `const ${this.inputType}, const pointer, const pointer, const ${this.inputType}, const pointer, pointer, const sint32, const sint32, const sint32`)
        const NORMALIZE = this.cu.buildkernel(kernelsDef.NORMALIZE(this.outputType, this.middleType, this.outputType), "normalize", `const pointer, const ${this.outputType}, pointer, const sint32`)
        const SUBTRACT = this.cu.buildkernel(kernelsDef.SUBTRACT(this.outputType, this.outputType, this.outputType), "subtract", "pointer, const pointer, float, const sint32, const sint32")
        const COPY_PARTITION_TO_VEC_FW = this.cu.buildkernel(kernelsDef.COPY_PARTITION_TO_VEC(this.inputType, this.middleType, this.outputType), "copy_partition_to_vec", "const pointer, pointer, const sint32, const sint32, const sint32")
        const COPY_PARTITION_TO_VEC_BW = this.cu.buildkernel(kernelsDef.COPY_PARTITION_TO_VEC(this.outputType, this.middleType, this.inputType), "copy_partition_to_vec", "const pointer, pointer, const sint32, const sint32, const sint32")
        const CAST_LP_HP = this.cu.buildkernel(kernelsDef.CAST(this.arrayInputType, this.inputType), "device_cast" ,"const pointer, pointer, const sint32")
        const CAST_HP_LP = this.cu.buildkernel(kernelsDef.CAST(this.inputType, this.arrayInputType), "device_cast" ,"const pointer, pointer, const sint32")

        console.log("finished buildkernels")
        this.kernels = {
            SPMV,
            AXPB_XTENDED,
            NORMALIZE,
            DOT_PRODUCT,
            DOT_PRODUCT_HP,
            L2_NORM,
            SUBTRACT,
            COPY_PARTITION_TO_VEC_FW,
            COPY_PARTITION_TO_VEC_BW,
            CAST_LP_HP,
            CAST_HP_LP
        }
    }

    /**
     * Launches the same kernel on multiple partitions
     * @param {string|number} kernelName
     * @param {Array} kernelConfig
     * @param {Array} kernelPartitionedArguments
     *
     */
    _launchPartitionKernel = (kernelName, kernelConfig, kernelPartitionedArguments) => {
        for (let i = 0; i < this.numPartitions; ++i) {
            const kernelArguments = take(kernelPartitionedArguments, i)
            launchKernel(
                this.kernels[kernelName],
                kernelConfig,
                kernelArguments
            )
        }
    }


    /**
     * Launches the actual computation of the kernel
     * Handles building and cacheing the kernels if they have not been built before, as well as
     * performing {this.numEigen} iterations of Lanczos
     * @returns {Lanczos} the instance itself for chaining
     */
    compute = (iteration) => {
        // console.log("compute")
        this.iteration = iteration
        this.tridiagonalMatrix = []

        const nnzs = this.matrixPartitions.map(p => p.size)
        const Ns = this.matrixPartitions.map(p => p.N)
        const offsets = [0]
        for (let i = 1; i < this.numPartitions; ++i) {
            offsets.push(this.matrixPartitions[i - 1].N + offsets[i - 1])
        }

        if (Object.keys(this.kernels).length === 0) {
            this._buildKernels()
        }

        const System = Java.type("java.lang.System");
        const beginTime = System.nanoTime()

        if (this.arrayInputType != this.inputType) {
            console.log("cast1")
            this._launchPartitionKernel(
                "CAST_HP_LP",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceCooVal, this.deviceCooValLP, nnzs]
            )
            console.log("cast2")
            this._launchPartitionKernel(
                "CAST_HP_LP",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecIn, this.deviceVecInLP, new Array(this.numPartitions).fill(this.N)]
            )

            this._launchPartitionKernel(
                "SPMV",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceCooX, this.deviceCooY, this.deviceCooValLP, this.deviceVecInLP, this.deviceVecOutSpmv, nnzs]
            )
            console.log("spmv", this.deviceVecOutSpmv[0][0])

            this._launchPartitionKernel(
                "DOT_PRODUCT",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecOutSpmv, this.deviceVecIn, this.alphaIntermediate, Ns, offsets]
            )
        } else {

            this._launchPartitionKernel(
                "SPMV",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceCooX, this.deviceCooY, this.deviceCooVal, this.deviceVecIn, this.deviceVecOutSpmv, nnzs]
            )
            console.log("spmv", this.deviceVecOutSpmv[0][0])

            this._launchPartitionKernel(
                "DOT_PRODUCT",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecOutSpmv, this.deviceVecIn, this.alphaIntermediate, Ns, offsets]
            )

            console.log("dp2", this.alphaIntermediate[0][0])

        }

        this.alpha = this.alphaIntermediate.reduce((acc, cur) => acc + cur[0], 0.0)
        this.tridiagonalMatrix.push(this.alpha)
        console.log("alpha")

        if (this.arrayInputType != this.inputType) {
            this._launchPartitionKernel(
                "AXPB_XTENDED",
                DEFAULT_KERNEL_CONFIG,
                [Array(this.numPartitions).fill(-this.alpha), this.deviceVecInLP, this.deviceVecOutSpmv, Array(this.numPartitions).fill(0), this.deviceLanczosVectors, this.deviceVecNext, Ns, offsets, Array(this.numPartitions).fill(0)]
            )
        } else {

            this._launchPartitionKernel(
                "AXPB_XTENDED",
                DEFAULT_KERNEL_CONFIG,
                [Array(this.numPartitions).fill(-this.alpha), this.deviceVecIn, this.deviceVecOutSpmv, Array(this.numPartitions).fill(0), this.deviceLanczosVectors, this.deviceVecNext, Ns, offsets, Array(this.numPartitions).fill(0)]
            )
        }
        console.log("axpb")

        for (let i = 1; i < this.numEigen; ++i) {

            //Compute the l2 norm
            this._launchPartitionKernel(
                "L2_NORM",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecNext, this.betaIntermediate, Ns, new Array(this.numPartitions).fill(0)]
            )


            console.log("l2_norm", this.betaIntermediate[0][0])

            this.beta = Math.sqrt(this.betaIntermediate.reduce((acc, cur) => acc + cur[0], 0.0))
            this.tridiagonalMatrix.push(this.beta)

            // Normalize the vector
            this._launchPartitionKernel(
                "NORMALIZE",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecNext, Array(this.numPartitions).fill(1 / this.beta), this.deviceNormalizeOut, Ns]
            )
            console.log("norm")

            // Copy vec in to lancsoz vectors
            if (this.arrayInputType != this.inputType) {
                this._launchPartitionKernel(
                    "COPY_PARTITION_TO_VEC_FW",
                    DEFAULT_KERNEL_CONFIG,
                    [this.deviceVecInLP, this.deviceLanczosVectors, Ns, Ns.map(N => N * (i - 1)), offsets]
                )
            } else {
                this._launchPartitionKernel(
                    "COPY_PARTITION_TO_VEC_FW",
                    DEFAULT_KERNEL_CONFIG,
                    [this.deviceVecIn, this.deviceLanczosVectors, Ns, Ns.map(N => N * (i - 1)), offsets]
                )
            }
            console.log("copy")


            // Copy deviceNormalizedOut slicewise, by rotating pointers that refer to a certain partition
            let deviceNormalizedOutTmp = this.deviceNormalizeOut
            for (let j = 0; j < this.numPartitions; ++j) {

                if (this.arrayInputType != this.inputType) {
                    this._launchPartitionKernel(
                        "COPY_PARTITION_TO_VEC_BW",
                        DEFAULT_KERNEL_CONFIG,
                        [deviceNormalizedOutTmp, this.deviceVecInLP, Ns, offsets, Array(this.numPartitions).fill(0)]
                    )
                    console.log("copytoVecIn")

                    let lastVector = this.deviceVecInLP.pop()
                    this.deviceVecInLP.unshift(lastVector)

                } else {
                    this._launchPartitionKernel(
                        "COPY_PARTITION_TO_VEC_BW",
                        DEFAULT_KERNEL_CONFIG,
                        [deviceNormalizedOutTmp, this.deviceVecIn, Ns, offsets, Array(this.numPartitions).fill(0)]
                    )
                    console.log("copytoVecIn")


                    let lastVector = this.deviceVecIn.pop()
                    this.deviceVecIn.unshift(lastVector)
                }

            }

            if (this.arrayInputType != this.inputType) {
                this._launchPartitionKernel(
                    "SPMV",
                    DEFAULT_KERNEL_CONFIG,
                    [this.deviceCooX, this.deviceCooY, this.deviceCooValLP, this.deviceVecInLP, this.deviceVecOutSpmv, nnzs]
                )
                console.log("spmv")

                this._launchPartitionKernel(
                    "DOT_PRODUCT",
                    DEFAULT_KERNEL_CONFIG,
                    [this.deviceVecOutSpmv, this.deviceVecInLP, this.alphaIntermediate, Ns, offsets]
                )
            } else {
                this._launchPartitionKernel(
                    "SPMV",
                    DEFAULT_KERNEL_CONFIG,
                    [this.deviceCooX, this.deviceCooY, this.deviceCooVal, this.deviceVecIn, this.deviceVecOutSpmv, nnzs]
                )
                console.log("spmv")

                this._launchPartitionKernel(
                    "DOT_PRODUCT",
                    DEFAULT_KERNEL_CONFIG,
                    [this.deviceVecOutSpmv, this.deviceVecIn, this.alphaIntermediate, Ns, offsets]
                )
            }

            console.log("dp2", this.alphaIntermediate[0][0])

            this.alpha = this.alphaIntermediate.reduce((acc, cur) => acc + cur[0], 0.0)
            this.tridiagonalMatrix.push(this.alpha)

            if (this.arrayInputType != this.inputType) {
                this._launchPartitionKernel(
                    "AXPB_XTENDED",
                    DEFAULT_KERNEL_CONFIG,
                    [Array(this.numPartitions).fill(-this.alpha), this.deviceVecInLP, this.deviceVecOutSpmv, Array(this.numPartitions).fill(-this.beta), this.deviceLanczosVectors, this.deviceVecNext, Ns, offsets, Ns.map(N => N * (i - 1))]
                )
            } else {
                this._launchPartitionKernel(
                    "AXPB_XTENDED",
                    DEFAULT_KERNEL_CONFIG,
                    [Array(this.numPartitions).fill(-this.alpha), this.deviceVecIn, this.deviceVecOutSpmv, Array(this.numPartitions).fill(-this.beta), this.deviceLanczosVectors, this.deviceVecNext, Ns, offsets, Ns.map(N => N * (i - 1))]
                )
            }

            console.log("axpb")

            if (this.reorthogonalize) {
                for (let j = 0; j < i; ++j) {

                    this._launchPartitionKernel(
                        "DOT_PRODUCT_HP",
                        DEFAULT_KERNEL_CONFIG,
                        [this.deviceVecNext, this.deviceLanczosVectors, this.alphaIntermediateHP, Ns, Ns.map(N => N * j)]
                    )
             
                    console.log("dp2")

                    this.alphaHP = this.alphaIntermediateHP.reduce((acc, cur) => acc + cur[0], 0)

                    this._launchPartitionKernel(
                        "SUBTRACT",
                        [BLOCKS, THREADS_PER_BLOCK],
                        [this.deviceVecNext, this.deviceLanczosVectors, Array(this.numPartitions).fill(this.alpha), Ns, Ns]
                    )
                    console.log("sub")


                }
            }


        }

        const endTime = System.nanoTime()
        this.executionTime = endTime - beginTime
        return this
    }


    /**
     * Resets and clears up vectors for the next computation
     * @returns {Lanczos} the instance itself for chaining
     */
    reset = () => {
        fillWith(this.deviceVecIn[0], Math.random)
        fillWith(this.deviceVecOutSpmv[0], () => { return 0 })
        arrayNormalize(this.deviceVecIn[0])

        for (let i = 1; i < this.numPartitions; ++i) {
            copy(this.deviceVecIn[0], this.deviceVecIn[i])
        }

        for (let i = 1; i < this.numPartitions; ++i) {
            copy(this.deviceVecOutSpmv[0], this.deviceVecOutSpmv[i])
        }

        return this
    }

    /**
     * Prints execution time
     * Generally speaking here is where, if needed, the result of the computation could be displayed.
     * @returns {Lanczos} the instance itself for chaining
     */
    printResults = () => {
        const accuracyOrth = Math.abs((this.computeAccuracy() * 180 / Math.PI) - 90)
        process.stdout.write(`${this.matrixName},${this.numEigen},${this.iteration},${accuracyOrth},${this.executionTime}`)
        //console.log(`${this.matrixName},${this.N},${this.nnz},${this.numEigen},${this.iteration},${this.executionTime / 1000},${this.numPartitions}`)
        return this
    }


    computeAccuracy = () => {

        // retrieve lanczos vectors
        this.lanczosVectors = []

        for (let i = 0; i < this.numEigen; ++i) {
            this.lanczosVectors.push([])
        }
        for (let i = 0; i < this.numEigen; ++i) {
            for (let pIdx = 0; pIdx < this.numPartitions; ++pIdx) {
                const N = this.matrixPartitions[pIdx].N

                for (let j = 0; j < N; ++j) {
                    this.lanczosVectors[i].push(this.deviceLanczosVectors[pIdx][i * N + j])
                }
            }
        }

        // console.log(this.lanczosVectors)
        for (let i = 0; i < this.numEigen; ++i) {
            arrayNormalize(this.lanczosVectors[i])
        }

        let orthogonality = 0.0

        for (let i = 0; i < this.numEigen; ++i) {
            for (let j = 0; j < this.numEigen; ++j) {
                if (i == j) continue
                const dp = dotProduct(this.lanczosVectors[i], this.lanczosVectors[j])
                if (!Number.isNaN(dp)) {
                    orthogonality += dp
                }

            }
        }

        return Math.acos(orthogonality / (this.numEigen * this.numEigen))
    }

    computeFullReconstructionError = (cooMatrix, lanczosVectors, eigenvalues, tEigenvectors) => {
        let transposedLanczosVectors = new Array(this.N).fill([])
        transposedLanczosVectors = transposedLanczosVectors.map(_ => new Array(this.numEigen).fill(0))
        let residual = 0.0

        for (let i = 0; i < this.numEigen; ++i) {
            for (let j = 0; j < this.N; ++j) {
                transposedLanczosVectors[j][i] = Number.isNaN(lanczosVectors[i][j]) ? 0.0 : lanczosVectors[i][j]
            }
        }


        for (let i = 0; i < this.numEigen; ++i) {
            const eigenvector = matmul(transposedLanczosVectors, tEigenvectors[i])
            const lhs = eigenvector.map(v => v * eigenvalues[i][i])
            const rhs = spmvCoo(cooMatrix, eigenvector, this.N, this.nnz)

            residual += rhs.reduce((acc, cur, idx) => {
                const tmp = (cur - lhs[idx])
                if (Number.isNaN(tmp)) return acc
                else return acc + tmp
            })

        }

        process.stdout.write(`${residual / this.N}\n`)

    }

    /**
     * Frees GPU memory.
     */
    free = () => {

        this.matrixPartitions.forEach((partition, i) => {
            this.deviceVecIn[i].free()
            this.deviceVecOutSpmv[i].free()
            this.deviceIntermediateDotProductValues[i].free()
            this.alphaIntermediate[i].free()
            this.betaIntermediate[i].free()
            this.deviceCooVal[i].free()
            this.deviceCooX[i].free()
            this.deviceCooY[i].free()
        })

    }

}

module.exports = { Lanczos }
