const {
    fillWith,
    arrayNormalize,
    launchKernel,
    take
} = require("../utils/utils.js")

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


    /**
     * Creates an object that handles the whole computation
     * @param cooMatrix Input matrix, in sparse coordinate format
     * @param cu Instance of GrCUDA
     * @param {number} numPartitions number of (balanced) partitions
     * @param {boolean} reorthogonalize whether to reorthogonalize the lanczos vectors or not
     */
    constructor(cooMatrix, numEigen, cu, numPartitions, reorthogonalize) {
        this.cooMatrix = cooMatrix
        this.N = Math.arrayMax(cooMatrix.x)
        this.M = Math.arrayMax(cooMatrix.y)
        this.nnz = cooMatrix.val.length
        this.cu = cu
        this.numEigen = numEigen
        this.reorthogonalize = reorthogonalize
        this.numPartitions = numPartitions

    }


    /**
     * Private method to build _somewhat_ balanced partitions wrt the number of nonzeros
     * After determining the optimal partition size, it advances the iterator until the reference y index does change
     * This is done in order to not interrupt the current partition at a given y index
     * @private
     */
    _createPartitions = () => {
        const nnzPerPartition = ( (this.nnz + this.numPartitions) / this.numPartitions) >> 0
        let fromIndex = 0
        let toIndex = nnzPerPartition
        let indexValue = this.cooMatrix.y[toIndex]

        for(let i = 0; i < this.numPartitions - 1; ++i){
            // advance the next index until the values differ
            while(indexValue === this.cooMatrix.y[toIndex]){
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
        let cooPartition = {x: [], y: [], val: [], size: -1, begin: -1, end: -1, N: -1}
        cooPartition.size   = toIndex - fromIndex
        cooPartition.begin  = fromIndex
        cooPartition.end    = toIndex
        cooPartition.x      = []//new Array(cooPartition.size).fill(0)
        cooPartition.y      = []//new Array(cooPartition.size).fill(0)
        cooPartition.val    = []//new Array(cooPartition.size).fill(0)


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
        for(let i = 1; i < this.numPartitions; ++i){
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
            copy(this.vecsOutSpmv[i], this.deviceVecOutSpmv[i])
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
        this.deviceVecIn                        = this.matrixPartitions.map(unused => this.cu.DeviceArray("float", this.N))
        this.deviceIntermediateDotProductValues = this.matrixPartitions.map(unused => this.cu.DeviceArray("float", BLOCKS))
        this.alphaIntermediate                  = this.matrixPartitions.map(unused => this.cu.DeviceArray("float", 1))
        this.betaIntermediate                   = this.matrixPartitions.map(unused => this.cu.DeviceArray("float", 1))
        this.deviceVecOutSpmv                   = this.matrixPartitions.map(partition => this.cu.DeviceArray("float", partition.N))
        this.deviceCooX                         = this.matrixPartitions.map(partition => this.cu.DeviceArray("int", partition.size))
        this.deviceCooY                         = this.matrixPartitions.map(partition => this.cu.DeviceArray("int", partition.size))
        this.deviceCooVal                       = this.matrixPartitions.map(partition => this.cu.DeviceArray("float", partition.size))
        this.deviceVecNext                      = this.matrixPartitions.map(partition => this.cu.DeviceArray("float", partition.N))
        this.deviceNormalizeOut                 = this.matrixPartitions.map(partition => this.cu.DeviceArray("float", partition.N))
        this.deviceLanczosVectors               = this.matrixPartitions.map(partition => this.cu.DeviceArray("float", partition.N * this.numEigen))
        this.alpha                              = this.cu.DeviceArray("float", 1)
        this.beta                               = this.cu.DeviceArray("float", 1)
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

        const SPMV = this.cu.buildkernel(kernelsDef.SPMV, "spmv", "const pointer, const pointer, const pointer, const pointer, pointer, const sint32")
        const DOT_PRODUCT_STAGE_ONE = this.cu.buildkernel(kernelsDef.DOT_PRODUCT_STAGE_ONE, "dot_product_stage_one",  "const pointer, const pointer, pointer, const sint32, const sint32")
        const DOT_PRODUCT_STAGE_TWO = this.cu.buildkernel(kernelsDef.DOT_PRODUCT_STAGE_TWO, "dot_product_stage_two",  "const pointer, pointer")
        const AXPB_XTENDED = this.cu.buildkernel(kernelsDef.AXPB_XTENDED, "axpb_xtended", "const float, const pointer, const pointer, const float, const pointer, pointer, const sint32, const sint32, const sint32")
        const NORMALIZE = this.cu.buildkernel(kernelsDef.NORMALIZE, "normalize", "const pointer, const float, pointer, const sint32")
        const STORE_AND_RESET = this.cu.buildkernel(kernelsDef.STORE_AND_RESET, "store_and_reset", "const pointer, pointer, pointer, sint32, sint32")
        const SUBTRACT = this.cu.buildkernel(kernelsDef.SUBTRACT, "subtract", "pointer, const pointer, float, const sint32, const sint32")
        const COPY_PARTITION_TO_VEC = this.cu.buildkernel(kernelsDef.COPY_PARTITION_TO_VEC, "copy_partition_to_vec", "const pointer, pointer, const sint32, const sint32, const sint32")


        this.kernels = {
            SPMV, DOT_PRODUCT_STAGE_ONE, DOT_PRODUCT_STAGE_TWO, AXPB_XTENDED, NORMALIZE, STORE_AND_RESET, SUBTRACT, COPY_PARTITION_TO_VEC
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
        for(let i = 0; i < this.numPartitions; ++i){
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
    compute = () => {

        this.tridiagonalMatrix = []

        const nnzs = this.matrixPartitions.map(p => p.size)
        const Ns = this.matrixPartitions.map(p => p.N)
        const offsets = [0]
        for(let i = 1; i < this.numPartitions; ++i){
            offsets.push(this.matrixPartitions[i - 1].N + offsets[i - 1])
        }

        if(Object.keys(this.kernels).length === 0){
            this._buildKernels()
        }

        const System = Java.type("java.lang.System");
        const beginTime = System.nanoTime()
    
        this._launchPartitionKernel(
            "SPMV",
            DEFAULT_KERNEL_CONFIG,
            [this.deviceCooX, this.deviceCooY, this.deviceCooVal, this.deviceVecIn, this.deviceVecOutSpmv, nnzs]
        )


        this._launchPartitionKernel(
            "DOT_PRODUCT_STAGE_ONE",
            [...DEFAULT_KERNEL_CONFIG, 4 * THREADS_PER_BLOCK],
            [this.deviceVecOutSpmv, this.deviceVecIn, this.deviceIntermediateDotProductValues, Ns, offsets]
        )

        this._launchPartitionKernel(
            "DOT_PRODUCT_STAGE_TWO",
            [1, BLOCKS],
            [this.deviceIntermediateDotProductValues, this.alphaIntermediate]
        )

        this.alpha = this.alphaIntermediate.reduce((acc, cur) => acc + cur[0], 0.0)
        this.tridiagonalMatrix.push(this.alpha)


        this._launchPartitionKernel(
            "AXPB_XTENDED",
            DEFAULT_KERNEL_CONFIG,
            [Array(this.numPartitions).fill(-this.alpha), this.deviceVecIn, this.deviceVecOutSpmv, Array(this.numPartitions).fill(0), this.deviceVecIn, this.deviceVecNext, Ns, offsets, Array(this.numPartitions).fill(0)]
        )


        for(let i = 1; i < this.numEigen; ++i){

            //Compute the l2 norm
            this._launchPartitionKernel(
                "DOT_PRODUCT_STAGE_ONE",
                [...DEFAULT_KERNEL_CONFIG, 4 * THREADS_PER_BLOCK],
                [this.deviceVecNext, this.deviceVecNext, this.deviceIntermediateDotProductValues, Ns, Array(this.numPartitions).fill(0)]
            )

            this._launchPartitionKernel(
                "DOT_PRODUCT_STAGE_TWO",
                [1, BLOCKS],
                [this.deviceIntermediateDotProductValues, this.betaIntermediate]
            )

            this.beta = Math.sqrt(this.betaIntermediate.reduce((acc, cur) => acc + cur[0], 0.0))
            this.tridiagonalMatrix.push(this.beta)


            // Normalize the vector
            this._launchPartitionKernel(
                "NORMALIZE",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecNext, Array(this.numPartitions).fill(1 / this.beta), this.deviceNormalizeOut, Ns]
            )

            // Copy vec in to lancsoz vectors
            this._launchPartitionKernel(
                "COPY_PARTITION_TO_VEC",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceVecIn, this.deviceLanczosVectors, Ns, Ns.map(N => N * (i - 1)), offsets]
            )

            // Copy deviceNormalizedOut slicewise, by rotating pointers that refer to a certain partition
            let deviceNormalizedOutTmp = this.deviceNormalizeOut
            for(let j = 0; j < this.numPartitions; ++j){
                this._launchPartitionKernel(
                    "COPY_PARTITION_TO_VEC",
                    DEFAULT_KERNEL_CONFIG,
                    [deviceNormalizedOutTmp, this.deviceVecIn, Ns, offsets, Array(this.numPartitions).fill(0)]
                )

                let lastVector = this.deviceVecIn.pop()
                this.deviceVecIn.unshift(lastVector)

            }

            this._launchPartitionKernel(
                "SPMV",
                DEFAULT_KERNEL_CONFIG,
                [this.deviceCooX, this.deviceCooY, this.deviceCooVal, this.deviceVecIn, this.deviceVecOutSpmv, nnzs]
            )


            this._launchPartitionKernel(
                "DOT_PRODUCT_STAGE_ONE",
                [...DEFAULT_KERNEL_CONFIG, 4 * THREADS_PER_BLOCK],
                [this.deviceVecOutSpmv, this.deviceVecIn, this.deviceIntermediateDotProductValues, Ns, offsets]
            )

            this._launchPartitionKernel(
                "DOT_PRODUCT_STAGE_TWO",
                [1, BLOCKS],
                [this.deviceIntermediateDotProductValues, this.alphaIntermediate]
            )

            this.alpha = this.alphaIntermediate.reduce((acc, cur) => acc + cur[0], 0.0)
            this.tridiagonalMatrix.push(this.alpha)

            this._launchPartitionKernel(
                "AXPB_XTENDED",
                DEFAULT_KERNEL_CONFIG,
                [Array(this.numPartitions).fill(-this.alpha), this.deviceVecIn, this.deviceVecOutSpmv, Array(this.numPartitions).fill(-this.beta), this.deviceLanczosVectors, this.deviceVecNext, Ns, offsets, Ns.map(N => N * (i - 1))]
            )


            if(this.reorthogonalize){
                for(let j = 0; j < i; ++j){

                    this._launchPartitionKernel(
                        "DOT_PRODUCT_STAGE_ONE",
                        [...DEFAULT_KERNEL_CONFIG, 4 * THREADS_PER_BLOCK],
                        [this.deviceVecNext, this.deviceLanczosVectors, this.deviceIntermediateDotProductValues, Ns, Ns.map(N => N * j)]
                    )

                    this._launchPartitionKernel(
                        "DOT_PRODUCT_STAGE_TWO",
                        [1, BLOCKS],
                        [this.deviceIntermediateDotProductValues, this.alphaIntermediate]
                    )

                    this.alpha = this.alphaIntermediate.reduce((acc, cur) => acc + cur[0], 0)

                    this._launchPartitionKernel(
                        "SUBTRACT",
                        [BLOCKS, THREADS_PER_BLOCK],
                        [this.deviceVecNext, this.deviceLanczosVectors, Array(this.numPartitions).fill(this.alpha), Ns, Ns]
                    )

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
        fillWith(this.deviceVecOutSpmv[0], () => {return 0})
        arrayNormalize(this.deviceVecIn[0])

        for(let i = 1; i < this.numPartitions; ++i){
            copy(this.deviceVecIn[0], this.deviceVecIn[i])
        }

        for(let i = 1; i < this.numPartitions; ++i){
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
        console.log(`[ LANCZOS ] computation took ${this.executionTime / 1000}us`)
        return this
    }

    /**
     * Frees GPU memory.
     */
    free = () => {

        this.matrixPartitions.forEach( (partition, i) => {
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