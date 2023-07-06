const {
    readDataset
} = require("./utils/io.js")

const { Lanczos } = require("./Kernels/Lanczos.js")
const { Jacobi } = require("./Kernels/Jacobi.js")


const cooMatrix = readDataset(process.argv[2], true)
const numEigen = parseInt(process.argv[3])
const numRuns = parseInt(process.argv[4])
const numPartitions = parseInt(process.argv[5])
const reorthogonalize = process.argv[6] === "false"
const debug = process.argv[7] === "true"

const input = process.argv[8] || "float"
const middle = process.argv[9] || "float"
const output = process.argv[10] || "float"

const cu = Polyglot.eval("grcuda", `CU`)
if (debug) console.log({ input, middle, output })
const lanczosKernel = new Lanczos(cooMatrix, numEigen, cu, numPartitions, reorthogonalize, process.argv[2], 0, { input, middle, output })
const jacobiKernel = new Jacobi(numEigen, cu)

lanczosKernel
    .build()
    .transferToGPU()

for (let i = 0; i < numRuns; ++i) {
    lanczosKernel
        .compute(i)
        .printResults()
/* 
    jacobiKernel
        .build(lanczosKernel.tridiagonalMatrix)
        .compute()
        .printResults()

    lanczosKernel
        .computeFullReconstructionError(
            cooMatrix,
            lanczosKernel.lanczosVectors,
            jacobiKernel.matrix,
            jacobiKernel.eigenvectors
        )
 */
    lanczosKernel.reset()
    //jacobiKernel.reset()
}

lanczosKernel.free()



