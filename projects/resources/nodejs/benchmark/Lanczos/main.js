const {
    readDataset
} = require("./utils/io.js")

const { Lanczos } = require("./Kernels/Lanczos.js")
const { Jacobi } = require("./Kernels/Jacobi.js")


const cooMatrix = readDataset(process.argv[2], true)
const numEigen = parseInt(process.argv[3])
const numRuns = parseInt(process.argv[4])
const numPartitions = parseInt(process.argv[5])
const reorthogonalize = process.argv[6] === "true"
const debug = process.argv[7] === "true"

const cu = Polyglot.eval("grcuda", `CU`)

const lanczosKernel = new Lanczos(cooMatrix, numEigen, cu, numPartitions,  reorthogonalize)

lanczosKernel
    .build()
lanczosKernel
    .transferToGPU()


for(let i = 0; i < numRuns; ++i){
    lanczosKernel
        .compute()
        .printResults()

    lanczosKernel.reset()
}

lanczosKernel.free()



