const {
    copy,
    launchKernel, fillWith, createMatrix
} = require("../utils/utils.js")

const kernelsDef = require("../utils/kernels.js");


class Jacobi {

    constructor(eigenCount, cu) {
        this.tridiagonalMatrix = []
        this.N = eigenCount
        this.cu = cu
        this.kernels = {}

        this.executionTime = 0
    }

    build = (tm) => {
        this.tridiagonalMatrix = tm
        this.matrix = createMatrix(this.N, this.N)
        this._asDenseMatrix()
        return this
    }

    _asDenseMatrix = () => {
        for(let i = 0; i < this.N - 1; ++i){
            const alpha = this.tridiagonalMatrix[i << 1]
            const beta = this.tridiagonalMatrix[(i << 1) + 1]
            this.matrix[i][i] = alpha
            this.matrix[i + 1][i] = beta
            this.matrix[i][i + 1] = beta
        }

        this.matrix[(this.N - 1)][this.N - 1] = this.tridiagonalMatrix[this.tridiagonalMatrix.length - 1]
        this.eigenvectors = createMatrix(this.N, this.N)
        for(let i = 0; i < this.N; ++i){
            this.eigenvectors[i][i] = 1
        }
    }


    _createBuffers = () => {
        this.deviceMatrix = this.cu.DeviceArray("float", this.N * this.N)
        this.deviceEigenvectors = this.cu.DeviceArray("float", this.N * this.N)
        this.deviceRotationAngles = this.cu.DeviceArray("float", this.N)
        copy(Array(this.N).fill(22), this.deviceRotationAngles)
        copy(this.tridiagonalMatrix, this.deviceMatrix)
        copy(this.eigenvectors, this.deviceEigenvectors)
    }

    transferToGPU = () => {
        //this._asDenseMatrix()
        this._createBuffers()
        return this
    }

    _buildKernels = () => {
        const JACOBI_DIAGONAL = this.cu.buildkernel(kernelsDef.JACOBI_DIAGONAL, "jacobi_diagonal", "pointer, pointer, const sint32")
        const JACOBI_OFFDIAGONAL = this.cu.buildkernel(kernelsDef.JACOBI_OFFDIAGONAL, "jacobi_offdiagonal", "pointer, const pointer, const sint32")

        this.kernels = {
            JACOBI_DIAGONAL,
            JACOBI_OFFDIAGONAL
        }

    }

    _maxOffdiagElement = () => {

        let k = -1
        let l = -1
        let maxValue = -Infinity

        for(let i = 0; i < this.N; ++i){
            for(let j = 0; j < this.N; ++j){
                if(i === j) continue
                if (this.matrix[i][j] > maxValue){
                    maxValue = this.matrix[i][j]
                    k = i
                    l = j
                }
            }
        }
        return [maxValue, k, l]
    }

    _jacobiInner = (pivot, pivotI, pivotJ) => {

        const diff = this.matrix[pivotJ][pivotJ] - this.matrix[pivotI][pivotI]
        const phi  = diff / (2 * pivot)
        const t = 1 / (Math.abs(phi) + Math.sqrt(phi * phi + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        const tau = s / (1 + c)
        let temp = pivot
        this.matrix[pivotI][pivotJ] = 0.0
        this.matrix[pivotJ][pivotI] = 0.0
        this.matrix[pivotI][pivotI] -= t * temp
        this.matrix[pivotJ][pivotJ] += t * temp

        for(let i = 0; i < pivotI; ++i){
            temp = this.matrix[i][pivotI]
            this.matrix[i][pivotI] = temp * s * (this.matrix[i][pivotJ] + tau * temp)
            this.matrix[i][pivotJ] += s * (temp - tau * this.matrix[i][pivotJ])
        }


        for(let i = pivotI + 1; i < pivotJ; ++i){
            temp = this.matrix[pivotI][i]
            this.matrix[pivotI][i] = temp - s * (this.matrix[i][pivotJ + tau * this.matrix[pivotI][i]])
            this.matrix[i][pivotJ] += s * (temp - tau * this.matrix[i][pivotJ])
        }


        for(let i = pivotJ + 1; i < this.N; ++i){
            temp = this.matrix[pivotI][i]
            this.matrix[pivotI][i] = temp - s * (this.matrix[pivotJ][i] + tau * temp)
            this.matrix[pivotJ][i] += s * (temp - tau * this.matrix[pivotJ][i])
        }


        for(let i = 0; i < this.N; ++i){
            temp = this.eigenvectors[i][pivotI]
            this.eigenvectors[i][pivotI] = temp - s * (this.eigenvectors[i][pivotJ] + tau * this.eigenvectors[i][pivotI])
            this.eigenvectors[i][pivotJ] += s * (temp - tau * this.eigenvectors[i][pivotJ])
        }


    }

    _truncate = () => {
        for(let i = 0; i < this.N; ++i){
            for(let j = 0; j < this.N; ++j){
                if(i == j) continue
                if(this.matrix[i][j] < 1e-9) this.matrix[i][j] = 0.0
                if(isNaN(this.matrix[i][j])) this.matrix[i][j] = 0.0
            }
        }
    }

    compute = (tol) => {

        if (tol === undefined) tol = 1e-14

        const System = Java.type("java.lang.System");
        const beginTime = System.nanoTime()
        let [pivot, pivotI, pivotJ]  = this._maxOffdiagElement()
        this.iterations = 0
        while(pivot > tol){
            let [pivot, pivotI, pivotJ]  = this._maxOffdiagElement()
            this._jacobiInner(pivot, pivotI, pivotJ)
            this._truncate()
            this.iterations++

            if(this.iterations > 100) break

        }

        this.executionTime = System.nanoTime() - beginTime
        return this
    }

    reset = () => {
        this._asDenseMatrix()
        //this._createBuffers()
        return this
    }

    printResults = () => {
        console.log(`[  JACOBI ] computation took ${this.executionTime / 1000}us with ${this.iterations} iterations`)
        // console.log("Eigenvalues")
        // console.log(this.matrix)
        // console.log("Eigenvectors")
        // console.log(this.eigenvectors)
        return this
    }

    free = () => {
        // this.deviceEigenvectors.free()
        // this.deviceRotationAngles.free()
        // this.deviceMatrix.free()
        return this
    }


}

module.exports = {Jacobi}