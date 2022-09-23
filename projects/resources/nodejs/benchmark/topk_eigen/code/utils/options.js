const BLOCKS = 32
const THREADS_PER_BLOCK = 256
const DEFAULT_KERNEL_CONFIG = [BLOCKS, THREADS_PER_BLOCK]

module.exports = {
    BLOCKS, THREADS_PER_BLOCK, DEFAULT_KERNEL_CONFIG
}