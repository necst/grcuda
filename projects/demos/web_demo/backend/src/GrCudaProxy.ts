import WebSocket from 'ws'
import {
  _sleep,
  _getDelayJitter,
  MOCK_OPTIONS,
  COMPUTATION_MODES,
  CONFIG_OPTIONS
} from './utils'

export class GrCUDAProxy {
  private ws: WebSocket
  private computationType: string
  private imagesToSend: {[id: string]: Array<string>} = {}
  

  constructor(ws: WebSocket){
    this.ws = ws
  }

  /*
   * Begins the computation using the mode specified
   * by `computationType`
   * @param computationType {string}
   * @returns `void`
   */
  public async beginComputation(computationType: string){
    this.computationType = computationType

    COMPUTATION_MODES.forEach(cm => this.imagesToSend[cm] = [])

    if (computationType == "sync" || computationType == "race-sync"){
      await this.computeSync(computationType.toString())
      return
    }
    if (computationType == "async" || computationType == "race-async"){
      await this.computeAsync(computationType.toString())
      return
    }
    if (computationType == "cuda-native" || computationType == "race-cuda-native"){
      await this.computeNative(computationType.toString())
      return
    }

    throw new Error(`Could not recognize computation type: ${computationType}`)

  }

  private communicateAll(imageId: number, computationType: string) {

    const {
      MAX_PHOTOS,
    } = CONFIG_OPTIONS

    this.communicateProgress(imageId / MAX_PHOTOS * 100, computationType)
    this.communicateImageProcessed(imageId, computationType)
  }

  /*
   * Compute the GrCUDA kernel using Sync mode
   * @returns `void`
   */
  private async computeSync(computationType: string){
    console.log("Computing using mode Sync")
    await this.mockCompute(computationType)
  }

  /*
   * Compute the GrCUDA kernel using Async mode
   * @returns `void`
   */
  private async computeAsync(computationType: string){
    console.log("Computing using mode Async")
    await this.mockCompute(computationType)
  }

  /*
   * Compute the GrCUDA kernel using native 
   * CUDA code by `exec`ing the kernel via 
   * a shell
   * @returns `void`
   */
  private async computeNative(computationType: string){
    console.log("Computing using mode Native")
    await this.mockCompute(computationType)
  }

  /* Mock the computation of the kernels 
   * inside GrCUDA.
   * Sends a `progress` message every time an image is computed
   * and a `image` message every time BATCH_SIZE images have been computed
   */
  private async mockCompute(computationType: string){

    const {
      DELAY
    } = MOCK_OPTIONS

    const {
      MAX_PHOTOS,
    } = CONFIG_OPTIONS

    let delay_jitter = _getDelayJitter(computationType)

    for(let imageId = 0; imageId < MAX_PHOTOS; ++imageId){
      // This does mock the actual computation that will happen 
      // in the CUDA realm
      await _sleep(DELAY + Math.random() * delay_jitter)
      this.communicateAll(imageId, computationType)
    }
    this.communicateAll(MAX_PHOTOS, computationType)


  }

  private communicateProgress(data: number, computationType: string){
    const {
      MAX_PHOTOS
    } = CONFIG_OPTIONS

    this.ws.send(JSON.stringify({
      type: "progress",
      data: data, 
      computationType
    }))
  }

  private communicateImageProcessed(imageId: number, computationType: string) {
    const {
      SEND_BATCH_SIZE, 
      MAX_PHOTOS
    } = CONFIG_OPTIONS

    this.imagesToSend[computationType].push(`./images/thumb/${("0000" + imageId).slice(-4)}.jpg`)

    if((imageId !== 0 && !(imageId % SEND_BATCH_SIZE) || imageId === MAX_PHOTOS - 1)) {

      this.ws.send(JSON.stringify({
            type: "image",
            images: this.imagesToSend[computationType],
            computationType
          }))

      this.imagesToSend[computationType] = []

    }
  }

}

