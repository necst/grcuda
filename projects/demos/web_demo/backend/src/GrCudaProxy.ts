import WebSocket from 'ws'


export class GrCUDAProxy {
  private ws: WebSocket
  private computationType: string
  private imagesToSend: Array<string> = []

  private MOCK_OPTIONS = {
    DELAY: 10 //ms
  }
  private CONFIG_OPTIONS = {
    MAX_PHOTOS: 200,
    SEND_BATCH_SIZE: 20
  }


  constructor(ws: WebSocket){
    this.ws = ws
  }

  /*
   * Begins the computation using the mode specified
   * by `computationType`
   * @param computationType {string}
   * @returns
   */
  public async beginComputation(computationType: string){
  
    this.computationType = computationType

    if (computationType == "sync"){
      await this.computeSync()
      return
    }
    if (computationType == "async"){
      await this.computeAsync()
      return
    }
    if (computationType == "cuda-native"){
      await this.computeNative()
      return
    }

   throw new Error(`Could not recognize computation type: ${computationType}`)

  }

  /*
   * Compute the GrCUDA kernel using Sync mode
   * @returns `void`
   */
  private async computeSync(){
    console.log("Computing using mode Sync")
    await this.mockCompute()
  }

  /*
   * Compute the GrCUDA kernel using Async mode
   * @returns `void`
   */
  private async computeAsync(){
    console.log("Computing using mode Async")
    await this.mockCompute()
  }

  /*
   * Compute the GrCUDA kernel using native 
   * CUDA code by `exec`ing the kernel via 
   * a shell
   * @returns `void`
   */
  private async computeNative(){
    console.log("Computing using mode Native")
    await this.mockCompute()
  }

  /* Mock the computation of the kernels 
   * inside GrCUDA.
   * Sends a `progress` message every time an image is computed
   * and a `image` message every time BATCH_SIZE images have been computed
   */
  private async mockCompute(){

    const {
      DELAY
    } = this.MOCK_OPTIONS
    const {
      MAX_PHOTOS,
      SEND_BATCH_SIZE
    } = this.CONFIG_OPTIONS


    for(let i = 0; i < MAX_PHOTOS; ++i){
      // This does mock the actual computation that will happen 
      // in the CUDA realm
      await this._sleep(DELAY + Math.random() * 20 - 10)

      this.communicateImageProcessed(i)


    }

  }

  private communicateImageProcessed(imageId: number) {

    const {
      SEND_BATCH_SIZE, 
      MAX_PHOTOS
    } = this.CONFIG_OPTIONS

    this.ws.send(JSON.stringify({
      type: "progress",
      data: imageId / MAX_PHOTOS * 100
    }))

    this.imagesToSend.push(`./images/thumb/${("0000" + imageId).slice(-4)}.jpg`)

    if((imageId !== 0 && !(imageId % SEND_BATCH_SIZE) || imageId === MAX_PHOTOS - 1)) {

      this.ws.send(JSON.stringify({
            type: "image",
            images: this.imagesToSend
          }))

      this.imagesToSend = []

    }
  }
  private _sleep(ms: number) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  } 
}

