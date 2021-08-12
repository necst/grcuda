import WebSocket from 'ws'


export class GrCUDAProxy {
  private ws: WebSocket
  private computationType: string
  private imagesToSend: Array<string> = []

  private MOCK_OPTIONS = {
    DELAY: 10,              //ms
    DELAY_JITTER_SYNC: 30,  //ms
    DELAY_JITTER_ASYNC: 0,  //ms
    DELAY_JITTER_NATIVE: 50 //ms
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
   * @returns `void`
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

    if (computationType == "race-mode"){
      await this.raceMode()
      return
    }

   throw new Error(`Could not recognize computation type: ${computationType}`)

  }

  private async raceMode() {
    console.log("Computing using mode RACE!")
    await this.mockRace()
  }

  private async mockRace() {
    this.mockRaceProgress("sync")
    this.mockRaceProgress("async")
    this.mockRaceProgress("cuda-native")

  }

  private async mockRaceProgress(computationType: string) {
    const {
      DELAY
    } = this.MOCK_OPTIONS

    const {
      MAX_PHOTOS,
      SEND_BATCH_SIZE
    } = this.CONFIG_OPTIONS
    
    let delay_jitter = this._getDelayJitter(computationType)

    for(let imageId = 0; imageId < MAX_PHOTOS; imageId++){
      await this._sleep(DELAY + Math.random() * delay_jitter)
      this.ws.send(JSON.stringify({
        type: "progress",
        data: imageId / MAX_PHOTOS * 100, 
        computationType: `race-${computationType}`
      }))
    }

    this.ws.send(JSON.stringify({
      type: "progress",
      data: 100, 
      computationType: `race-${computationType}`
    }))
  
  }

  /*
   * Compute the GrCUDA kernel using Sync mode
   * @returns `void`
   */
  private async computeSync(){
    console.log("Computing using mode Sync")
    await this.mockCompute("sync")
  }

  /*
   * Compute the GrCUDA kernel using Async mode
   * @returns `void`
   */
  private async computeAsync(){
    console.log("Computing using mode Async")
    await this.mockCompute("async")
  }

  /*
   * Compute the GrCUDA kernel using native 
   * CUDA code by `exec`ing the kernel via 
   * a shell
   * @returns `void`
   */
  private async computeNative(){
    console.log("Computing using mode Native")
    await this.mockCompute("cuda-native")
  }

  /* Mock the computation of the kernels 
   * inside GrCUDA.
   * Sends a `progress` message every time an image is computed
   * and a `image` message every time BATCH_SIZE images have been computed
   */
  private async mockCompute(computationType: string){

    const {
      DELAY
    } = this.MOCK_OPTIONS

    const {
      MAX_PHOTOS,
      SEND_BATCH_SIZE
    } = this.CONFIG_OPTIONS

    let delay_jitter = this._getDelayJitter(computationType)

    for(let i = 0; i < MAX_PHOTOS; ++i){
      // This does mock the actual computation that will happen 
      // in the CUDA realm
      await this._sleep(DELAY + Math.random() * delay_jitter)

      this.communicateImageProcessed(i, computationType)
    }
    this.communicateImageProcessed(MAX_PHOTOS, computationType)

  }

  private communicateImageProcessed(imageId: number, computationType: string) {

    const {
      SEND_BATCH_SIZE, 
      MAX_PHOTOS
    } = this.CONFIG_OPTIONS

    this.ws.send(JSON.stringify({
      type: "progress",
      data: imageId / MAX_PHOTOS * 100, 
      computationType
    }))

    this.imagesToSend.push(`./images/thumb/${("0000" + imageId).slice(-4)}.jpg`)

    if((imageId !== 0 && !(imageId % SEND_BATCH_SIZE) || imageId === MAX_PHOTOS - 1)) {

      this.ws.send(JSON.stringify({
            type: "image",
            images: this.imagesToSend,
            computationType
          }))

      this.imagesToSend = []

    }
  }
  private _sleep(ms: number) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  } 

  private _getDelayJitter(computationType: string) {

    const {
      DELAY_JITTER_ASYNC,
      DELAY_JITTER_SYNC,
      DELAY_JITTER_NATIVE
    } = this.MOCK_OPTIONS

    switch(computationType) {
      case "sync": {
        return DELAY_JITTER_SYNC
      }
      case "async": {
        return DELAY_JITTER_ASYNC
      }
      case "cuda-native": {
        return DELAY_JITTER_NATIVE
      }
    }
  
  }

}

