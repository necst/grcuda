import WebSocket from 'ws'


export class GrCUDAProxy {
  private ws: WebSocket
  private computationType: string
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

  private async computeSync(){
    console.log("Computing using mode Sync")
    await this.mockCompute()
  }
  private async computeAsync(){
    console.log("Computing using mode Async")
    await this.mockCompute()
  }
  private async computeNative(){
    console.log("Computing using mode Native")
    await this.mockCompute()
  }

  private async mockCompute(){

    const {
      DELAY
    } = this.MOCK_OPTIONS
    const {
      MAX_PHOTOS,
      SEND_BATCH_SIZE
    } = this.CONFIG_OPTIONS

    let imagesToSend: Array<string> = []


    for(let i = 0; i < MAX_PHOTOS; ++i){

      await this._sleep(DELAY)

      this.ws.send(JSON.stringify({
        type: "progress",
        data: i / MAX_PHOTOS * 100
      }))

      imagesToSend.push(`./images/thumb/${("0000" + i).slice(-4)}.jpg`)

      if((i !== 0 && !(i % SEND_BATCH_SIZE) || i === MAX_PHOTOS - 1)) {

        this.ws.send(JSON.stringify({
              type: "image",
              images: imagesToSend
            }))

        imagesToSend = []

      }


    }

  }

  private _sleep(ms: number) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  } 
}

