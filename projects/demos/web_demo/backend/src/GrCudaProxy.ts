import WebSocket from 'ws'


export class GrCUDAProxy {
  private ws: WebSocket
  private computationType: string

  constructor(ws: WebSocket){
    this.ws = ws
  }

  public beginComputation(computationType: string){
  
    this.computationType = computationType

    if (computationType == "sync"){
      this.computeSync()
      return
    }
    if (computationType == "async"){
      this.computeAsync()
      return
    }
    if (computationType == "cuda-native"){
      this.computeNative()
      return
    }

   throw new Error(`Could not recognize computation type: ${computationType}`)

  }

  private computeSync(){
    console.log("Computing using mode Sync")
    this.mockCompute()
  }
  private computeAsync(){
    console.log("Computing using mode Async")
    this.mockCompute()
  }
  private computeNative(){
    console.log("Computing using mode Native")
    this.mockCompute()
  }

  private mockCompute(){



  }

}

