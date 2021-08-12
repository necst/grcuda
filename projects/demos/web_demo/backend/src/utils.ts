export const MOCK_OPTIONS = {
  DELAY: 10,              //ms
  DELAY_JITTER_SYNC: 30,  //ms
  DELAY_JITTER_ASYNC: 0,  //ms
  DELAY_JITTER_NATIVE: 50 //ms
}

export const CONFIG_OPTIONS = {
  MAX_PHOTOS: 200,
  SEND_BATCH_SIZE: 20
}

export const COMPUTATION_MODES: Array<string> = ["sync", "async", "cuda-native", "race-sync", "race-async", "race-cuda-native"]

export const _sleep = (ms: number) => {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
} 

export const _getDelayJitter = (computationType: string) => {

  const {
    DELAY_JITTER_ASYNC,
    DELAY_JITTER_SYNC,
    DELAY_JITTER_NATIVE
  } = MOCK_OPTIONS

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