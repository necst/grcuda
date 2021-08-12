const ws = new WebSocket("ws://localhost:8080")
const sendWSMessage = document.getElementById("btn-send-msg-ws")
const progressBar = document.getElementById("progress-bar")
const imageGallery = document.getElementById("images")
const selectElement = document.getElementById("computation-type")
const containerInfo = document.getElementById("container-info")
const raceModeContainer = document.getElementById("race-mode")
const progressBarColor = {
  "race-sync": "progress-bar-striped", 
  "race-async": "progress-bar-striped",
  "race-cuda-native": "progress-bar-striped"
}
let progressSync = 0
let progressAsync = 0
let progressNative = 0

const progressBarsCompletionAmount = {
  
}

let imageGalleryContent = ""


ws.addEventListener("open", (evt) => {
  console.log("Connection to websocket is open at ws://localhost:8080")
})

ws.addEventListener("message", (evt) => {
  const data = JSON.parse(evt.data)

  if (data.type === "progress") {
    const { data: progressData, computationType } = JSON.parse(evt.data)
    // This is really bad, it forces a rerender of the whole element
    // at every progress message received.
    // Works for now but TODO: change this

    if (!computationType.includes("race")){
      if (progressData < 99.99) {
        progressBar.innerHTML = `
        <div class="progress m-4">
          <div style="width: ${progressData}%" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="${progressData}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressData)}%</div>
        </div>
      `
      } else {
        progressBar.innerHTML = `
        <div class="progress m-4">
          <div style="width: ${progressData}%" class="progress-bar bg-success" role="progressbar" aria-valuenow="${progressData}" aria-valuemin="0" aria-valuemax="100">${progressData}%</div>
        </div>
        `
      }
    } else {
      progressBar.innerHTML = ""

      progressBarsCompletionAmount[computationType] = progressData

      if (progressData > 99.99) {
        
        progressBarColor[computationType] = "bg-success"
      
      }

      raceModeContainer.innerHTML = `
        <div class="row m-3"> 
          <div class="col-sm-3">
            <span> Compute Mode: Sync </span>
          </div>
          <div class="col-sm-9">
            <div class="progress">
              <div style="width: ${progressBarsCompletionAmount["race-sync"]}%" class="progress-bar ${progressBarColor["race-sync"]}" role="progressbar" aria-valuenow="${progressBarsCompletionAmount["race-sync"]}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressBarsCompletionAmount["race-sync"])}%</div>
            </div>
          </div>
        </div>
        <div class="row m-3"> 
        <div class="col-sm-3">
          <span> Compute Mode: Async </span>
        </div>
        <div class="col-sm-9">
          <div class="progress">
            <div style="width: ${progressBarsCompletionAmount["race-async"]}%" class="progress-bar ${progressBarColor["race-async"]}" role="progressbar" aria-valuenow="${progressBarsCompletionAmount["race-async"]}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressBarsCompletionAmount["race-async"])}%</div>
          </div>
        </div>
      </div>
      <div class="row m-3"> 
      <div class="col-sm-3">
        <span> Compute Mode: Cuda Native </span>
      </div>
      <div class="col-sm-9">
        <div class="progress">
          <div style="width: ${progressBarsCompletionAmount["race-cuda-native"]}%" class="progress-bar ${progressBarColor["race-cuda-native"]}" role="progressbar" aria-valuenow="${progressBarsCompletionAmount["race-cuda-native"]}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressBarsCompletionAmount["race-cuda-native"])}%</div>
        </div>
      </div>
    </div>

      `

    }

  }

  if (data.type === "image") {
    const { images, computationType } = data

    console.log(`Received: ${images}`)

    if (computationType != "race-mode") {

      for (const image of images) {
        const imageId = image.split("/").pop().replace(".jpg", "")
        console.log("Adding image with id", imageId)
        imageGalleryContent += `<img class="image-pad image" src="${image}" id="${imageId}" onclick="openLightBox(${imageId})">`
      }

      imageGallery.innerHTML = imageGalleryContent

    }


  }

})

sendWSMessage.onclick = () => {

  const { value: computationType } = document.getElementById("computation-type")
  console.log(`Beginning computation on ${computationType}`)
  ws.send(computationType)

  progressBar.innerHTML = ` 
  <div class="progress m-4">
     <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"> 0%</div>
  </div>
  `

  containerInfo.innerHTML = ""

}

selectElement.onchange = () => {
  const { value: computationType } = document.getElementById("computation-type")
  console.log(`Value changed to ${computationType}`)

  switch (computationType) {
    case "sync": {
      containerInfo.innerHTML = window.getSyncTemplate()
      break
    }
    case "async": {
      containerInfo.innerHTML = window.getAsyncTemplate()
      break
    }
    case "cuda-native": {
      containerInfo.innerHTML = window.getCudaNativeTemplate()
      break
    }
    case "race-mode": {
      containerInfo.innerHTML = window.getRaceModeTemplate()
      break
    }

  }

}


openLightBox = (imageId) => {
  const mainContainer = document.getElementById("main-container")
  const overlayImage = document.getElementById('overlay');
  const paddedImageId = ("0000" + imageId).slice(-4)
  const imageElement = `<img src="./images/full_res/${paddedImageId}.jpg" id="${imageId}-full-res" onclick="openLightBox(${imageId})">`
  overlayImage.innerHTML = imageElement
  overlayImage.style.display = 'block';
  mainContainer.setAttribute('class', 'blur');
  const currentImage = document.getElementById(`${imageId}-full-res`)
  currentImage.onclick = () => {
    const mainContainer = document.getElementById("main-container")
    const overlayImage = document.getElementById('overlay');
    overlayImage.style.display = 'none';
    mainContainer.removeAttribute('class', 'blur');

  }
}


console.log("JS is loaded.")