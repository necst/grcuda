const ws = new WebSocket("ws://localhost:8080")
const sendWSMessage = document.getElementById("btn-send-msg-ws")
const progressBar = document.getElementById("progress-bar")
const imageGallery = document.getElementById("images")
const selectElement = document.getElementById("computation-type")
const containerInfo = document.getElementById("container-info")
const raceModeContainer = document.getElementById("race-mode")

const progressBarSync = document.getElementById("progress-bar-sync")
const progressBarAsync = document.getElementById("progress-bar-async")
const progressBarCudaNative = document.getElementById("progress-bar-cuda-native")

const imageGallerySync = document.getElementById("image-gallery-sync")
const imageGalleryAsync = document.getElementById("image-gallery-async")
const imageGalleryCudaNative = document.getElementById("image-gallery-cuda-native")

const progressBarsRace = {
  "race-sync": progressBarSync,
  "race-async": progressBarAsync,
  "race-cuda-native": progressBarCudaNative
}

const imageGalleriesRace = {
  "race-sync": imageGallerySync,
  "race-async": imageGalleryAsync,
  "race-cuda-native": imageGalleryCudaNative
}

const imageGalleriesRaceContent = {
  "race-sync": "",
  "race-async": "",
  "race-cuda-native": ""
}

const progressBarRaceColor = {
  "race-sync": "progress-bar-striped",
  "race-async": "progress-bar-striped",
  "race-cuda-native": "progress-bar-striped"
}

const labelMap = {
  "race-sync": "Sync",
  "race-async": "Async",
  "race-cuda-native": "Cuda Native",
  "sync": "Sync",
  "async": "Async",
  "cuda-native": "Cuda Native"
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
    processProgressMessage(evt)
  }

  if (data.type === "image") {
    processImageMessage(evt)
  }

})

sendWSMessage.onclick = () => {

  const { value: computationType } = document.getElementById("computation-type")
  console.log(`Beginning computation on ${computationType}`)
  ws.send(computationType)

  progressBar.innerHTML = ` 
  <div class="progress">
     <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"> 0%</div>
  </div>
  `

  containerInfo.innerHTML = ""

}

const clearAll = () => {
  progressBar.innerHTML = ""
  imageGallery.innerHTML = ""

  // for(const k of imageGalleriesRace) {
  //   imageGalleriesRace[k].innerHTML = ""
  // }
}

selectElement.onchange = () => {
  const { value: computationType } = document.getElementById("computation-type")

  // Remove progressbar if present
  clearAll()

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

const processImageMessage = (evt) => {
  const data = JSON.parse(evt.data)
  const { images, computationType } = data

    console.log(`Received: ${images}`)

    if (!computationType.includes("race")) {

      for (const image of images) {
        const imageId = image.split("/").pop().replace(".jpg", "")
        imageGalleryContent += `<img class="image-pad image" src="${image}" id="${imageId}" onclick="openLightBox(${imageId})">`
      }

      imageGallery.innerHTML = imageGalleryContent

    } else {

      imageGalleriesRaceContent[computationType] = images.reduce((rest, image) => {
        const imageId = image.split("/").pop().replace(".jpg", "")
        const imgContent = `<img class="image-pad image" src="${image}" id="${imageId}" onclick="openLightBox(${imageId})">`
        return rest + imgContent
      }, imageGalleriesRaceContent[computationType])

      imageGalleriesRace[computationType].innerHTML = imageGalleriesRaceContent[computationType]
    }
}

const processProgressMessage = (evt) => {
  const data = JSON.parse(evt.data)
  const { data: progressData, computationType } = data

  if (!computationType.includes("race")) {
    if (progressData < 99.99) {
      progressBar.innerHTML = `
        <div class="progress">
          <div style="width: ${progressData}%" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="${progressData}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressData)}%</div>
        </div>
      `
    } else {
      progressBar.innerHTML = `
        <div class="progress">
          <div style="width: ${progressData}%" class="progress-bar bg-success" role="progressbar" aria-valuenow="${progressData}" aria-valuemin="0" aria-valuemax="100">${progressData}%</div>
        </div>
        `
    }

  } else {

    progressBar.innerHTML = ""

    progressBarsCompletionAmount[computationType] = progressData

    if (progressData > 99.99) {

      progressBarRaceColor[computationType] = "bg-success"

    }

    const label = labelMap[computationType]
    progressBarsRace[computationType].innerHTML = `
      <div class="m-3">
        <div class="row">
          <div class="col-sm-12 mb-3">
            <span> Compute Mode: ${label} </span>
          </div>
        </div>
        <div class="row">
          <div class="col-sm-12">
            <div class="progress">
              <div style="width: ${progressBarsCompletionAmount[computationType]}%" class="progress-bar ${progressBarRaceColor[computationType]}" role="progressbar" aria-valuenow="${progressBarsCompletionAmount[computationType]}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressBarsCompletionAmount[computationType])}%</div>
            </div>
          </div>
        </div>
      </div>  
    `
  }

}




console.log("JS is loaded.")