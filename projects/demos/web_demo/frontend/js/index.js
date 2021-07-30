const ws = new WebSocket("ws://localhost:8080")
const sendWSMessage = document.getElementById("btn-send-msg-ws")
const progressBar = document.getElementById("progress-bar")
const imageGallery = document.getElementById("images")
let imageGalleryContent = ""

ws.addEventListener("open", (evt) => {
  console.log("Connection to websocket is open at ws://localhost:8080")
})

ws.addEventListener("message", (evt) => {
  const data = JSON.parse(evt.data)

  if (data.type === "progress") {
    const { data: progressData } = JSON.parse(evt.data)
    // This is really bad, it forces a rerender of the whole element
    // at every progress message received.
    // Works for now but TODO: change this
    progressBar.innerHTML = `
    <label for="computing">Processing:</label>
    <progress id="file" value="${progressData}" max="100"></progress>
    `
  }
  if(data.type === "image") {
    const { images } = data
    
    console.log(`Received: ${images}`)

    for(const image of images) {
      const imageId = image.split("/").pop().replace(".jpg", "")
      console.log("Adding image with id", imageId)
      imageGalleryContent += `<img class="image-pad" src="${image}" id="${imageId}" onclick="openLightBox(${imageId})">`
    }

    imageGallery.innerHTML = imageGalleryContent

  }

})

sendWSMessage.onclick = () => {

  const { value: computationType } = document.getElementById("computation-type")
  console.log(`Beginning computation on ${computationType}`)
  ws.send(computationType)

  progressBar.innerHTML = `
    <label for="computing">Processing:</label>
    <progress id="file" value="0" max="100"></progress>
  `
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