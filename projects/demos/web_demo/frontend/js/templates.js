window.getSyncTemplate = () => `
<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <h3 class="display-4">Computation Mode: Sync</h3>
    <p class="lead">Description of the sync pipeline</p>
    <p>3 different processing channels blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/pipeline-async.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">First the image gets blurred </p>
    <p>blurring kernel blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/blurred.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">Then sharpened </p>
    <p>sharpening kernel blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/sharpened.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">Then combined </p>
    <p>combining blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/combined.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>
`
window.getAsyncTemplate = () => `
  <div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <h3 class="display-4">Computation Mode: Async</h3>
    <p class="lead">Description of the sync pipeline</p>
    <p>3 different processing channels blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/pipeline-async.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">First the image gets blurred </p>
    <p>blurring kernel blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/blurred.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">Then sharpened </p>
    <p>sharpening kernel blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/sharpened.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">Then combined </p>
    <p>combining blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/combined.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>
`
window.getCudaNativeTemplate = () => `
  <div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <h3 class="display-4">Computation Mode: Cuda Native</h3>
    <p class="lead">Description of the sync pipeline</p>
    <p>3 different processing channels blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/pipeline-async.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">First the image gets blurred </p>
    <p>blurring kernel blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/blurred.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">Then sharpened </p>
    <p>sharpening kernel blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/sharpened.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-9">
    <p class="lead">Then combined </p>
    <p>combining blablabla</p>
</div>
<div class="col-sm-3">
  <img src="./images/description/async/combined.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>
`

window.getRaceModeTemplate = () => `
      <div class="row">
          <div class="col-sm-12">
            <div id="container-info" class="">
              <div class="row" id="sync-pipeline-description">
                <div class="col-sm-9">
                    <h3 class="display-4">RACE MOTHERFUCKER RACE</h3>
                    <p class="lead">The gloves are off</p>
                    <p>Only one will win</p>
                </div>
              </div>
            </div>
        </div>
`


window.getImageLightBoxTemplate = (paddedImageId, imageId) => `<img src="./images/full_res/${paddedImageId}.jpg" id="${imageId}-full-res" onclick="openLightBox(${imageId})">`
window.getGalleryImageContentTemplate = (image, imageId) => `<img class="image-pad image" src="${image}" id="${imageId}" onclick="openLightBox(${imageId})">`

window.getProgressBarTemplate = (progressData, completed) => {
  if (!completed) {
    return `<div class="progress">
            <div style="width: ${progressData}%" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="${progressData}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressData)}%</div>
          </div>`
  } else {
    return `<div class="progress">
              <div style="width: ${progressData}%" class="progress-bar bg-success" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">100%</div>
            </div>`
  }
}
window.getProgressBarWithWrapperTemplate = (
  label, 
  progressBarsCompletionAmount, 
  progressBarRaceColor, 
  computationType
) => `
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

