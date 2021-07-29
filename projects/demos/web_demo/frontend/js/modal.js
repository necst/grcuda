// Pictures array;
img_path = "resources/img/pictures/";
img_src = [
  // "F62A3701-2.jpg",
  // "F62A3691-2.jpg",
  // "IMG_0024-2.jpg",
  // "F62A6903-2.jpg",
  // "F62A3141-2.jpg",
  // "IMG_4404-2.jpg",
  // "F62A3684-2.jpg",
  // "F62A3668-2.jpg",
  // "F62A3660-2.jpg",
  // "F62A2211-2.jpg",
  // "IMG_4700-2.jpg",
  // "IMG_0859-2.jpg"
  "IMG_4700-2.jpg",
  "F62A4960.jpg",
  "F62A5104.jpg",
  "F62A4858.jpg",

  "F62A3141-2.jpg",
  "IMG_1788.jpg",
  "IMG_0024-2.jpg",
  "IMG_1836.jpg",
  "F62A3701-2.jpg",
  "IMG_0859-2.jpg",

  "F62A3660-2.jpg",
  "F62A3668-2.jpg",
  "F62A3684-2.jpg",
  "F62A3691-2.jpg",

  "F62A2211-2.jpg",
  "IMG_4404-2.jpg"
];
// Fix paths;
for (let i = 0; i < img_src.length; i++) {
  img_src[i] = img_path + img_src[i];
}
// Set the image sources;
var images = document.getElementsByClassName("image-zoomable");
for (let i = 0; i < images.length; i++) {
  images[i].src = img_src[i];
}

// Get the modal;
var modal = document.getElementById("modal-img-container");

// Current image in the modal;
curr_image = 0;

// Get the image and insert it inside the modal;
var modalImg = document.getElementById("modal-img");
for (let i = 0; i < images.length; i++) {
  images[i].onclick = function(){
    modal.style.display = "flex";
    modal.style.justifyContent = "center";
    modal.style.alignItems = "center";
    modalImg.src = this.src;
    curr_image = i;
  }
}

// Get the <span> element that closes the modal;
var span = document.getElementsByClassName("close-modal")[0];

// When the user clicks on <span> (x), close the modal;
span.onclick = function() {
  modal.style.display = "none";
  modal.style.justifyContent = "";
  modal.style.alignItems = "";
}

// Close zoomed images by clicking on background;
$('#modal-img-container').click(function() {
  modal.style.display = "none";
  modal.style.justifyContent = "";
  modal.style.alignItems = "";
});

$('#modal-img').click(function(e) {
  e.stopPropagation();
});

// Move across images by clicking on the ">" & "<";
$('#right-arrow').click(function(e) {
  curr_image = (curr_image + 1) % images.length;
  modalImg.src = images[curr_image].src;
  e.stopPropagation();
});
$('#left-arrow').click(function(e) {
  curr_image--;
  if (curr_image < 0) {
    curr_image = images.length - 1;
  }
  modalImg.src = images[curr_image].src;
  e.stopPropagation();
});