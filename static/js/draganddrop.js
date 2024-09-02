const imageDropzone = document.getElementById('image-dropzone');
const textDropzone = document.getElementById('text-dropzone');
const imageUploadInput = document.getElementById('image-upload');
const uploadButton = document.getElementById('upload-button');

function handleDragOver(event) {
    event.preventDefault();
    this.classList.add('dragover');
}

function handleDragLeave(event) {
    this.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    this.classList.remove('dragover');
    const files = event.dataTransfer.files;

    if (this === imageDropzone && files.length > 0) {
        const file = files[0];
        displayImage(file);
    } else if (this === textDropzone) {
        const data = event.dataTransfer.getData('text');
        textDropzone.innerHTML = `<p>${data}</p>`;
    }
}

function displayImage(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        //img.id = "upload-image"
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';
        imageDropzone.innerHTML = '';
        imageDropzone.appendChild(img);
    };
    reader.readAsDataURL(file);
}

imageDropzone.addEventListener('dragover', handleDragOver);
imageDropzone.addEventListener('dragleave', handleDragLeave);
imageDropzone.addEventListener('drop', handleDrop);

/*textDropzone.addEventListener('dragover', handleDragOver);
textDropzone.addEventListener('dragleave', handleDragLeave);
textDropzone.addEventListener('drop', handleDrop);*/

document.addEventListener('dragstart', function (event) {
    if (event.target && event.target.tagName === 'P') {
        event.dataTransfer.setData('text', event.target.innerText);
    }
});

uploadButton.addEventListener('click', function () {
    imageUploadInput.click();
});

imageUploadInput.addEventListener('change', function () {
    const file = imageUploadInput.files[0];
    if (file) {
        displayImage(file);
    }
});
