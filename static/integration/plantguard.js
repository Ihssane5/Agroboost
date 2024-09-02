document.getElementById('get-diag').addEventListener('click', async function(event) {
    event.preventDefault();
    const imageInput = document.getElementById('image-upload');
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    const response = await fetch('/plantguard', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    console.log(result)
    document.getElementById('generated-text').innerHTML = result;
});
