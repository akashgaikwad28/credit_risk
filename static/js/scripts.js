// JavaScript for handling drag-and-drop functionality
const dropzone = document.querySelector('.dropzone');
const fileInput = document.getElementById('file-upload');
const fileText = dropzone.querySelector('p');

// Handle file selection on click
dropzone.addEventListener('click', () => {
    fileInput.click();
});

// Handle drag over effect
dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropzone.classList.add('dragging');
    fileText.textContent = "Release to upload the file";
});

// Handle drag leave effect
dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragging');
    fileText.textContent = "Drag your CSV file here or click to select";
});

// Handle file drop
dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragging');
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        fileText.textContent = files[0].name; // Display the file name
    }
});
