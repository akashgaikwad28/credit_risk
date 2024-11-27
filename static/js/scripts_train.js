document.addEventListener("DOMContentLoaded", () => {
    const progressBar = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");
    const statusMessage = document.getElementById("status-message");

    const statuses = [
        "Initializing...",
        "Loading data...",
        "Cleaning data...",
        "Optimizing model...",
        "Training model...",
        "Finalizing...",
        "Training complete!"
    ];

    let progress = 0;
    let statusIndex = 0;

    const updateProgress = () => {
        if (progress < 100) {
            progress += 20; // Increase progress
            statusIndex = Math.min(statusIndex + 1, statuses.length - 1);
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${progress}% Complete`;
            statusMessage.textContent = statuses[statusIndex];

            setTimeout(updateProgress, 1000); // Update every second
        } else {
            progressText.textContent = "100% Complete";
            statusMessage.textContent = "Model successfully trained!";
        }
    };

    updateProgress();
});
