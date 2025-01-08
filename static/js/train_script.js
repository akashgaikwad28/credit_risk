document.addEventListener("DOMContentLoaded", () => {
    const progressElement = document.querySelector(".progress");
    const statusElement = document.getElementById("training-status");
    const messages = [
        "Initializing...",
        "Setting up the environment...",
        "Processing data...",
        "Training the model...",
        "Optimizing parameters...",
        "Finalizing..."
    ];

    let progress = 0;
    let messageIndex = 0;

    const updateProgress = () => {
        if (progress < 100) {
            progress += 20;
            progressElement.style.width = `${progress}%`;
            statusElement.textContent = messages[messageIndex];
            messageIndex = (messageIndex + 1) % messages.length;
        } else {
            clearInterval(progressInterval);
            statusElement.textContent = "Training Completed!";
        }
    };

    // Simulate training progress
    const progressInterval = setInterval(updateProgress, 2000);
});
