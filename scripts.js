
// Add an event listener to handle file selection
document.getElementById('videoInput').addEventListener('change', function() {
    const videoInput = document.getElementById('videoInput');
    const videoPlayer = document.getElementById('videoPlayer');
    // Check if a file is selected
    if(videoInput.files && videoInput.files[0]) {
        const fileURL = URL.createObjectURL(videoInput.files[0]);
        
        // Set the source and play video
        videoPlayer.src = fileURL;
    } else {
        alert('Please select a video file first.');
    }
});
function loadAndPlayVideo() {
    const videoPlayer = document.getElementById('videoPlayer');
    // Play the video
    if (videoPlayer.src) {
        videoPlayer.play();
    } else {
        alert('Please select a video file first.');
    }
}


function sendEmail() {
    window.location.href = "mailto:mobina.mb96@gmail.com";
  }