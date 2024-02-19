let playing = false;
let speed = 1.0;
let frameRate = 30;
let intervalID;

let videoElement = document.getElementById('videoElementId');

function updateVideoFeed() {
    fetch('/video_feed')
        .then(response => response.blob())
        .then(blob => {
            let url = URL.createObjectURL(blob);
            videoElement.src = url;
        })
        .catch(error => {
            console.error("Error fetching video frame:", error);
        });
}

function startInterval() {
    let intervalDuration = (1000 / frameRate) / speed;
    intervalID = setInterval(updateVideoFeed, intervalDuration);
}

function stopInterval() {
    clearInterval(intervalID);
}

function playVideo() {
    if (!playing) {
        playing = true;
        startInterval();
    }
}

function pauseVideo() {
    if (playing) {
        playing = false;
        stopInterval();
    }
}

function adjustSpeed(newSpeed) {
    setBackendSpeed(newSpeed);
    speed = newSpeed;
    if (playing) {
        stopInterval();
        startInterval();
    }
}

function increaseSpeed() {
    adjustSpeed(speed + 0.5);
}

function decreaseSpeed() {
    adjustSpeed(Math.max(speed - 0.5, 0.5));
}


function setBackendSpeed(newSpeed) {
    fetch(`/set_speed/${newSpeed}`)
        .then(response => response.text())
        .then(data => {
            console.log(data);
        })
        .catch(error => {
            console.error("Error setting speed:", error);
        });
}


// Function to update the date and time in the HTML
function updateDateTime() {
    fetch('/get_ocr_data')
        .then(response => response.json())
        .then(data => {
            // Assuming the first two items in the list are date and time
            document.querySelector('li#date').textContent = 'Date: ' + data.date;
            document.querySelector('li#time').textContent = 'Time: ' + data.time;
        })
        .catch(error => console.error('Error fetching OCR data:', error));
}

// Call updateDateTime() every second (1000 milliseconds)
setInterval(updateDateTime, 1000);