const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('startButton');
const startCapture = document.getElementById('startCapture');
const trainButton = document.getElementById('trainButton');
const identityField = document.getElementById('identityField');
const result = document.getElementById('result');

let streaming = false;

const constraints = {
    video: true
};

navigator.mediaDevices.getUserMedia(constraints)
    .then(function (stream) {
        video.srcObject = stream;
        streaming = true;
    })
    .catch(function (err) {
        console.log('An error occurred: ' + err);
    });

startButton.addEventListener('click', function () {
    if (streaming) {
        const context = canvas.getContext('2d');
        canvas.width = video.width;
        canvas.height = video.height;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        var imgData = canvas.toDataURL('image/jpeg');
        
        // Send the image to the server for face recognition
        fetch('/recognize', {
            method: 'POST',
            body: JSON.stringify({ image: imgData }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            result.textContent = 'Recognition Result: ' + data.result;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

startCapture.addEventListener('click', function () {
    if (streaming) {
        const context = canvas.getContext('2d');
        canvas.width = video.width;
        canvas.height = video.height;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        let imgData = canvas.toDataURL('image/jpeg');

        console.log([imgData])
        // Send the image to the server for face recognition
        fetch('/capture', {
            method: 'POST',
            body: JSON.stringify({ image: imgData, identity: identityField.value }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            result.textContent = 'Capture Result: ' + data.result;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

trainButton.addEventListener('click', function () {
        
    // Send the image to the server for face recognition
    fetch('/train', {
        method: 'POST',
        body: {},
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        result.textContent = 'Capture Result: ' + data.result;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
