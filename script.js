
// Function to load the selected image and display it in the preview
document.getElementById("intro-video").addEventListener("ended", function() {
    // Hide the video screen and show the main content
    document.getElementById("video-screen").style.display = "none";
});
function loadImage(event) {
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.innerHTML = ''; // Clear previous images
    const img = document.createElement('img');
    img.src = URL.createObjectURL(event.target.files[0]);
    img.alt = "Selected Bird Image";
    img.style.width = '100%'; // Adjust width as needed
    img.style.height = 'auto'; // Maintain aspect ratio
    imagePreview.appendChild(img);
}
function loadImage(event) {
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.innerHTML = '';  // Clear previous preview
    const img = document.createElement('img');
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.width = '300px';  // Preview image size
    imagePreview.appendChild(img);
}

// Function to identify the bird
function identifyBird() {
    const fileInput = document.getElementById('birdImage');
    const file = fileInput.files[0];
    
    // Check if a file is selected
    if (!file) {
        alert('Please upload an image first!');
        return;
    }
    
    // Prepare FormData to send the image to the backend
    const formData = new FormData();
    formData.append('file', file);
    
    // Send the image to the Flask backend for prediction
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            // If the response is not OK, throw an error
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const resultDiv = document.getElementById('birdIdentificationResult');
        
        // Log the response data to the console for debugging
        console.log("Response from API:", data);
        
        // Check if we received valid prediction data from the backend
        if (data.predicted_class && data.confidence !== undefined) {
            resultDiv.innerHTML = `<h3>Predicted Bird: ${data.predicted_class}</h3><p>Confidence: ${data.confidence.toFixed(2)}%</p>`;
        } else if (data.error) {
            resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = '<p>Unexpected error occurred. Please try again later.</p>';
        }
    })
    .catch(error => {
        // Log error details for debugging
        console.error('Error:', error);
        const resultDiv = document.getElementById('birdIdentificationResult');
        resultDiv.innerHTML = '<p>Error: Unable to identify bird.</p>';
    });
}



// Function to load and play audio preview
function loadAudio(event) {
    const audioPreview = document.getElementById('birdAudioPreview');
    audioPreview.src = URL.createObjectURL(event.target.files[0]);
    audioPreview.load(); // Load the audio
}

// Function to analyze bird audio
function analyzeBirdAudio() {
    const audioPreview = document.getElementById('birdAudioPreview');
    const resultDiv = document.getElementById('audioAnalysisResult');
    if (audioPreview.src) {
        // Mock analysis result
        resultDi// Function to load the selected image
        function loadImage(event) {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.innerHTML = '';
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = 'Selected Bird';
                img.style.maxWidth = '100%';
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
        
        // Function to identify the bird (placeholder function)
        function identifyBird() {
            const resultDiv = document.getElementById('birdIdentificationResult');
            resultDiv.innerHTML = '<p>Identifying bird...</p>';
            // Add your bird identification logic here
            setTimeout(() => {
                resultDiv.innerHTML = '<p>Bird identified: Example Bird</p>';
            }, 2000);
        }
        
        // Function to load the selected audio
        function loadAudio(event) {
            const audioPreview = document.getElementById('birdAudioPreview');
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                audioPreview.src = e.target.result;
                audioPreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
        
        // Function to analyze bird audio (placeholder function)
        function analyzeBirdAudio() {
            const resultDiv = document.getElementById('audioAnalysisResult');
            resultDiv.innerHTML = '<p>Analyzing audio...</p>';
            // Add your audio analysis logic here
            setTimeout(() => {
                resultDiv.innerHTML = '<p>Audio analysis complete: Example Bird Sound</p>';
            }, 2000);
        }
        
        // Remove any click event listeners from bird cards
        document.querySelectorAll('.bird-card').forEach(card => {
            card.addEventListener('click', event => {
                event.stopPropagation();
            });
        });
        v.innerHTML = '<h3>Audio Analysis Result</h3><p>This audio is likely a call from a common sparrow.</p>';
    } else {
        resultDiv.innerHTML = '<p>Please upload an audio file first.</p>';
    }
}

// Simple search functionality
const searchInput = document.getElementById('searchInput');
searchInput.addEventListener('input', function() {
    const filter = searchInput.value.toLowerCase();
    birdCards.forEach(card => {
        const cardTitle = card.querySelector('h3').innerText.toLowerCase();
        card.style.display = cardTitle.includes(filter) ? 'block' : 'none'; // Show or hide cards based on search
    });
});
// script.js


