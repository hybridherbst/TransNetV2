import './style.css'
import { Input, BlobSource, VideoSampleSink, ALL_FORMATS } from 'mediabunny';

// Create worker
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

document.querySelector('#app').innerHTML = `
  <div class="container">
    <h1>TransNetV2 Web</h1>
    <p>Shot Boundary Detection in the Browser</p>
    
    <div class="upload-section">
      <input type="file" id="video-upload" accept="video/*" />
      <label for="video-upload" class="button">Choose Video</label>
    </div>

    <div class="duration-section">
      <label for="duration-input">Process first (seconds):</label>
      <input type="number" id="duration-input" value="6" min="1" />
      <button id="process-button" class="button">Process</button>
    </div>

    <div class="video-container">
      <video id="video-player" controls></video>
      <canvas id="overlay-canvas"></canvas>
    </div>

    <div id="status">Ready</div>
    <div id="results"></div>
  </div>
`

const videoUpload = document.getElementById('video-upload');
const videoPlayer = document.getElementById('video-player');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const canvas = document.getElementById('overlay-canvas');
const processButton = document.getElementById('process-button');
const ctx = canvas.getContext('2d');

let session = null;

let totalFrames = 0;

// Initialize process button as disabled
processButton.disabled = true;

async function init() {
    statusDiv.textContent = 'Loading model...';
    try {
        // Initialize worker
        worker.postMessage({ type: 'init' });
        
        worker.onmessage = function(e) {
            const { type, error, predictions, progress, total, provider } = e.data;
            if (type === 'init-done') {
                statusDiv.textContent = `Model loaded using ${provider.toUpperCase()}. Select a video.`;
            } else if (type === 'progress') {
                statusDiv.textContent = `Inference: ${progress} / ${total}`;
            } else if (type === 'done') {
                visualize(predictions, totalFrames);
                statusDiv.textContent = 'Done!';
                processButton.disabled = false;
            } else if (type === 'error') {
                statusDiv.textContent = 'Error: ' + error;
                console.error(error);
                processButton.disabled = false;
            }
        };
        
        worker.onerror = function(error) {
            statusDiv.textContent = 'Worker error: ' + error.message;
            console.error(error);
        };
    } catch (e) {
        statusDiv.textContent = 'Failed to load model: ' + e.message;
        console.error(e);
    }
}

init();

videoUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    videoPlayer.src = url;

    statusDiv.textContent = 'Video loaded. Click "Process" to analyze.';
    resultsDiv.innerHTML = '';
    processButton.disabled = false;
});

processButton.addEventListener('click', async () => {
    const file = videoUpload.files[0];
    if (!file) return;

    statusDiv.textContent = 'Processing video...';
    resultsDiv.innerHTML = '';
    processButton.disabled = true;

    try {
        await processVideo(file);
    } catch (err) {
        console.error(err);
        statusDiv.textContent = 'Error: ' + err.message;
        processButton.disabled = false;
    }
});

async function processVideo(file) {
    // Sliding window parameters
    const height = 27;
    const width = 48;
    const windowSize = 100;

    // Get user-specified duration
    const maxSeconds = parseInt(document.getElementById('duration-input').value) || 60;

    // Use Mediabunny to decode video frames
    statusDiv.textContent = 'Opening video with Mediabunny...';

    const input = new Input({
        source: new BlobSource(file),
        formats: ALL_FORMATS
    });

    const videoTrack = await input.getPrimaryVideoTrack();

    if (!videoTrack) {
        throw new Error('No video track found');
    }

    const decodable = await videoTrack.canDecode();
    if (!decodable) {
        throw new Error('Video track cannot be decoded');
    }

    statusDiv.textContent = 'Decoding and resizing frames...';

    // Create a canvas for resizing frames to 48x27
    const offscreen = new OffscreenCanvas(width, height);
    const offCtx = offscreen.getContext('2d', { willReadFrequently: true });

    const frames = [];

    // Use VideoSampleSink to decode frames
    const sink = new VideoSampleSink(videoTrack);

    // Get duration and limit to user input
    const fullDuration = await videoTrack.computeDuration();
    const limitedDuration = Math.min(fullDuration, maxSeconds);

    let frameCount = 0;
    // Iterate through all samples/frames in the video up to limited duration
    for await (const sample of sink.samples(0, limitedDuration)) {
        // Draw the frame to our offscreen canvas at 48x27
        sample.draw(offCtx, 0, 0, width, height);
        const frameData = offCtx.getImageData(0, 0, width, height);
        frames.push(frameData);

        // Close the sample to prevent memory leaks
        sample.close();

        frameCount++;
        if (frameCount % 100 === 0) {
            statusDiv.textContent = `Processed ${frameCount} frames...`;
        }
    }

    input.dispose();

    totalFrames = frames.length;
    statusDiv.textContent = `Extracted ${frames.length} frames. Running inference...`;

    // Send frames to worker for processing
    worker.postMessage({
        type: 'process',
        data: {
            frames: frames,
            height: height,
            width: width,
            windowSize: windowSize
        }
    });
}

function visualize(preds, totalFrames) {
    const threshold = 0.5;
    const shots = [];
    let start = 0;
    let prev = 0;

    for (let i = 0; i < preds.length; i++) {
        const p = preds[i] > threshold ? 1 : 0;
        if (prev === 1 && p === 0) start = i;
        if (prev === 0 && p === 1 && i !== 0) shots.push([start, i]);
        prev = p;
    }
    if (prev === 0) shots.push([start, preds.length - 1]);

    resultsDiv.innerHTML = `<h3>Detected Shots: ${shots.length}</h3>`;
    const list = document.createElement('ul');
    shots.forEach(shot => {
        const li = document.createElement('li');
        li.textContent = `Frame ${shot[0]} - ${shot[1]}`;
        li.onclick = async () => {
            // Clear any existing shot end listener
            if (videoPlayer._shotEndListener) {
                videoPlayer.removeEventListener('timeupdate', videoPlayer._shotEndListener);
            }
            
            // Pause first to stop any current playback
            videoPlayer.pause();
            
            // Set the start time
            videoPlayer.currentTime = shot[0] / 30; // Assuming 30fps
            
            // Add listener to stop at end of shot
            videoPlayer._shotEndListener = () => {
                if (videoPlayer.currentTime >= shot[1] / 30) {
                    // Remove listener first to prevent re-triggering
                    videoPlayer.removeEventListener('timeupdate', videoPlayer._shotEndListener);
                    // Only pause if not already paused (to avoid interrupting play)
                    if (!videoPlayer.paused) {
                        videoPlayer.pause();
                    }
                }
            };
            videoPlayer.addEventListener('timeupdate', videoPlayer._shotEndListener);
            
            // Small delay to ensure pause has taken effect
            await new Promise(resolve => setTimeout(resolve, 10));
            
            // Play the video and wait for it to actually start
            try {
                await videoPlayer.play();
                // Now that play has succeeded, we can safely pause later
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Error playing video:', error);
                }
                // Clean up listener on error
                if (videoPlayer._shotEndListener) {
                    videoPlayer.removeEventListener('timeupdate', videoPlayer._shotEndListener);
                }
            }
        };
        list.appendChild(li);
    });
    resultsDiv.appendChild(list);
}
