// ObjectLens 3D - live webcam panel: MediaPipe hands + TF.js COCO-SSD.
const video = document.getElementById('cam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const nocamEl = document.getElementById('nocam');
const chkHands = document.getElementById('chkHands');
const chkObjects = document.getElementById('chkObjects');

let stream = null, hands = null, cocoModel = null;
let latestHands = null, latestBoxes = [], detecting = false;

function setStatus(t) { statusEl.textContent = t; }

async function startCamera() {
  const btn = document.getElementById('btnCam');
  btn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }, audio: false });
  } catch (e) {
    setStatus('camera unavailable: ' + e.name);
    btn.disabled = false;
    return;
  }
  video.srcObject = stream;
  await video.play();
  nocamEl.style.display = 'none';
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  setStatus('camera live - loading models…');
  initHands();
  cocoSsd.load().then(m => { cocoModel = m; setStatus('camera live - models ready'); })
              .catch(() => setStatus('camera live - object model failed to load'));
  requestAnimationFrame(drawLoop);
  setInterval(detectObjects, 250);
}

function initHands() {
  hands = new Hands({ locateFile: f =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
  hands.setOptions({ maxNumHands: 2, modelComplexity: 1,
                     minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  hands.onResults(r => { latestHands = r; });
  const pump = async () => {
    if (stream && chkHands.checked) { try { await hands.send({ image: video }); } catch (e) {} }
    if (stream) requestAnimationFrame(pump);
  };
  pump();
}

async function detectObjects() {
  if (!cocoModel || !stream || detecting || !chkObjects.checked) return;
  detecting = true;
  try { latestBoxes = await cocoModel.detect(video); } catch (e) {}
  detecting = false;
}

function drawLoop() {
  if (!stream) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // object boxes
  if (chkObjects.checked) {
    ctx.lineWidth = 3; ctx.font = '600 20px Segoe UI, sans-serif';
    for (const b of latestBoxes) {
      const [x, y, w, h] = b.bbox;
      ctx.strokeStyle = '#ffb020'; ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = '#ffb020';
      ctx.fillText(`${b.class} ${(b.score * 100).toFixed(0)}%`, x + 4, y + 22);
    }
  }
  // hand landmarks
  if (chkHands.checked && latestHands && latestHands.multiHandLandmarks) {
    for (const lm of latestHands.multiHandLandmarks) {
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: '#00e69a', lineWidth: 3 });
      drawLandmarks(ctx, lm, { color: '#ff7a1a', radius: 4 });
    }
  }
  requestAnimationFrame(drawLoop);
}

document.getElementById('btnCam').addEventListener('click', startCamera);
