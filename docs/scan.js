// ObjectLens 3D - "Scan your own video": fully in-browser visual hull.
// Mirrors the Python pipeline (objectlens/segment.py + hull.py):
//   1. sample N frames evenly across one full 360-degree turn,
//   2. segment each silhouette by background-colour distance
//      (background = median of the four corner patches, like measure.py),
//   3. keep the largest blob, fill holes,
//   4. derive the rotation axis (median centroid column) and ground row,
//   5. carve a voxel visual hull with orthographic projection,
//   6. show the surface points in three.js and export them as PLY.
// If a printed ArUco marker (OpenCV DICT_4X4_50, the Python pipeline's
// dictionary - vendored js-aruco2 under vendor/) is visible in any sampled
// frame, the most frontal detection calibrates px-per-mm and dimensions
// come out in TRUE millimetres; the marker quad is cut out of the masks
// so it can never merge with the object. Without a marker, dimensions
// stay RELATIVE (height = 1.00). Nothing is uploaded - all local.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id);
const dropEl = $('drop'), fileEl = $('fileVideo'), fileNameEl = $('fileName');
const viewsEl = $('inViews'), threshEl = $('inThresh'), threshValEl = $('threshVal');
const startEl = $('inStart'), endEl = $('inEnd'), markerMmEl = $('inMarkerMm');
const buildEl = $('btnBuild'), plyEl = $('btnPly');
const statusEl = $('scanStatus'), errorEl = $('scanError'), calibEl = $('calibStatus');
const thumbsEl = $('thumbs'), viewerEl = $('scanViewer'), dimsEl = $('scanDims');
const relNoteEl = $('relNote');
const REL_NOTE_DEFAULT = relNoteEl.innerHTML;

const GRID = 64;          // voxel grid (64^3) - fast enough in JS
const MAX_W = 640;        // processing width cap for speed

// hidden working video element (kept in the DOM so the decoder stays active)
const vid = document.createElement('video');
vid.muted = true; vid.playsInline = true; vid.preload = 'auto';
vid.style.cssText = 'position:absolute;width:2px;height:2px;opacity:0;pointer-events:none';
document.getElementById('scan').appendChild(vid);
const fc = document.createElement('canvas');           // frame canvas
const fctx = fc.getContext('2d', { willReadFrequently: true });

let videoURL = null;
let lastPoints = null;    // Float32Array, relative units (height = 1)
let lastCalib = null;     // null, or {pxPerMm, heightPx, frames, markerId}
let three = null;         // three.js bits, created on first result
let pointsObj = null;

const tick = () => new Promise(r => setTimeout(r, 0));
const median = arr => { const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };

function setStatus(t) { statusEl.textContent = t; }
function showError(msg) { errorEl.textContent = msg; errorEl.hidden = false; }
function hideError() { errorEl.hidden = true; }

// ---------------------------------------------------------------- file input
function handleFile(file) {
  if (!file) return;
  if (!/^video\//.test(file.type) && !/\.(mp4|webm|mov|m4v|mkv)$/i.test(file.name)) {
    showError('That does not look like a video file — please choose an MP4 or WebM clip.');
    return;
  }
  hideError();
  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = URL.createObjectURL(file);
  buildEl.disabled = true;
  fileNameEl.textContent = `${file.name} — loading…`;
  setStatus('loading video…');
  vid.src = videoURL;
}
vid.addEventListener('loadedmetadata', () => {
  if (!isFinite(vid.duration) || !vid.videoWidth) return;
  fileNameEl.textContent =
    `${vid.duration.toFixed(1)} s · ${vid.videoWidth}×${vid.videoHeight} — processed locally, never uploaded.`;
  startEl.value = 0;
  endEl.value = vid.duration.toFixed(1);
  buildEl.disabled = false;
  setStatus('video ready — set the turn range if needed, then Build');
});
vid.addEventListener('error', () => {
  if (!vid.src) return;
  showError('This video could not be decoded by your browser. Try an MP4 (H.264) or WebM file.');
  setStatus('video failed to load');
});

fileEl.addEventListener('change', () => handleFile(fileEl.files[0]));
$('btnBrowse').addEventListener('click', e => { e.stopPropagation(); fileEl.click(); });
dropEl.addEventListener('click', () => fileEl.click());
dropEl.addEventListener('dragover', e => { e.preventDefault(); dropEl.classList.add('over'); });
dropEl.addEventListener('dragleave', () => dropEl.classList.remove('over'));
dropEl.addEventListener('drop', e => {
  e.preventDefault(); dropEl.classList.remove('over');
  handleFile(e.dataTransfer.files && e.dataTransfer.files[0]);
});
threshEl.addEventListener('input', () => { threshValEl.textContent = threshEl.value; });

// ------------------------------------------------------------------- seeking
function seekTo(t) {
  return new Promise((resolve, reject) => {
    if (Math.abs(vid.currentTime - t) < 0.01 && vid.readyState >= 2) { resolve(); return; }
    const timer = setTimeout(() => { cleanup(); reject(new Error(
      'The browser could not seek through this video — it may be corrupted or use an unsupported codec.')); }, 15000);
    const on = () => { cleanup(); resolve(); };
    const cleanup = () => { clearTimeout(timer); vid.removeEventListener('seeked', on); };
    vid.addEventListener('seeked', on);
    vid.currentTime = t;
  });
}

// ------------------------------------------------------------ ArUco markers
// Vendored js-aruco2 (vendor/js-aruco2/) as classic scripts -> window.AR.
// The registered dictionary 'ARUCO_4X4_50' is bit-for-bit OpenCV's
// DICT_4X4_50 (see the header of vendor/js-aruco2/aruco_4x4_50.js).
let arDetector = null, arBroken = false;
function getDetector() {
  if (arDetector || arBroken) return arDetector;
  try {
    if (window.AR && AR.Detector) arDetector = new AR.Detector({ dictionaryName: 'ARUCO_4X4_50' });
    else arBroken = true;
  } catch (e) { arBroken = true; }
  return arDetector;
}

// Score one detected quad: squareness = 16*area / perimeter^2 (1.0 for a
// perfect square, less as perspective slant grows) with the mean edge.
function quadStats(c) {
  let per = 0; const edges = [];
  for (let i = 0; i < 4; i++) {
    const a = c[i], b = c[(i + 1) % 4];
    const e = Math.hypot(b.x - a.x, b.y - a.y);
    edges.push(e); per += e;
  }
  let area = 0;
  for (let i = 0; i < 4; i++) {
    const a = c[i], b = c[(i + 1) % 4];
    area += a.x * b.y - b.x * a.y;
  }
  area = Math.abs(area) / 2;
  return { meanEdge: per / 4, squareness: per > 0 ? 16 * area / (per * per) : 0, area };
}

// Run detection on one already-drawn frame (RGBA ImageData).
// Returns { quads, best } - every marker quad (for mask exclusion) and the
// single most frontal detection of this frame, or null.
function detectMarkers(imgData) {
  const det = getDetector();
  if (!det) return { quads: [], best: null };
  let found = [];
  try { found = det.detect({ width: imgData.width, height: imgData.height, data: imgData.data }) || []; }
  catch (e) { return { quads: [], best: null }; }
  const quads = [], scored = [];
  for (const mk of found) {
    if (!mk.corners || mk.corners.length !== 4) continue;
    const st = quadStats(mk.corners);
    quads.push(mk.corners);
    scored.push({ id: mk.id, corners: mk.corners, ...st });
  }
  scored.sort((a, b) => (b.squareness - a.squareness) || (b.area - a.area));
  return { quads, best: scored[0] || null };
}

// Cut marker quads out of a raw mask (before the largest-component step),
// so a marker leaning against the object can never join its silhouette.
// Quads are convex; corners are pushed 3% away from the centroid first.
function excludeQuads(mask, w, h, quads) {
  for (const q of quads) {
    const gx = (q[0].x + q[1].x + q[2].x + q[3].x) / 4;
    const gy = (q[0].y + q[1].y + q[2].y + q[3].y) / 4;
    const ex = q.map(p => ({ x: gx + (p.x - gx) * 1.06, y: gy + (p.y - gy) * 1.06 }));
    const x0 = Math.max(0, Math.floor(Math.min(...ex.map(p => p.x))));
    const x1 = Math.min(w - 1, Math.ceil(Math.max(...ex.map(p => p.x))));
    const y0 = Math.max(0, Math.floor(Math.min(...ex.map(p => p.y))));
    const y1 = Math.min(h - 1, Math.ceil(Math.max(...ex.map(p => p.y))));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      let pos = true, neg = true;
      for (let i = 0; i < 4; i++) {
        const a = ex[i], b = ex[(i + 1) % 4];
        const cr = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
        if (cr > 0) neg = false; else if (cr < 0) pos = false;
      }
      if (pos || neg) mask[y * w + x] = 0;
    }
  }
}

// -------------------------------------------------------------- segmentation
// Same recipe as the Python pipeline: background colour from the corners,
// colour-distance threshold, largest connected component, hole filling.
// `data` is the frame's RGBA bytes; `quads` are marker quads to cut out.
function segmentFrame(data, w, h, thresh, quads) {
  // background colour: mean of each corner patch, median across patches
  const p = Math.max(4, Math.floor(Math.min(w, h) / 20));
  const patches = [[0, 0], [w - p, 0], [0, h - p], [w - p, h - p]];
  const means = patches.map(([px, py]) => {
    let r = 0, g = 0, b = 0;
    for (let y = py; y < py + p; y++) for (let x = px; x < px + p; x++) {
      const i = (y * w + x) * 4; r += data[i]; g += data[i + 1]; b += data[i + 2];
    }
    const n = p * p; return [r / n, g / n, b / n];
  });
  const bg = [0, 1, 2].map(c => means.map(m => m[c]).sort((a, b) => a - b)[1]);

  const mask = new Uint8Array(w * h);
  const t2 = thresh * thresh;
  for (let i = 0, px = 0; px < w * h; px++, i += 4) {
    const dr = data[i] - bg[0], dg = data[i + 1] - bg[1], db = data[i + 2] - bg[2];
    if (dr * dr + dg * dg + db * db > t2) mask[px] = 1;
  }
  if (quads && quads.length) excludeQuads(mask, w, h, quads);
  const clean = largestComponent(mask, w, h);
  fillHoles(clean, w, h);
  // stats in one pass
  let count = 0, minX = w, maxX = -1, minY = h, maxY = -1, sumX = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (clean[y * w + x]) { count++; sumX += x; if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
  }
  return { mask: clean, count, frac: count / (w * h), minX, maxX, minY, maxY,
           centroidX: count ? sumX / count : 0 };
}

function largestComponent(mask, w, h) {
  const label = new Int32Array(w * h).fill(-1);
  const stack = new Int32Array(w * h);
  let best = -1, bestSize = 0, cur = 0;
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || label[s] >= 0) continue;
    let sp = 0, size = 0;
    stack[sp++] = s; label[s] = cur;
    while (sp > 0) {
      const q = stack[--sp]; size++;
      const x = q % w, y = (q / w) | 0;
      if (x > 0     && mask[q - 1] && label[q - 1] < 0) { label[q - 1] = cur; stack[sp++] = q - 1; }
      if (x < w - 1 && mask[q + 1] && label[q + 1] < 0) { label[q + 1] = cur; stack[sp++] = q + 1; }
      if (y > 0     && mask[q - w] && label[q - w] < 0) { label[q - w] = cur; stack[sp++] = q - w; }
      if (y < h - 1 && mask[q + w] && label[q + w] < 0) { label[q + w] = cur; stack[sp++] = q + w; }
    }
    if (size > bestSize) { bestSize = size; best = cur; }
    cur++;
  }
  const out = new Uint8Array(w * h);
  if (best >= 0) for (let i = 0; i < w * h; i++) out[i] = label[i] === best ? 1 : 0;
  return out;
}

function fillHoles(mask, w, h) {
  // flood the background from the borders; unreached background = a hole
  const seen = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const push = i => { if (!mask[i] && !seen[i]) { seen[i] = 1; stack[sp++] = i; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp > 0) {
    const q = stack[--sp];
    const x = q % w, y = (q / w) | 0;
    if (x > 0) push(q - 1);
    if (x < w - 1) push(q + 1);
    if (y > 0) push(q - w);
    if (y < h - 1) push(q + w);
  }
  for (let i = 0; i < w * h; i++) if (!mask[i] && !seen[i]) mask[i] = 1;
}

// --------------------------------------------------------------- thumbnails
function addThumb(seg, w, h, angle) {
  const tw = 156, th = Math.max(1, Math.round(tw * h / w));
  const c = document.createElement('canvas');
  c.width = tw; c.height = th;
  const cx2 = c.getContext('2d');
  cx2.drawImage(fc, 0, 0, tw, th);
  const img = cx2.getImageData(0, 0, tw, th), d = img.data;
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
    const mx = Math.min(w - 1, (x / tw * w) | 0), my = Math.min(h - 1, (y / th * h) | 0);
    const i = (y * tw + x) * 4;
    if (seg.mask[my * w + mx]) {           // object: green tint
      d[i] = d[i] * 0.35 + 0; d[i + 1] = d[i + 1] * 0.35 + 150; d[i + 2] = d[i + 2] * 0.35 + 60;
    } else {                               // background: dim it
      d[i] *= 0.45; d[i + 1] *= 0.45; d[i + 2] *= 0.45;
    }
  }
  cx2.putImageData(img, 0, 0);
  const fig = document.createElement('figure');
  fig.appendChild(c);
  const cap = document.createElement('figcaption');
  cap.textContent = `${Math.round(angle)}° · ${(seg.frac * 100).toFixed(0)}%`;
  fig.appendChild(cap);
  thumbsEl.appendChild(fig);
}

// ------------------------------------------------------------------- carving
// Same geometry as hull.py: orthographic projection,
//   u = x*cos(t) + z*sin(t) + cx ,  v = cy - y   (pixel units, scale = 1)
async function carveHull(masks, w, h, cx, cy, onProgress) {
  let half = 0, top = 0;
  for (const m of masks) {
    half = Math.max(half, m.maxX - cx, cx - m.minX);
    top = Math.max(top, cy - m.minY);
  }
  const extent = 2 * Math.max(half, top / 2) * 1.15;   // px, cube side
  const vox = extent / GRID;
  const G = GRID, N = G * G * G;
  const X = new Float32Array(N), Y = new Float32Array(N), Z = new Float32Array(N);
  for (let ix = 0; ix < G; ix++) for (let iy = 0; iy < G; iy++) for (let iz = 0; iz < G; iz++) {
    const i = (ix * G + iy) * G + iz;
    X[i] = (ix + 0.5) * vox - extent / 2;
    Y[i] = (iy + 0.5) * vox;
    Z[i] = (iz + 0.5) * vox - extent / 2;
  }
  let alive = new Int32Array(N);
  for (let i = 0; i < N; i++) alive[i] = i;
  let aliveCount = N;
  for (let v = 0; v < masks.length; v++) {
    const th = v * 2 * Math.PI / masks.length;
    const cos = Math.cos(th), sin = Math.sin(th), mask = masks[v].mask;
    let write = 0;
    for (let r = 0; r < aliveCount; r++) {
      const i = alive[r];
      const u = Math.round(X[i] * cos + Z[i] * sin + cx);
      const vv = Math.round(cy - Y[i]);
      if (u >= 0 && u < w && vv >= 0 && vv < h && mask[vv * w + u]) alive[write++] = i;
    }
    aliveCount = write;
    onProgress(v + 1, masks.length);
    await tick();
    if (!aliveCount) break;
  }
  // filled set + bbox + surface points
  const filled = new Uint8Array(N);
  for (let r = 0; r < aliveCount; r++) filled[alive[r]] = 1;
  let minIx = G, maxIx = -1, minIy = G, maxIy = -1, minIz = G, maxIz = -1;
  const pts = [];
  for (let ix = 0; ix < G; ix++) for (let iy = 0; iy < G; iy++) for (let iz = 0; iz < G; iz++) {
    const i = (ix * G + iy) * G + iz;
    if (!filled[i]) continue;
    if (ix < minIx) minIx = ix; if (ix > maxIx) maxIx = ix;
    if (iy < minIy) minIy = iy; if (iy > maxIy) maxIy = iy;
    if (iz < minIz) minIz = iz; if (iz > maxIz) maxIz = iz;
    const surf = ix === 0 || !filled[i - G * G] || ix === G - 1 || !filled[i + G * G] ||
                 iy === 0 || !filled[i - G]     || iy === G - 1 || !filled[i + G]     ||
                 iz === 0 || !filled[i - 1]     || iz === G - 1 || !filled[i + 1];
    if (surf) pts.push(X[i], Y[i], Z[i]);
  }
  if (!aliveCount || maxIy < 0) return { points: null, voxelCount: 0 };
  const widthPx = (maxIx - minIx + 1) * vox;
  const heightPx = (maxIy - minIy + 1) * vox;
  const depthPx = (maxIz - minIz + 1) * vox;
  const s = 1 / heightPx;                       // normalise: height = 1.00
  const cxRel = (minIx + maxIx + 1) / 2 * vox - extent / 2;
  const czRel = (minIz + maxIz + 1) / 2 * vox - extent / 2;
  const baseY = minIy * vox;
  const rel = new Float32Array(pts.length);
  for (let i = 0; i < pts.length; i += 3) {
    rel[i] = (pts[i] - cxRel) * s;
    rel[i + 1] = (pts[i + 1] - baseY) * s;
    rel[i + 2] = (pts[i + 2] - czRel) * s;
  }
  return { points: rel, voxelCount: aliveCount,
           dims: { x: widthPx * s, y: 1, z: depthPx * s }, voxRel: vox * s,
           dimsPx: { x: widthPx, y: heightPx, z: depthPx } };
}

// -------------------------------------------------------------------- viewer
function showResult(points, voxRel) {
  viewerEl.hidden = false;
  if (!three) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, viewerEl.clientWidth / viewerEl.clientHeight, 0.01, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    viewerEl.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const grid = new THREE.GridHelper(2.4, 24, 0x1e2c44, 0x14203a);
    scene.add(grid);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.1;
    controls.target.set(0, 0.48, 0);
    camera.position.set(1.05, 0.8, 1.6);
    renderer.domElement.addEventListener('pointerdown', () => { controls.autoRotate = false; });
    three = { scene, camera, renderer, controls };
    (function loop() {
      requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    })();
    addEventListener('resize', () => {
      camera.aspect = viewerEl.clientWidth / viewerEl.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
    });
  }
  if (pointsObj) {
    three.scene.remove(pointsObj);
    pointsObj.geometry.dispose(); pointsObj.material.dispose();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(points, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x2effa8, size: Math.max(0.012, voxRel * 1.6), sizeAttenuation: true });
  pointsObj = new THREE.Points(geo, mat);
  three.scene.add(pointsObj);
}

// ---------------------------------------------------------------------- PLY
plyEl.addEventListener('click', () => {
  if (!lastPoints) return;
  const n = lastPoints.length / 3;
  // calibrated: rel points are height-normalised, so mm = rel * height in mm
  const f = lastCalib ? lastCalib.heightPx / lastCalib.pxPerMm : 1;
  const comment = lastCalib
    ? `comment ObjectLens 3D browser scan - units are MILLIMETRES (ArUco-calibrated, ${lastCalib.pxPerMm.toFixed(2)} px/mm)`
    : 'comment ObjectLens 3D browser scan - RELATIVE units, height = 1.0 (no ArUco marker in video)';
  let out = 'ply\nformat ascii 1.0\n' + comment + '\n' +
    `element vertex ${n}\nproperty float x\nproperty float y\nproperty float z\n` +
    'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
  for (let i = 0; i < lastPoints.length; i += 3) {
    out += `${(lastPoints[i] * f).toFixed(4)} ${(lastPoints[i + 1] * f).toFixed(4)} ${(lastPoints[i + 2] * f).toFixed(4)} 0 255 160\n`;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out], { type: 'text/plain' }));
  a.download = 'objectlens-scan.ply';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
});

// -------------------------------------------------------------------- build
buildEl.addEventListener('click', async () => {
  hideError();
  thumbsEl.innerHTML = '';
  dimsEl.hidden = true;
  plyEl.disabled = true;
  lastPoints = null;
  lastCalib = null;
  calibEl.hidden = true;
  calibEl.classList.remove('none');
  relNoteEl.innerHTML = REL_NOTE_DEFAULT;

  const views = Math.max(4, Math.min(36, parseInt(viewsEl.value, 10) || 12));
  const thresh = parseFloat(threshEl.value) || 40;
  const markerMm = Math.max(1, parseFloat(markerMmEl.value) || 40);
  const dur = vid.duration;
  let t0 = parseFloat(startEl.value) || 0;
  let t1 = parseFloat(endEl.value) || 0;
  if (!isFinite(dur) || !vid.videoWidth) { showError('Load a video first.'); return; }
  if (t1 <= 0 || t1 > dur) t1 = dur;
  if (t0 < 0) t0 = 0;
  if (t1 - t0 < 0.5) {
    showError('The start–end range is too short — it must cover one full 360° turn of the object.');
    return;
  }
  buildEl.disabled = true;
  const tBegin = performance.now();
  try {
    const scale = Math.min(1, MAX_W / vid.videoWidth);
    const W = Math.round(vid.videoWidth * scale), H = Math.round(vid.videoHeight * scale);
    fc.width = W; fc.height = H;
    const masks = [];
    let bestDet = null, markerFrames = 0;
    for (let i = 0; i < views; i++) {
      const t = Math.min(t0 + (t1 - t0) * i / views, dur - 0.05);
      const angle = i * 360 / views;
      setStatus(`sampling frame ${i + 1}/${views} — ${Math.round(angle)}°…`);
      await seekTo(t);
      await new Promise(r => requestAnimationFrame(r));
      fctx.drawImage(vid, 0, 0, W, H);
      const imgData = fctx.getImageData(0, 0, W, H);
      const det = detectMarkers(imgData);
      if (det.best) {
        markerFrames++;
        if (!bestDet || det.best.squareness > bestDet.squareness ||
            (det.best.squareness === bestDet.squareness && det.best.area > bestDet.area)) {
          bestDet = det.best;
        }
      }
      const seg = segmentFrame(imgData.data, W, H, thresh, det.quads);
      masks.push(seg);
      addThumb(seg, W, H, angle);
      await tick();
    }
    // sanity checks on the masks
    const empty = masks.findIndex(m => m.count === 0);
    if (empty >= 0) {
      showError(`View ${empty + 1} produced an empty silhouette — the object could not be ` +
        'separated from the background there. Use a plain, contrasting background, even light, ' +
        'and try a lower threshold. The thumbnails above show what was extracted.');
      setStatus('failed — empty silhouette');
      return;
    }
    const medFrac = median(masks.map(m => m.frac));
    if (medFrac < 0.002) {
      showError('The silhouettes are almost empty — the object is too close in colour to the ' +
        'background. Try a lower threshold or a plainer, more contrasting background.');
      setStatus('failed — silhouettes nearly empty');
      return;
    }
    if (medFrac > 0.9) {
      showError('The silhouettes fill almost the whole frame — the background is not one plain ' +
        'colour, or the object fills the camera view. Film on a plain background with the ' +
        'whole object in frame, or raise the threshold.');
      setStatus('failed — silhouettes fill the frame');
      return;
    }
    const cx = median(masks.map(m => m.centroidX));
    const cy = median(masks.map(m => m.maxY));
    setStatus('carving visual hull…');
    const res = await carveHull(masks, W, H, cx, cy,
      (done, total) => setStatus(`carving visual hull — view ${done}/${total}…`));
    if (!res.points || res.points.length < 150) {
      showError('The silhouettes carved away to nothing — they do not agree with each other. ' +
        'Check that the clip is exactly one full 360° turn, the camera does not move, and the ' +
        'thumbnails above each hug the object. Adjust the threshold and try again.');
      setStatus('failed — empty hull');
      return;
    }
    lastPoints = res.points;
    showResult(res.points, res.voxRel);
    // ArUco calibration: the most frontal marker detection sets px-per-mm;
    // hull pixel dims / px-per-mm = true millimetres.
    if (bestDet && markerFrames > 0) {
      const pxPerMm = bestDet.meanEdge / markerMm;
      lastCalib = { pxPerMm, heightPx: res.dimsPx.y, frames: markerFrames,
                    markerId: bestDet.id };
      $('sDimX').textContent = (res.dimsPx.x / pxPerMm).toFixed(1) + ' mm';
      $('sDimY').textContent = (res.dimsPx.y / pxPerMm).toFixed(1) + ' mm';
      $('sDimZ').textContent = (res.dimsPx.z / pxPerMm).toFixed(1) + ' mm';
      calibEl.textContent = `✓ Calibrated by ArUco marker (${markerFrames} frame${markerFrames === 1 ? '' : 's'}, ` +
        `${pxPerMm.toFixed(2)} px/mm) — sizes above are true millimetres.`;
      calibEl.hidden = false;
      relNoteEl.innerHTML = '<strong>True millimetres</strong> — an ArUco marker calibrated ' +
        'this scan. Relative proportions (height = 1.00): ' +
        `${res.dims.x.toFixed(2)} × 1.00 × ${res.dims.z.toFixed(2)}. ` +
        'Single-plane caveat: the scale is exact at the marker\'s distance from the camera.';
    } else {
      $('sDimX').textContent = res.dims.x.toFixed(2);
      $('sDimY').textContent = '1.00';
      $('sDimZ').textContent = res.dims.z.toFixed(2);
      calibEl.textContent = 'No ArUco marker found in the sampled frames — showing relative units. ' +
        'Print the marker (section 4), stand it beside the object and keep it visible to get true millimetres.';
      calibEl.classList.add('none');
      calibEl.hidden = false;
    }
    $('sDimViews').textContent = views;
    $('sDimPts').textContent = (res.points.length / 3).toLocaleString();
    dimsEl.hidden = false;
    plyEl.disabled = false;
    const secs = ((performance.now() - tBegin) / 1000).toFixed(1);
    setStatus(`done in ${secs} s — ${res.voxelCount.toLocaleString()} voxels carved, ` +
      `${(res.points.length / 3).toLocaleString()} surface points ` +
      (lastCalib ? '(true mm — ArUco calibrated)' : '(relative units)'));
  } catch (e) {
    showError(e.message || 'Something went wrong while processing this video.');
    setStatus('failed');
  } finally {
    buildEl.disabled = false;
  }
});
