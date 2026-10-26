// ObjectLens 3D - "Scan your own video": fully in-browser visual hull.
// Mirrors the Python pipeline (objectlens/segment.py + hull.py):
//   1. sample N frames evenly across one full 360-degree turn,
//   2. segment each silhouette by edge detection (Sobel row fill),
//      falling back to background-colour distance (median of the four
//      corner patches, like measure.py) if the edge masks are implausible,
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
const latheEl = $('chkLathe');
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
let lastMesh = null;      // lathe mesh, or null if the profile was invalid
let lastVoxDims = null, lastVoxDimsPx = null, lastVoxRel = 1;
let three = null;         // three.js bits, created on first result
let pointsObj = null, meshObj = null;

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

// ------------------------------------------------- edge segmentation
// For transparent objects (glass) on a gradient background, background-
// colour distance fails: the glass interior is almost the background
// colour, and the gradient itself is counted as foreground, so the mask
// becomes a wide static blob and the carved hull collapses to a flat
// disc. Edges do not have that problem: the background gradient is
// smooth (low Sobel magnitude) while the object's outline is sharp.
// Per row, take the leftmost/rightmost strong edge inside a central
// band, keep rows whose width is plausible for one centred object,
// interpolate missing rows, and fill between the edges. This is the
// primary method; background-colour segmentation is the fallback.
function segmentEdgeFrame(data, w, h, quads) {
  const empty = { mask: new Uint8Array(w * h), count: 0, frac: 0, minX: w, maxX: -1,
                  minY: h, maxY: -1, centroidX: 0 };
  const gray = new Float32Array(w * h);
  for (let i = 0, px = 0; px < w * h; px++, i += 4)
    gray[px] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  const edge = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const gx = -gray[(y - 1) * w + x - 1] + gray[(y - 1) * w + x + 1]
             - 2 * gray[y * w + x - 1] + 2 * gray[y * w + x + 1]
             - gray[(y + 1) * w + x - 1] + gray[(y + 1) * w + x + 1];
    const gy = -gray[(y - 1) * w + x - 1] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + x + 1]
             + gray[(y + 1) * w + x - 1] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + x + 1];
    if (gx * gx + gy * gy > 120 * 120) edge[y * w + x] = 1;
  }
  // 3x3 dilate, so broken outline segments join up
  const dil = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let dy = -1; dy <= 1 && !v; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && xx < w && yy >= 0 && yy < h && edge[yy * w + xx]) { v = 1; break; }
    }
    dil[y * w + x] = v;
  }
  const x0 = Math.floor(w * 0.27), x1 = Math.floor(w * 0.74);
  const minW = Math.max(20, Math.floor(w * 0.05)), maxW = Math.floor(w * 0.25);
  const left = new Int32Array(h).fill(-1), right = new Int32Array(h).fill(-1);
  const good = [];
  for (let y = 0; y < h; y++) {
    let l = -1, r = -1;
    for (let x = x0; x <= x1; x++) if (dil[y * w + x]) { if (l < 0) l = x; r = x; }
    if (l >= 0 && r - l >= minW && r - l <= maxW) { left[y] = l; right[y] = r; good.push(y); }
  }
  if (good.length < h * 0.2) return empty;
  const mask = new Uint8Array(w * h);
  for (let gi = 0; gi < good.length; gi++) {
    const y0 = good[gi], y1 = good[gi + 1] !== undefined ? good[gi + 1] : y0;
    const span = Math.max(1, y1 - y0);
    for (let y = y0; y <= y1; y++) {
      const t = (y - y0) / span;
      const l = Math.round(left[y0] + (left[y1] - left[y0]) * t);
      const r = Math.round(right[y0] + (right[y1] - right[y0]) * t);
      if (r - l >= minW && r - l <= maxW) for (let x = l; x <= r; x++) mask[y * w + x] = 1;
    }
  }
  // cut out any marker quad (beside the object, never touching it)
  if (quads && quads.length) {
    for (const q of quads) {
      const xs = q.map(p => p.x), ys = q.map(p => p.y);
      const bx0 = Math.max(0, Math.floor(Math.min(...xs)) - 2), bx1 = Math.min(w - 1, Math.ceil(Math.max(...xs)) + 2);
      const by0 = Math.max(0, Math.floor(Math.min(...ys)) - 2), by1 = Math.min(h - 1, Math.ceil(Math.max(...ys)) + 2);
      for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) mask[y * w + x] = 0;
    }
  }
  // 3x3 closing (dilate then erode) to smooth the filled shape
  const dil2 = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0;
    for (let dy = -1; dy <= 1 && !v; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && xx < w && yy >= 0 && yy < h && mask[yy * w + xx]) { v = 1; break; }
    }
    dil2[y * w + x] = v;
  }
  const clean = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 1;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || xx >= w || yy < 0 || yy >= h || !dil2[yy * w + xx]) { v = 0; break; }
    }
    clean[y * w + x] = v;
  }
  let count = 0, minX = w, maxX = -1, minY = h, maxY = -1, sumX = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (clean[y * w + x]) { count++; sumX += x; if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
  }
  return { mask: clean, count, frac: count / (w * h), minX, maxX, minY, maxY,
           centroidX: count ? sumX / count : 0 };
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
function ensureScene() {
  if (three) return;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, viewerEl.clientWidth / viewerEl.clientHeight, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  viewerEl.appendChild(renderer.domElement);
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(1, 2, 1.5);
  scene.add(dir);
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

function clearViewerObjects() {
  if (pointsObj) { three.scene.remove(pointsObj); pointsObj.geometry.dispose(); pointsObj.material.dispose(); pointsObj = null; }
  if (meshObj) { three.scene.remove(meshObj); meshObj.geometry.dispose(); meshObj.material.dispose(); meshObj = null; }
}

function showResult(points, voxRel) {
  viewerEl.hidden = false;
  ensureScene();
  clearViewerObjects();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(points, 3));
  const mat = new THREE.PointsMaterial({
    color: 0x2effa8, size: Math.max(0.012, voxRel * 1.6), sizeAttenuation: true });
  pointsObj = new THREE.Points(geo, mat);
  three.scene.add(pointsObj);
}

function showMesh(mesh) {
  viewerEl.hidden = false;
  ensureScene();
  clearViewerObjects();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(mesh.colors, 3));
  geo.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.35, metalness: 0.05 });
  meshObj = new THREE.Mesh(geo, mat);
  three.scene.add(meshObj);
}

// ------------------------------------------------------- lathe smooth model
// A turntable object that is (approximately) rotationally symmetric — a
// bottle, cup, vase — is fully described by its radius profile: for each
// image row, the silhouette's half-width, medianed across views. Revolving
// that profile gives a smooth, solid mesh, instead of a blocky voxel
// point cloud. This is NOT valid for non-round objects (boxes, shoes,
// faces): for those, uncheck the lathe box and use the voxel model.
// Returns null if fewer than 20 rows have a usable radius.
function buildLathe(masks, w, h, cx, cy) {
  let minY = h, maxY = -1;
  for (const m of masks) { if (m.minY < minY) minY = m.minY; if (m.maxY > maxY) maxY = m.maxY; }
  const heightPx = cy - minY;
  if (heightPx < 20) return null;
  const radii = new Float32Array(h).fill(NaN);
  for (let y = minY; y <= cy && y < h; y++) {
    const vals = [];
    for (const m of masks) {
      let l = -1, r = -1;
      for (let x = 0; x < w; x++) if (m.mask[y * w + x]) { if (l < 0) l = x; r = x; }
      if (l >= 0 && r - l > 0 && r - l < w * 0.8) vals.push((r - l) / 2);
    }
    if (vals.length) radii[y] = median(vals);
  }
  // interpolate rows with no data between the first and last data rows
  let first = -1, last = -1;
  for (let y = minY; y <= cy && y < h; y++) { if (!isNaN(radii[y])) { if (first < 0) first = y; last = y; } }
  if (first < 0 || last - first < 20) return null;
  for (let y = first; y <= last; y++) {
    if (isNaN(radii[y])) {
      let a = y - 1; while (a >= first && isNaN(radii[a])) a--;
      let b = y + 1; while (b <= last && isNaN(radii[b])) b++;
      radii[y] = (a >= first && b <= last) ? radii[a] + (radii[b] - radii[a]) * (y - a) / (b - a) : (a >= first ? radii[a] : radii[b]);
    }
  }
  // moving-average smoothing, window 11 rows
  const smooth = new Float32Array(h);
  for (let y = first; y <= last; y++) {
    let s = 0, n = 0;
    for (let k = -5; k <= 5; k++) { const yy = y + k; if (yy >= first && yy <= last) { s += radii[yy]; n++; } }
    smooth[y] = s / n;
  }
  const step = Math.max(1, Math.floor((last - first) / 100));
  const levels = [];
  for (let y = first; y <= last; y += step) levels.push({ hRel: (cy - y) / heightPx, rRel: smooth[y] / heightPx, y });
  if (levels[levels.length - 1].y !== last) levels.push({ hRel: (cy - last) / heightPx, rRel: smooth[last] / heightPx, y: last });
  const drdh = levels.map((lv, i) => {
    if (levels.length < 2) return 0;
    if (i === 0) return (levels[1].rRel - levels[0].rRel) / (levels[1].hRel - levels[0].hRel || 1);
    if (i === levels.length - 1) return (levels[i].rRel - levels[i - 1].rRel) / (levels[i].hRel - levels[i - 1].hRel || 1);
    return (levels[i + 1].rRel - levels[i - 1].rRel) / (levels[i + 1].hRel - levels[i - 1].hRel || 1);
  });
  const SEG = 64;
  const positions = [], normals = [], colors = [];
  levels.forEach((lv, i) => {
    const cap = lv.hRel > 0.925;
    for (let j = 0; j < SEG; j++) {
      const th = 2 * Math.PI * j / SEG;
      positions.push(lv.rRel * Math.cos(th), lv.hRel, lv.rRel * Math.sin(th));
      const nx = Math.cos(th), ny = -drdh[i], nz = Math.sin(th);
      const nl = Math.hypot(nx, ny, nz) || 1;
      normals.push(nx / nl, ny / nl, nz / nl);
      if (cap) colors.push(0.78, 0.78, 0.80); else colors.push(0.69, 0.76, 0.80);
    }
  });
  const indices = [];
  for (let i = 0; i < levels.length - 1; i++) for (let j = 0; j < SEG; j++) {
    const a = i * SEG + j, b = i * SEG + (j + 1) % SEG, c = (i + 1) * SEG + (j + 1) % SEG, d = (i + 1) * SEG + j;
    indices.push(a, b, c, a, c, d);
  }
  const topC = positions.length / 3;
  positions.push(0, 1, 0); normals.push(0, 1, 0); colors.push(0.78, 0.78, 0.80);
  const botC = positions.length / 3;
  positions.push(0, 0, 0); normals.push(0, -1, 0); colors.push(0.69, 0.76, 0.80);
  for (let j = 0; j < SEG; j++) indices.push(topC, (j + 1) % SEG, j);
  const base = levels.length - 1;
  for (let j = 0; j < SEG; j++) indices.push(botC, base * SEG + j, base * SEG + (j + 1) % SEG);
  let maxR = 0;
  for (const lv of levels) if (lv.rRel > maxR) maxR = lv.rRel;
  return { positions: new Float32Array(positions), normals: new Float32Array(normals),
           colors: new Float32Array(colors), indices: new Uint32Array(indices),
           dims: { x: 2 * maxR, y: 1, z: 2 * maxR }, vertexCount: positions.length / 3, faceCount: indices.length / 3 };
}

function latheMode() { return !!(latheEl && latheEl.checked && lastMesh); }

function currentDims() {
  if (latheMode()) return lastMesh.dims;
  return lastVoxDims || { x: 0, y: 1, z: 0 };
}

function updateDims() {
  const dims = currentDims();
  if (lastCalib) {
    const heightMm = lastVoxDimsPx ? lastVoxDimsPx.y / lastCalib.pxPerMm : 0;
    $('sDimX').textContent = (dims.x * heightMm).toFixed(1) + ' mm';
    $('sDimY').textContent = heightMm.toFixed(1) + ' mm';
    $('sDimZ').textContent = (dims.z * heightMm).toFixed(1) + ' mm';
  } else {
    $('sDimX').textContent = dims.x.toFixed(2);
    $('sDimY').textContent = '1.00';
    $('sDimZ').textContent = dims.z.toFixed(2);
  }
  $('sDimPts').textContent = latheMode() ? lastMesh.vertexCount.toLocaleString() : (lastPoints ? (lastPoints.length / 3).toLocaleString() : '—');
  plyEl.textContent = latheMode() ? 'Download mesh PLY' : 'Download PLY';
}

function showCurrentModel() {
  if (latheMode()) showMesh(lastMesh);
  else if (lastPoints) showResult(lastPoints, lastVoxRel);
  updateDims();
}

if (latheEl) latheEl.addEventListener('change', () => { if (lastPoints || lastMesh) showCurrentModel(); });

// ---------------------------------------------------------------------- PLY
plyEl.addEventListener('click', () => {
  // calibrated: rel coords are height-normalised, so mm = rel * height in mm
  const f = lastCalib ? lastCalib.heightPx / lastCalib.pxPerMm : 1;
  const comment = lastCalib
    ? `comment ObjectLens 3D browser scan - units are MILLIMETRES (ArUco-calibrated, ${lastCalib.pxPerMm.toFixed(2)} px/mm)`
    : 'comment ObjectLens 3D browser scan - RELATIVE units, height = 1.0 (no ArUco marker in video)';
  let out, download;
  if (latheMode()) {
    const m = lastMesh, nv = m.positions.length / 3;
    out = 'ply\nformat ascii 1.0\n' + comment + '\ncomment lathe smooth mesh\n' +
      `element vertex ${nv}\nproperty float x\nproperty float y\nproperty float z\n` +
      'property float nx\nproperty float ny\nproperty float nz\n' +
      'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
      `element face ${m.indices.length / 3}\nproperty list uchar int vertex_indices\nend_header\n`;
    for (let i = 0; i < nv; i++) {
      out += `${(m.positions[i * 3] * f).toFixed(4)} ${(m.positions[i * 3 + 1] * f).toFixed(4)} ${(m.positions[i * 3 + 2] * f).toFixed(4)} ` +
        `${m.normals[i * 3].toFixed(4)} ${m.normals[i * 3 + 1].toFixed(4)} ${m.normals[i * 3 + 2].toFixed(4)} ` +
        `${Math.round(m.colors[i * 3] * 255)} ${Math.round(m.colors[i * 3 + 1] * 255)} ${Math.round(m.colors[i * 3 + 2] * 255)}\n`;
    }
    for (let i = 0; i < m.indices.length; i += 3) out += `3 ${m.indices[i]} ${m.indices[i + 1]} ${m.indices[i + 2]}\n`;
    download = 'objectlens-mesh.ply';
  } else {
    if (!lastPoints) return;
    const n = lastPoints.length / 3;
    out = 'ply\nformat ascii 1.0\n' + comment + '\n' +
      `element vertex ${n}\nproperty float x\nproperty float y\nproperty float z\n` +
      'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
    for (let i = 0; i < lastPoints.length; i += 3) {
      out += `${(lastPoints[i] * f).toFixed(4)} ${(lastPoints[i + 1] * f).toFixed(4)} ${(lastPoints[i + 2] * f).toFixed(4)} 0 255 160\n`;
    }
    download = 'objectlens-scan.ply';
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out], { type: 'text/plain' }));
  a.download = download;
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
  lastMesh = null;
  lastVoxDims = null; lastVoxDimsPx = null;
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
    const frames = [];
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
      frames.push({ data: new Uint8ClampedArray(imgData.data), quads: det.quads, angle });
      await tick();
    }
    // Segmentation: edge first (works for glass and gradient backgrounds),
    // background-colour fallback if the edge masks are implausible.
    setStatus('segmenting frames…');
    const edgeMasks = frames.map(f => segmentEdgeFrame(f.data, W, H, f.quads));
    const medEdge = median(edgeMasks.map(m => m.frac));
    let masks, segMethod;
    if (medEdge >= 0.02 && medEdge <= 0.30) { masks = edgeMasks; segMethod = 'edge'; }
    else { masks = frames.map(f => segmentFrame(f.data, W, H, thresh, f.quads)); segMethod = 'colour'; }
    for (let i = 0; i < frames.length; i++) {
      fctx.putImageData(new ImageData(frames[i].data, W, H), 0, 0);
      addThumb(masks[i], W, H, frames[i].angle);
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
    lastVoxDims = res.dims; lastVoxDimsPx = res.dimsPx; lastVoxRel = res.voxRel;
    setStatus('building smooth lathe model…');
    lastMesh = buildLathe(masks, W, H, cx, cy);
    // ArUco calibration: the most frontal marker detection sets px-per-mm;
    // hull pixel dims / px-per-mm = true millimetres.
    if (bestDet && markerFrames > 0) {
      const pxPerMm = bestDet.meanEdge / markerMm;
      lastCalib = { pxPerMm, heightPx: res.dimsPx.y, frames: markerFrames,
                    markerId: bestDet.id };
      calibEl.textContent = `✓ Calibrated by ArUco marker (${markerFrames} frame${markerFrames === 1 ? '' : 's'}, ` +
        `${pxPerMm.toFixed(2)} px/mm) — sizes above are true millimetres.`;
      calibEl.hidden = false;
      const cd = currentDims();
      relNoteEl.innerHTML = '<strong>True millimetres</strong> — an ArUco marker calibrated ' +
        'this scan. Relative proportions (height = 1.00): ' +
        `${cd.x.toFixed(2)} × 1.00 × ${cd.z.toFixed(2)}. ` +
        'Single-plane caveat: the scale is exact at the marker\'s distance from the camera.';
    } else {
      calibEl.textContent = 'No ArUco marker found in the sampled frames — showing relative units. ' +
        'Print the marker (section 4), stand it beside the object and keep it visible to get true millimetres.';
      calibEl.classList.add('none');
      calibEl.hidden = false;
    }
    $('sDimViews').textContent = views;
    showCurrentModel();
    dimsEl.hidden = false;
    plyEl.disabled = false;
    const secs = ((performance.now() - tBegin) / 1000).toFixed(1);
    setStatus(`done in ${secs} s — ${res.voxelCount.toLocaleString()} voxels carved, ` +
      `${(res.points.length / 3).toLocaleString()} surface points ` +
      (lastCalib ? '(true mm — ArUco calibrated)' : '(relative units)') +
      ` — segmentation: ${segMethod} — model: ${latheMode() ? 'lathe smooth mesh' : 'voxel points'}`);
  } catch (e) {
    showError(e.message || 'Something went wrong while processing this video.');
    setStatus('failed');
  } finally {
    buildEl.disabled = false;
  }
});
