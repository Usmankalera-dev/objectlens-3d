// ObjectLens 3D - three.js viewer for the sample scan (PLY from the pipeline).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const holder = document.getElementById('viewer');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, holder.clientWidth / holder.clientHeight, 1, 5000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(holder.clientWidth, holder.clientHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
holder.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(180, 300, 220);
scene.add(key);
const rim = new THREE.DirectionalLight(0x66aaff, 0.7);
rim.position.set(-220, 120, -180);
scene.add(rim);
const grid = new THREE.GridHelper(400, 40, 0x1e2c44, 0x14203a);
scene.add(grid);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.1;
renderer.domElement.addEventListener('pointerdown', () => { controls.autoRotate = false; });

new PLYLoader().load('sample.ply', geometry => {
  geometry.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.55, metalness: 0.05,
    side: THREE.DoubleSide, flatShading: true });
  const mesh = new THREE.Mesh(geometry, mat);
  geometry.computeBoundingBox();
  const c = geometry.boundingBox.getCenter(new THREE.Vector3());
  mesh.position.set(-c.x, -geometry.boundingBox.min.y, -c.z);
  scene.add(mesh);
  controls.target.set(0, 55, 0);
  camera.position.set(150, 130, 240);
});

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, camera);
}
tick();

addEventListener('resize', () => {
  camera.aspect = holder.clientWidth / holder.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(holder.clientWidth, holder.clientHeight);
});

// fill the dimension cards from the real scan report
fetch('sample-report.json').then(r => r.json()).then(rep => {
  const d = rep.dimensions_mm;
  const set = (id, v) => { document.getElementById(id).textContent = v; };
  set('dimX', d.width_x.toFixed(1) + ' mm');
  set('dimY', d.height_y.toFixed(1) + ' mm');
  set('dimZ', d.depth_z.toFixed(1) + ' mm');
  set('dimViews', rep.views_used);
  set('dimFaces', rep.mesh_faces ? rep.mesh_faces.toLocaleString() : 'points');
}).catch(() => {});
