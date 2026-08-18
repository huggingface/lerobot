// === Hand Visualizer with Pre-Connect Sliders + Per-Joint Angle Limits ===
// Assumes your HTML already has elements with the following IDs:
// connectButton, disconnectButton, baudRate, statusIndicator, jointsContainer, logContainer,
// canvas-container, frontView, sideView, topView, resetView
// Requires Three.js + OrbitControls loaded on the page.

// -------------------- Config --------------------
const MAX_JOINTS = 16;
const RAW_MIN = 0, RAW_MAX = 4096;
const RAW_CENTER = (RAW_MIN + RAW_MAX) / 2;
const DEG = Math.PI / 180;
const UI_DEG_MIN = -90, UI_DEG_MAX = 90; // UI sliders for angle limits

// -------------------- State --------------------
let port;
let reader;
let keepReading = false;
let isConnected = false;
const decoder = new TextDecoder();
let inputBuffer = '';

let jointValues = new Array(MAX_JOINTS).fill(RAW_CENTER);

// Auto-calibration: track observed min/max per joint
let observedMin = new Array(MAX_JOINTS).fill(Infinity);
let observedMax = new Array(MAX_JOINTS).fill(-Infinity);
let calibrationEnabled = true;

// Three.js
let scene, camera, renderer, controls;
let hand = { palm: null, fingers: [] };

// DOM
const connectButton     = document.getElementById('connectButton');
const disconnectButton  = document.getElementById('disconnectButton');
const baudRateSelect    = document.getElementById('baudRate');
const statusIndicator   = document.getElementById('statusIndicator');
const jointsContainer   = document.getElementById('jointsContainer');
const logContainer      = document.getElementById('logContainer');
const canvasContainer   = document.getElementById('canvas-container');
const frontViewBtn      = document.getElementById('frontView');
const sideViewBtn       = document.getElementById('sideView');
const topViewBtn        = document.getElementById('topView');
const resetViewBtn      = document.getElementById('resetView');

// Helpers
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const invLerp = (a, b, x) => clamp((x - a) / (b - a), 0, 1);

// -------------------- Joint Map with per-joint angle limits --------------------
const fingerJointMap = [
  // Thumb (4)
  { finger:0, joint:0, type:'CMC_ABDUCTION',  min:RAW_MIN, max:RAW_MAX, inverted:true },
  { finger:0, joint:1, type:'CMC_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:true },
  { finger:0, joint:2, type:'MCP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:true }, // +45° only
  { finger:0, joint:3, type:'IP_FLEXION',     min:RAW_MIN, max:RAW_MAX, inverted:true }, // +45° only

  // Index (3)
  { finger:1, joint:0, type:'MCP_ABDUCTION',  min:RAW_MIN, max:RAW_MAX, inverted:true },
  { finger:1, joint:1, type:'MCP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:false },
  { finger:1, joint:2, type:'PIP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:true }, // +45° only

  // Middle (3)
  { finger:2, joint:0, type:'MCP_ABDUCTION',  min:RAW_MIN, max:RAW_MAX, inverted:true },
  { finger:2, joint:1, type:'MCP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:true },
  { finger:2, joint:2, type:'PIP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:true }, // +45° only

  // Ring (3)
  { finger:3, joint:0, type:'MCP_ABDUCTION',  min:RAW_MIN, max:RAW_MAX, inverted:true },
  { finger:3, joint:1, type:'MCP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:false },
  { finger:3, joint:2, type:'PIP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:false }, // +45° only

  // Pinky (3)
  { finger:4, joint:0, type:'MCP_ABDUCTION',  min:RAW_MIN, max:RAW_MAX, inverted:false },
  { finger:4, joint:1, type:'MCP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:false },
  { finger:4, joint:2, type:'PIP_FLEXION',    min:RAW_MIN, max:RAW_MAX, inverted:false }  // +45° only
];

// Assign angle limits (radians) per joint (default ±45°, exceptions: +45° only)
for (const j of fingerJointMap) {
  const isThumb = j.finger === 0;
  const isPIP   = j.type === 'PIP_FLEXION';
  let minA = -45 * DEG, maxA = +45 * DEG;
  if ((isThumb && (j.type === 'MCP_FLEXION' || j.type === 'IP_FLEXION')) || (!isThumb && isPIP)) {
    minA = 0;
    maxA = +45 * DEG;
  }
  j.angleMin = minA;
  j.angleMax = maxA;
}

// -------------------- UI: Joint Panel --------------------
const uiRefs = []; // per joint: { valueLabel, bar, barWrap, slider, invertChk, minDeg, maxDeg }

function initializeJointElements() {
  jointsContainer.innerHTML = '';
  uiRefs.length = 0;

  for (let i = 0; i < MAX_JOINTS; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'joint-info';

    const fingerIndex = i < 4 ? 0 : Math.floor((i - 4) / 3) + 1;
    const jointInfo = fingerJointMap[i];
    const jointType = jointInfo?.type || 'Unknown';
    const fingerName = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'][fingerIndex];

    // Header
    const nameEl = document.createElement('div');
    nameEl.className = 'joint-name';
    nameEl.textContent = `${fingerName} – ${jointType}`;

    // Value + bar
    const valueEl = document.createElement('div');
    valueEl.className = 'joint-value';
    valueEl.textContent = `Value: ${jointValues[i]}`;

    const barWrap = document.createElement('div');
    barWrap.className = 'bar-container';
    const barEl = document.createElement('div');
    barEl.className = 'bar';
    barWrap.appendChild(barEl);

    // Slider for pre-connect manual control
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(RAW_MIN);
    slider.max = String(RAW_MAX);
    slider.value = String(jointValues[i]);
    slider.step = '1';
    slider.className = 'joint-slider';

    slider.addEventListener('input', () => {
      if (isConnected) return; // ignore while connected
      let v = parseInt(slider.value, 10);
      if (jointInfo?.inverted) v = (jointInfo.min + jointInfo.max) - v;
      jointValues[i] = clamp(jointInfo ? v : 0, RAW_MIN, RAW_MAX);
      updateJointDisplay(i, jointValues[i]);
      updateHandModel();
    });

    // Invert checkbox
    const invertLbl = document.createElement('label');
    invertLbl.className = 'invert-toggle';
    const invertChk = document.createElement('input');
    invertChk.type = 'checkbox';
    invertChk.checked = !!jointInfo?.inverted;
    invertChk.addEventListener('change', () => {
      if (jointInfo) jointInfo.inverted = invertChk.checked;
      addLogMessage(`${fingerName} ${jointType} inversion ${invertChk.checked ? 'enabled' : 'disabled'}`);
    });
    invertLbl.appendChild(invertChk);
    invertLbl.appendChild(document.createTextNode('Invert Values'));

    // Angle limits (deg) controls
    const limitsRow = document.createElement('div');
    limitsRow.className = 'limits-row';

    const minDeg = document.createElement('input');
    minDeg.type = 'number';
    minDeg.min = String(UI_DEG_MIN);
    minDeg.max = String(UI_DEG_MAX);
    minDeg.step = '1';
    minDeg.value = String(Math.round((jointInfo.angleMin || 0) / DEG));
    minDeg.className = 'limit-num';

    const maxDeg = document.createElement('input');
    maxDeg.type = 'number';
    maxDeg.min = String(UI_DEG_MIN);
    maxDeg.max = String(UI_DEG_MAX);
    maxDeg.step = '1';
    maxDeg.value = String(Math.round((jointInfo.angleMax || 0) / DEG));
    maxDeg.className = 'limit-num';

    const minLbl = document.createElement('span'); minLbl.textContent = 'min°';
    const maxLbl = document.createElement('span'); maxLbl.textContent = 'max°';
    minLbl.className = 'limit-label'; maxLbl.className = 'limit-label';

    function syncLimits() {
      let mn = parseFloat(minDeg.value);
      let mx = parseFloat(maxDeg.value);
      if (isNaN(mn)) mn = -45;
      if (isNaN(mx)) mx = +45;
      if (mn > mx) [mn, mx] = [mx, mn];
      jointInfo.angleMin = clamp(mn, UI_DEG_MIN, UI_DEG_MAX) * DEG;
      jointInfo.angleMax = clamp(mx, UI_DEG_MIN, UI_DEG_MAX) * DEG;
      minDeg.value = String(Math.round(jointInfo.angleMin / DEG));
      maxDeg.value = String(Math.round(jointInfo.angleMax / DEG));
      updateHandModel();
    }
    minDeg.addEventListener('change', syncLimits);
    maxDeg.addEventListener('change', syncLimits);

    limitsRow.appendChild(minLbl);
    limitsRow.appendChild(minDeg);
    limitsRow.appendChild(maxLbl);
    limitsRow.appendChild(maxDeg);

         // Calibration controls
     const calibRow = document.createElement('div');
     calibRow.className = 'calib-row';

     const resetCalibBtn = document.createElement('button');
     resetCalibBtn.textContent = 'Reset Calib';
     resetCalibBtn.className = 'calib-btn';
     resetCalibBtn.addEventListener('click', () => {
       observedMin[i] = Infinity;
       observedMax[i] = -Infinity;
       addLogMessage(`Reset calibration for ${fingerName} ${jointType}`);
     });

     const calibStatus = document.createElement('span');
     calibStatus.className = 'calib-status';
     calibStatus.textContent = `Range: --`;

     calibRow.appendChild(resetCalibBtn);
     calibRow.appendChild(calibStatus);

     // Compose
     wrap.appendChild(nameEl);
     wrap.appendChild(valueEl);
     wrap.appendChild(barWrap);
     wrap.appendChild(slider);
     wrap.appendChild(invertLbl);
     wrap.appendChild(limitsRow);
     wrap.appendChild(calibRow);

     jointsContainer.appendChild(wrap);

     uiRefs[i] = { valueLabel: valueEl, bar: barEl, barWrap, slider, invertChk, minDeg, maxDeg, nameEl, calibStatus };
  }

  setConnectedUI(false); // initial state: sliders active
}

 // Toggle UI between pre-connect SLIDERS vs post-connect BARS
 function setConnectedUI(connected) {
   isConnected = connected;
   for (let i = 0; i < uiRefs.length; i++) {
     const ui = uiRefs[i];
     if (!ui) continue;
     // Show bars when connected; sliders disabled/hidden
     ui.barWrap.style.display = connected ? '' : 'none';
     ui.slider.disabled = connected;
     ui.slider.style.display = connected ? 'none' : '';
   }

   // Reset calibration when connecting
   if (connected) {
     observedMin.fill(Infinity);
     observedMax.fill(-Infinity);
     addLogMessage('Calibration reset - move joints through full range for best results');
   }
 }

// Update joint display (value text + bar color/width + slider position if needed)
function updateJointDisplay(jointIndex, value) {
  const ui = uiRefs[jointIndex];
  const info = fingerJointMap[jointIndex];
  if (!ui || !info) return;

  ui.valueLabel.textContent = `Value: ${value}`;

  // bar
  const min = info.min, max = info.max;
  const pct = clamp((value - min) / (max - min), 0, 1) * 100;
  ui.bar.style.width = `${pct}%`;
  const hue = Math.floor(pct * 1.2); // 0..120
  ui.bar.style.backgroundColor = `hsl(${hue}, 80%, 50%)`;

  // slider (only meaningful when not connected; keep in sync anyway)
  const rawForSlider = info.inverted ? (info.min + info.max) - value : value;
  if (!isConnected) ui.slider.value = String(clamp(Math.round(rawForSlider), RAW_MIN, RAW_MAX));
}

// -------------------- Serial I/O --------------------
async function readSerialData() {
  while (port?.readable && keepReading) {
    reader = port.readable.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) processData(decoder.decode(value));
      }
    } catch (err) {
      console.error('Error reading:', err);
      addLogMessage(`Error: ${err.message}`);
      break;
    } finally {
      reader.releaseLock();
    }
  }
}

 function processData(chunk) {
   inputBuffer += chunk;
   let idx;
   while ((idx = inputBuffer.indexOf('\n')) !== -1) {
     const line = inputBuffer.slice(0, idx).trim();
     inputBuffer = inputBuffer.slice(idx + 1);

     const vals = line.split(/\s+/).map(v => parseInt(v, 10));
     if (vals.length === MAX_JOINTS && vals.every(v => Number.isFinite(v))) {
       for (let i = 0; i < MAX_JOINTS; i++) {
         const info = fingerJointMap[i];
         if (!info) continue;

         let rawValue = vals[i];

         // Update calibration tracking
         if (calibrationEnabled) {
           observedMin[i] = Math.min(observedMin[i], rawValue);
           observedMax[i] = Math.max(observedMax[i], rawValue);

           // Update calibration display
           const ui = uiRefs[i];
           if (ui && ui.calibStatus) {
             if (observedMin[i] !== Infinity && observedMax[i] !== -Infinity) {
               ui.calibStatus.textContent = `Range: ${observedMin[i]}-${observedMax[i]}`;
             }
           }

           // Remap observed range to target range
           if (observedMin[i] !== Infinity && observedMax[i] !== -Infinity && observedMax[i] > observedMin[i]) {
             const observedRange = observedMax[i] - observedMin[i];
             const targetRange = info.max - info.min;
             const normalizedValue = (rawValue - observedMin[i]) / observedRange;
             rawValue = info.min + (normalizedValue * targetRange);
           }
         }

         let v = clamp(rawValue, info.min, info.max);
         if (info.inverted) v = (info.min + info.max) - v;
         jointValues[i] = v;
         updateJointDisplay(i, v);
       }
       updateHandModel();
     } else {
       addLogMessage(`Received: ${line}`);
     }
   }
 }

async function connectToDevice() {
  try {
    port = await navigator.serial.requestPort();
    const baudRate = parseInt(baudRateSelect.value, 10) || 115200;
    await port.open({ baudRate });

    keepReading = true;
    setConnectedUI(true);

    statusIndicator.textContent = 'Status: Connected';
    statusIndicator.className = 'status connected';
    connectButton.disabled = true;
    disconnectButton.disabled = false;
    baudRateSelect.disabled = true;

    addLogMessage(`Connected at ${baudRate} baud`);
    readSerialData();
  } catch (e) {
    console.error('Connect error:', e);
    addLogMessage(`Connection error: ${e.message}`);
  }
}

async function disconnectFromDevice() {
  try {
    keepReading = false;
    if (reader) {
      try { reader.cancel(); } catch {}
    }
    if (port) {
      await port.close();
      port = null;
    }
  } catch (e) {
    console.error('Disconnect error:', e);
    addLogMessage(`Disconnection error: ${e.message}`);
  } finally {
    setConnectedUI(false);
    statusIndicator.textContent = 'Status: Disconnected';
    statusIndicator.className = 'status disconnected';
    connectButton.disabled = false;
    disconnectButton.disabled = true;
    baudRateSelect.disabled = false;
    addLogMessage('Disconnected');
  }
}

// -------------------- Three.js Scene --------------------
function initThreeJS() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  camera = new THREE.PerspectiveCamera(
    75,
    canvasContainer.clientWidth / canvasContainer.clientHeight,
    0.1, 1000
  );
  camera.position.set(0, 15, 15);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  canvasContainer.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;

  const ambientLight = new THREE.AmbientLight(0x404040);
  scene.add(ambientLight);
  const dir1 = new THREE.DirectionalLight(0xffffff, 0.5);
  dir1.position.set(1, 1, 1);
  scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dir2.position.set(-1, 1, -1);
  scene.add(dir2);

  const gridHelper = new THREE.GridHelper(20, 20);
  scene.add(gridHelper);

  createHandModel();
  window.addEventListener('resize', onWindowResize);
  animate();
}

function createHandModel() {
  const palmMaterial   = new THREE.MeshPhongMaterial({ color: 0xf5c396 });
  const fingerMaterial = new THREE.MeshPhongMaterial({ color: 0xf5c396 });
  const jointMaterial  = new THREE.MeshPhongMaterial({ color: 0xe3a977 });

     const palmGeometry = new THREE.BoxGeometry(7, 1, 8);
   hand.palm = new THREE.Mesh(palmGeometry, palmMaterial);
   hand.palm.position.set(0, 0, 0);
   hand.palm.rotation.x = Math.PI / 2; // hand vertical, palm facing forward
   scene.add(hand.palm);

  const fingerWidth = 1, fingerHeight = 0.8;
  const fingerSegmentLengths = [3, 2, 1.5];
  const thumbSegmentLengths  = [2, 2, 1.5];

  const fingerBasePositions = [
    [ 3,   0,  -2],  // Thumb
    [ 1.5,-0.5,-4],  // Index
    [ 0,  -0.5,-4],  // Middle
    [-1.5,-0.5,-4],  // Ring
    [-3,  -0.5,-4],  // Pinky
  ];
  const fingerBaseRot = [
    { x:0, y:-Math.PI/3,  z: Math.PI/3 },  // Thumb
    { x:0, y:-Math.PI/48, z: 0 },
    { x:0, y: Math.PI/48, z: 0 },
    { x:0, y: Math.PI/32, z: 0 },
    { x:0, y: Math.PI/24, z: 0 }
  ];

  for (let fIdx = 0; fIdx < 5; fIdx++) {
    const finger = { name:['Thumb','Index','Middle','Ring','Pinky'][fIdx], segments:[], joints:[] };
    const isThumb = fIdx === 0;
    const segLens = isThumb ? thumbSegmentLengths : fingerSegmentLengths;

    finger.group = new THREE.Group();
    finger.group.position.set(...fingerBasePositions[fIdx]);
    finger.group.rotation.x = fingerBaseRot[fIdx].x;
    finger.group.rotation.y = fingerBaseRot[fIdx].y;
    finger.group.rotation.z = fingerBaseRot[fIdx].z;
    finger.group.userData.baseRot = {
      x:finger.group.rotation.x,
      y:finger.group.rotation.y,
      z:finger.group.rotation.z
    };
    hand.palm.add(finger.group);

    let parent = finger.group;
    for (let s = 0; s < segLens.length; s++) {
      const segGroup = new THREE.Group();

      const jGeom = new THREE.SphereGeometry(fingerWidth * 0.6, 8, 8);
      const joint = new THREE.Mesh(jGeom, jointMaterial);
      segGroup.add(joint);

      const segGeom = new THREE.BoxGeometry(fingerWidth, fingerHeight, segLens[s]);
      const seg = new THREE.Mesh(segGeom, fingerMaterial);
      seg.position.z = -segLens[s] / 2;
      segGroup.add(seg);

      parent.add(segGroup);

      finger.segments.push(segGroup);
      finger.joints.push(joint);

      if (s < segLens.length - 1) {
        const connector = new THREE.Group();
        connector.position.z = -segLens[s];
        segGroup.add(connector);
        parent = connector;
      }
    }

    hand.fingers.push(finger);
  }

  addFingerLabels();
  addHandLabel();
}

function addFingerLabels() {
  const names = ['Thumb','Index','Middle','Ring','Pinky'];
  for (let i = 0; i < hand.fingers.length; i++) {
    const finger = hand.fingers[i];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 128; canvas.height = 32;
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.font = 'bold 16px Arial';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(names[i], canvas.width/2, canvas.height/2);

    const texture = new THREE.CanvasTexture(canvas);
    const geom = new THREE.PlaneGeometry(2, 0.5);
    const mat = new THREE.MeshBasicMaterial({ map:texture, transparent:true, side:THREE.DoubleSide });
    const label = new THREE.Mesh(geom, mat);
    label.position.set(0, -1.5, -2);
    label.rotation.x = Math.PI / 2;
    finger.group.add(label);
  }
}

function addHandLabel() {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 256; canvas.height = 64;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.font = 'bold 24px Arial';
  ctx.fillStyle = '#000000';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
     ctx.fillText('RIGHT HAND (VERTICAL)', canvas.width/2, canvas.height/2);

  const texture = new THREE.CanvasTexture(canvas);
  const geom = new THREE.PlaneGeometry(7, 1.75);
  const mat = new THREE.MeshBasicMaterial({ map:texture, transparent:true, side:THREE.DoubleSide });
  const label = new THREE.Mesh(geom, mat);
  label.position.set(0, -2, 0);
  label.rotation.x = Math.PI / 2;
  scene.add(label);
}

function updateHandModel() {
  for (let i = 0; i < MAX_JOINTS; i++) {
    const info = fingerJointMap[i];
    if (!info) continue;
    const { finger, joint, type, min, max, angleMin, angleMax } = info;
    const raw = jointValues[i];
    const f = hand.fingers[finger];
    if (!f) continue;

    const center = (min + max) / 2;
    let angle = 0;

         if (type.includes('ABDUCTION')) {
       // symmetric around neutral
       const k = clamp((raw - center) / ((max - min) / 2), -1, 1);
       angle = angleMin + (k + 1) * 0.5 * (angleMax - angleMin);

       const base = f.group.userData.baseRot || {x:0,y:0,z:0};
       if (finger === 0 && joint === 0) {
         // Thumb: abduction about Z (toward/away from palm)
         f.group.rotation.z = base.z + angle;
       } else {
         // Other fingers: side-to-side about Y
         f.group.rotation.y = base.y + angle;
       }
     } else if (type.includes('FLEXION')) {
       const isThumb = finger === 0;
       const isMCP = type === 'MCP_FLEXION';
       const isPIP = type === 'PIP_FLEXION';
       const positiveOnly = (isThumb && (type === 'MCP_FLEXION' || type === 'IP_FLEXION')) || (!isThumb && isPIP);

       if (positiveOnly) {
         const t = raw <= center ? 0 : invLerp(center, max, raw); // 0..1
         angle = angleMin + t * (angleMax - angleMin);           // 0..+limit
       } else {
         const k = clamp((raw - center) / ((max - min) / 2), -1, 1);
         angle = angleMin + (k + 1) * 0.5 * (angleMax - angleMin);
       }

       if (isMCP) {
         // MCP flexion applies to the finger base group (same as abduction)
         const base = f.group.userData.baseRot || {x:0,y:0,z:0};
         f.group.rotation.x = base.x + angle;
       } else if (f.segments[joint]) {
         // PIP/DIP/IP flexion applies to individual segments
         f.segments[joint].rotation.x = angle;
       }
     }
  }
}

// -------------------- Render Loop --------------------
function onWindowResize() {
  camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// -------------------- Misc UI --------------------
function addLogMessage(msg) {
  const el = document.createElement('div');
  el.textContent = msg;
  logContainer.appendChild(el);
  logContainer.scrollTop = logContainer.scrollHeight;
  while (logContainer.children.length > 100) {
    logContainer.removeChild(logContainer.firstChild);
  }
}

// Camera view controls
frontViewBtn?.addEventListener('click', () => { camera.position.set(0, 0, 20); camera.lookAt(0,0,0); controls.update(); });
sideViewBtn?.addEventListener('click',  () => { camera.position.set(20, 0, 0); camera.lookAt(0,0,0); controls.update(); });
topViewBtn?.addEventListener('click',   () => { camera.position.set(0, 20, 0); camera.lookAt(0,0,0); controls.update(); });
resetViewBtn?.addEventListener('click', () => { camera.position.set(10,10,10); camera.lookAt(0,0,0); controls.update(); });

// Serial connect buttons
connectButton?.addEventListener('click', connectToDevice);
disconnectButton?.addEventListener('click', disconnectFromDevice);

// Web Serial support check
if (!navigator.serial) {
  statusIndicator.textContent = 'Status: Web Serial API not supported in this browser';
  connectButton.disabled = true;
  addLogMessage('ERROR: Web Serial API is not supported in this browser. Try Chrome or Edge.');
}

// -------------------- Boot --------------------
initThreeJS();
initializeJointElements();

 // -------------------- Styles (inline) --------------------
 const styleElement = document.createElement('style');
 styleElement.textContent = `
 .joint-info { border-bottom: 1px solid #eee; padding: 8px 0; }
 .joint-name { font-weight: 600; margin-bottom: 4px; }
 .joint-value { font-size: 12px; color: #333; margin-bottom: 4px; }
 .bar-container { width: 100%; height: 8px; background: #ddd; border-radius: 4px; overflow: hidden; }
 .bar { height: 100%; width: 0%; background: #4caf50; }
 .joint-slider { width: 100%; margin: 6px 0; }
 .invert-toggle { display: inline-flex; align-items: center; gap: 6px; margin-top: 4px; font-size: 12px; color: #555; }
 .limits-row { display: flex; align-items: center; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
 .limit-label { font-size: 11px; color: #666; }
 .limit-num { width: 60px; }
 .calib-row { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
 .calib-btn { padding: 2px 6px; font-size: 11px; background: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer; }
 .calib-btn:hover { background: #d32f2f; }
 .calib-status { font-size: 11px; color: #666; }
 .status.connected { color: #0a0; }
 .status.disconnected { color: #a00; }
 `;
 document.head.appendChild(styleElement);
