import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  RigController,
  SomaOverlay,
  SomaEditController,
  MeshTransformController,
  ViewController,
} from "./rig_tools.js?v=skeleton-verify-1";
import { TwoBoneIKController } from "./ik_controls.js?v=two-bone-1";
import { FullBodyIKController } from "./full_body_ik.js?v=full-body-1";

const API_PREFIX = window.MKA_API_PREFIX || "";
function apiUrl(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_PREFIX}${p}`;
}

function t(key, vars) {
  return window.MKA_I18N ? window.MKA_I18N.t(key, vars) : key;
}

function ta(text) {
  return window.MKA_I18N ? window.MKA_I18N.translateApi(text) : text;
}

const healthEl = document.getElementById("health");
const uploadStatus = document.getElementById("uploadStatus");
const bindBtn = document.getElementById("bindBtn");
const jobStatus = document.getElementById("jobStatus");
const downloadLink = document.getElementById("downloadLink");
const canvas = document.getElementById("viewport");
const alignPanel = document.getElementById("alignPanel");
const autoAlignBtn = document.getElementById("autoAlignBtn");
const resetAlignBtn = document.getElementById("resetAlignBtn");
const confirmAlignBtn = document.getElementById("confirmAlignBtn");
const alignStatus = document.getElementById("alignStatus");
const alignModeTranslate = document.getElementById("alignModeTranslate");
const alignModeRotate = document.getElementById("alignModeRotate");
const alignModeScale = document.getElementById("alignModeScale");
const alignTx = document.getElementById("alignTx");
const alignTy = document.getElementById("alignTy");
const alignTz = document.getElementById("alignTz");
const alignRx = document.getElementById("alignRx");
const alignRy = document.getElementById("alignRy");
const alignRz = document.getElementById("alignRz");
const alignScale = document.getElementById("alignScale");
const viewFront = document.getElementById("viewFront");
const viewSide = document.getElementById("viewSide");
const viewTop = document.getElementById("viewTop");
const viewPerspective = document.getElementById("viewPerspective");
const viewToolbar = document.getElementById("viewToolbar");
const confirmPanel = document.getElementById("confirmPanel");
const acceptSkeletonBtn = document.getElementById("acceptSkeletonBtn");
const skeletonModeTranslate = document.getElementById("skeletonModeTranslate");
const skeletonModeRotate = document.getElementById("skeletonModeRotate");
const skeletonModeScale = document.getElementById("skeletonModeScale");
const skeletonSpaceLocal = document.getElementById("skeletonSpaceLocal");
const skeletonSpaceWorld = document.getElementById("skeletonSpaceWorld");
const skeletonSelectedBone = document.getElementById("skeletonSelectedBone");
const resetSkeletonBtn = document.getElementById("resetSkeletonBtn");
const confirmStatus = document.getElementById("confirmStatus");
const somaEditPanel = document.getElementById("somaEditPanel");
const somaModeTranslate = document.getElementById("somaModeTranslate");
const somaModeRotate = document.getElementById("somaModeRotate");
const somaModeScale = document.getElementById("somaModeScale");
const somaSpaceLocal = document.getElementById("somaSpaceLocal");
const somaSpaceWorld = document.getElementById("somaSpaceWorld");
const somaSelectedBone = document.getElementById("somaSelectedBone");
const resetSomaBtn = document.getElementById("resetSomaBtn");
const confirmSomaBtn = document.getElementById("confirmSomaBtn");
const somaEditStatus = document.getElementById("somaEditStatus");
const rigPanel = document.getElementById("rigPanel");
const meshOpacity = document.getElementById("meshOpacity");
const modeTranslate = document.getElementById("modeTranslate");
const modeRotate = document.getElementById("modeRotate");
const modeScale = document.getElementById("modeScale");
const selectedBone = document.getElementById("selectedBone");
const weightPaintBtn = document.getElementById("weightPaintBtn");
const weightEraseBtn = document.getElementById("weightEraseBtn");
const cancelWeightToolBtn = document.getElementById("cancelWeightToolBtn");
const weightBrushRadius = document.getElementById("weightBrushRadius");
const weightBrushStrength = document.getElementById("weightBrushStrength");
const weightEditStatus = document.getElementById("weightEditStatus");
const confirmSkinBtn = document.getElementById("confirmSkinBtn");
const resetPose = document.getElementById("resetPose");
const autoRiggingOverlay = document.getElementById("autoRiggingOverlay");
const autoRiggingTitle = document.getElementById("autoRiggingTitle");
const autoRiggingStage = document.getElementById("autoRiggingStage");
const autoRiggingProgressBar = document.getElementById("autoRiggingProgressBar");
const autoRiggingProgressText = document.getElementById("autoRiggingProgressText");
const viewportInfo = document.getElementById("viewportInfo");
const ikPanel = document.getElementById("ikPanel");
const autoIkBtn = document.getElementById("autoIkBtn");
const finishIkBtn = document.getElementById("finishIkBtn");
const ikPreviewToggle = document.getElementById("ikPreviewToggle");
const ikSkeletonMode = document.getElementById("ikSkeletonMode");
const ikControlMode = document.getElementById("ikControlMode");
const ikSelectedControl = document.getElementById("ikSelectedControl");
const resetIkPreview = document.getElementById("resetIkPreview");
const ikSolverToggle = document.getElementById("ikSolverToggle");
const ikTwoBoneSolver = document.getElementById("ikTwoBoneSolver");
const ikFullBodySolver = document.getElementById("ikFullBodySolver");
const ikTransformTools = document.getElementById("ikTransformTools");
const ikModeTranslate = document.getElementById("ikModeTranslate");
const ikModeRotate = document.getElementById("ikModeRotate");
const ikSpaceLocal = document.getElementById("ikSpaceLocal");
const ikSpaceWorld = document.getElementById("ikSpaceWorld");
const fullBodySettings = document.getElementById("fullBodySettings");
const fullBodyRootBehavior = document.getElementById("fullBodyRootBehavior");
const fullBodyIterations = document.getElementById("fullBodyIterations");
const ikStatus = document.getElementById("ikStatus");

confirmSkinBtn.setAttribute("data-i18n", "bind.ik.confirmSkin");
confirmSkinBtn.textContent = t("bind.ik.confirmSkin");

const uploadStepSection = document.getElementById("uploadStepSection");
const bindStepSection = document.getElementById("bindStepSection");

function setPipelineStep(step) {
  const steps = {
    upload: uploadStepSection,
    align: alignPanel,
    bind: bindStepSection,
    confirm: confirmPanel,
    soma: somaEditPanel,
    rig: rigPanel,
    ik: ikPanel,
  };
  for (const [name, el] of Object.entries(steps)) {
    if (el) setPanelVisible(el, name === step);
  }
}

let meshId = null;
let jobId = null;
let pollTimer = null;
let currentMesh = null;
let currentMaterial = null;
let currentGeo = null;
let rigController = null;
let ikController = null;
let fullBodyIkController = null;
let somaOverlay = null;
let somaEditController = null;
let meshTransform = null;
let viewController = null;
let alignConfirmed = false;
let somaReference = null;
let basePreview = null;
let syncingAlignInputs = false;
let pendingUploadPayload = null;
let sceneReady = false;
let autoRiggingMode = false;
let autoRiggingLabel = "自动骨骼生成中";
let autoRiggingTurn = 0;
let autoRiggingRotationBases = [];
let skeletonAcceptInFlight = false;
const autoRiggingAxis = new THREE.Vector3(0, 1, 0);
const autoRiggingTurnQuaternion = new THREE.Quaternion();
const editorParams = new URLSearchParams(window.location.search);
const initialIkCharacterId = editorParams.get("mode") === "ik" ? editorParams.get("character_id") : null;
let ikSourceCharacterId = null;
let ikCompleted = false;
let ikSolverMode = "two_bone";
let ikPreviewMode = "skeleton";
let ikTransformMode = "translate";
let ikTransformSpace = "world";

function setViewportInfo(message = "") {
  if (!viewportInfo) return;
  viewportInfo.textContent = message;
  viewportInfo.classList.toggle("hidden", !message);
}

window.__meshprepsSetViewportInfo = setViewportInfo;

async function checkHealth() {
  try {
    const res = await fetch(apiUrl("/api/health"));
    const data = await res.json();
    healthEl.textContent = ta(data.message);
    healthEl.className = "health " + (data.ok ? "ok" : "err");
  } catch (e) {
    healthEl.textContent = t("bind.serviceFail");
    healthEl.className = "health err";
  }
}

checkHealth();

let ws = null;
function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}${API_PREFIX}/ws`);
  ws.onmessage = (ev) => {
    if (ev.data === "ping") ws.send("pong");
  };
  ws.onclose = () => {
    setTimeout(connectWs, 3000);
  };
}
connectWs();

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0f1115);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
camera.position.set(2.5, 1.8, 3.5);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.1);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(3, 5, 2);
scene.add(dir);
scene.add(new THREE.GridHelper(4, 20, 0x333333, 0x222222));

const meshGroup = new THREE.Group();
scene.add(meshGroup);

rigController = new RigController({
  scene,
  camera,
  canvas,
  orbitControls: controls,
  meshGroup,
  onSelect: (info) => {
    selectedBone.textContent = info ? t("bind.skin.selected", { name: info.name }) : t("bind.soma.clickSelect");
    if (skeletonSelectedBone) {
      skeletonSelectedBone.textContent = info
        ? `已选择 UniRig 骨骼：${info.name}`
        : "点击黄色 UniRig 骨骼节点进行手动验证和调整。";
    }
  },
});
ikController = new TwoBoneIKController({
  scene,
  camera,
  canvas,
  orbitControls: controls,
  rigController,
  onSelect: (info) => {
    ikSelectedControl.textContent = info ? t("bind.ik.selected", info) : t("bind.ik.none");
    if (info && ikPreviewMode === "ik") {
      const resolvedMode = ikController.setTransformMode(ikTransformMode);
      syncIkTransformButtons(resolvedMode);
      if (resolvedMode !== ikTransformMode) setViewportInfo(t("bind.ik.poleTranslateOnly"));
    }
  },
});
fullBodyIkController = new FullBodyIKController({
  scene,
  camera,
  canvas,
  orbitControls: controls,
  rigController,
  onSelect: (info) => {
    ikSelectedControl.textContent = info ? t("bind.ik.selected", info) : t("bind.ik.none");
    if (info && ikPreviewMode === "ik") {
      const resolvedMode = fullBodyIkController.setTransformMode(ikTransformMode);
      syncIkTransformButtons(resolvedMode);
    }
  },
});
somaOverlay = new SomaOverlay(scene);
somaEditController = new SomaEditController({
  scene,
  camera,
  canvas,
  orbitControls: controls,
  onSelect: (info) => {
    somaSelectedBone.textContent = info ? t("bind.skin.selected", { name: info.name }) : t("bind.soma.clickSelect");
  },
});
meshTransform = new MeshTransformController({
  scene,
  camera,
  canvas,
  orbitControls: controls,
});
viewController = new ViewController({
  perspectiveCamera: camera,
  controls,
  canvas,
  getFocusBounds,
});
viewController.setCameraHook((cam) => {
  meshTransform.setCamera(cam);
  rigController.setCamera(cam);
  ikController.setCamera(cam);
  fullBodyIkController.setCamera(cam);
  somaEditController.setCamera(cam);
});

function getFocusBounds() {
  const box = new THREE.Box3();
  if (currentMesh) {
    currentMesh.updateMatrixWorld(true);
    box.setFromObject(currentMesh);
  }
  if (somaOverlay?.root) {
    somaOverlay.root.updateMatrixWorld(true);
    box.expandByObject(somaOverlay.root);
  }
  if (rigController?.root?.children.length) {
    box.expandByObject(rigController.root);
  }
  if (ikController?.root?.visible) {
    box.expandByObject(ikController.root);
  }
  if (fullBodyIkController?.root?.visible) {
    box.expandByObject(fullBodyIkController.root);
  }
  if (somaEditController?.root?.children.length) {
    box.expandByObject(somaEditController.root);
  }
  if (box.isEmpty()) {
    box.setFromCenterAndSize(new THREE.Vector3(0, 0.8, 0), new THREE.Vector3(1.7, 1.7, 1.7));
  }
  const center = new THREE.Vector3();
  const sizeVec = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(sizeVec);
  const size = Math.max(sizeVec.length(), 0.5);
  return { center, size };
}

function syncAlignInputsFromTransform() {
  if (syncingAlignInputs) return;
  syncingAlignInputs = true;
  const trs = meshTransform.getTRS();
  alignTx.value = trs.tx.toFixed(4);
  alignTy.value = trs.ty.toFixed(4);
  alignTz.value = trs.tz.toFixed(4);
  alignRx.value = trs.rx.toFixed(2);
  alignRy.value = trs.ry.toFixed(2);
  alignRz.value = trs.rz.toFixed(2);
  alignScale.value = trs.scale.toFixed(4);
  syncingAlignInputs = false;
}

function applyAlignInputsToTransform() {
  if (syncingAlignInputs) return;
  meshTransform.setTRS({
    tx: Number(alignTx.value) || 0,
    ty: Number(alignTy.value) || 0,
    tz: Number(alignTz.value) || 0,
    rx: Number(alignRx.value) || 0,
    ry: Number(alignRy.value) || 0,
    rz: Number(alignRz.value) || 0,
    scale: Math.max(Number(alignScale.value) || 1, 0.001),
  });
}

meshTransform.onChange = () => syncAlignInputsFromTransform();

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  const activeCam = viewController.getCamera();
  if (activeCam.isPerspectiveCamera) {
    activeCam.aspect = w / Math.max(h, 1);
    activeCam.updateProjectionMatrix();
  }
  viewController.onResize();
}
window.addEventListener("resize", resize);
resize();

function beginAutoRiggingRotation() {
  const seen = new Set();
  const targets = [meshGroup, somaOverlay?.root, somaEditController?.root, rigController?.root];
  autoRiggingTurn = 0;
  autoRiggingRotationBases = [];
  for (const target of targets) {
    if (!target || seen.has(target)) continue;
    seen.add(target);
    autoRiggingRotationBases.push({ target, quaternion: target.quaternion.clone() });
  }
}

function updateAutoRiggingRotation() {
  autoRiggingTurn = (autoRiggingTurn + 0.0025) % (Math.PI * 2);
  autoRiggingTurnQuaternion.setFromAxisAngle(autoRiggingAxis, autoRiggingTurn);
  for (const entry of autoRiggingRotationBases) {
    entry.target.quaternion.copy(autoRiggingTurnQuaternion).multiply(entry.quaternion);
    entry.target.updateMatrixWorld(true);
  }
}

function restoreAutoRiggingRotation() {
  for (const entry of autoRiggingRotationBases) {
    entry.target.quaternion.copy(entry.quaternion);
    entry.target.updateMatrixWorld(true);
  }
  autoRiggingTurn = 0;
  autoRiggingRotationBases = [];
}

function animate() {
  requestAnimationFrame(animate);
  if (autoRiggingMode) {
    updateAutoRiggingRotation();
    if (currentMaterial?.emissive) {
      currentMaterial.emissiveIntensity = 0.14 + Math.sin(performance.now() * 0.004) * 0.06;
    }
  }
  controls.update();
  renderer.render(scene, viewController.getCamera());
}
animate();
sceneReady = true;
flushPendingUploadPreview();
if (initialIkCharacterId) loadExistingCharacterForIk(initialIkCharacterId);

function setPanelVisible(el, visible) {
  el.classList.toggle("hidden", !visible);
}

function setAutoRiggingUI(active, progress = 0, label = null) {
  const wasActive = autoRiggingMode;
  if (label) autoRiggingLabel = label;
  if (active && !wasActive) beginAutoRiggingRotation();
  if (!active && wasActive) restoreAutoRiggingRotation();
  autoRiggingMode = active;
  autoRiggingOverlay.classList.toggle("hidden", !active);
  viewToolbar?.classList.toggle("hidden", active);
  const percent = Math.max(0, Math.min(100, Math.round(Number(progress) || 0)));
  autoRiggingProgressBar.style.width = `${percent}%`;
  autoRiggingProgressText.textContent = `${percent}%`;
  if (autoRiggingTitle) autoRiggingTitle.textContent = autoRiggingLabel;
  if (autoRiggingStage) autoRiggingStage.textContent = "";
  if (currentMaterial?.emissive) {
    currentMaterial.emissive.setHex(active ? 0x2c68b7 : 0x000000);
    currentMaterial.emissiveIntensity = active ? 0.14 : 0;
  }
  if (!active) autoRiggingLabel = "自动骨骼生成中";
}

function setAlignMode(mode) {
  alignModeTranslate.classList.toggle("active", mode === "translate");
  alignModeRotate.classList.toggle("active", mode === "rotate");
  alignModeScale.classList.toggle("active", mode === "scale");
  meshTransform.setMode(mode);
}

function geometryToPreviewVerts(geo) {
  const pos = geo.attributes.position;
  const verts = [];
  for (let i = 0; i < pos.count; i += 1) {
    verts.push([pos.getX(i), pos.getY(i), pos.getZ(i)]);
  }
  return verts;
}

function resetMeshObjectTransform(mesh) {
  mesh.position.set(0, 0, 0);
  mesh.rotation.set(0, 0, 0);
  mesh.scale.set(1, 1, 1);
  mesh.updateMatrix();
  mesh.updateMatrixWorld(true);
}

function replaceMeshPreview(data) {
  if (!currentMesh) return;
  if (currentGeo) currentGeo.dispose();
  const verts = new Float32Array(data.vertices.flat());
  const indices = new Uint32Array(data.faces.flat());
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeVertexNormals();
  geo.computeBoundingBox();
  currentMesh.geometry = geo;
  currentGeo = geo;
  resetMeshObjectTransform(currentMesh);
}

function updateMeshGeometry(data) {
  replaceMeshPreview(data);
}

function clearScene() {
  setAutoRiggingUI(false);
  ikController.setEnabled(false);
  ikController.clear();
  fullBodyIkController.setEnabled(false);
  fullBodyIkController.clear();
  rigController.clear();
  somaOverlay.clear();
  somaEditController.clear();
  meshTransform.detach();
  setPanelVisible(rigPanel, false);
  setPanelVisible(ikPanel, false);
  setPanelVisible(confirmPanel, false);
  setPanelVisible(somaEditPanel, false);
  setPanelVisible(alignPanel, false);
  setPanelVisible(bindStepSection, false);
  currentMesh = null;
  currentMaterial = null;
  currentGeo = null;
  while (meshGroup.children.length) {
    const c = meshGroup.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
}

function createMeshFromPreview(data) {
  const verts = new Float32Array(data.vertices.flat());
  const indices = new Uint32Array(data.faces.flat());
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts.slice(), 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x8ab4f8,
    metalness: 0.1,
    roughness: 0.65,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  meshGroup.add(mesh);
  geo.computeBoundingBox();
  const box = geo.boundingBox;
  const center = new THREE.Vector3();
  box.getCenter(center);
  controls.target.copy(center);
  const size = box.getSize(new THREE.Vector3()).length();
  camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.4, size * 0.8));
  return { mesh, mat, geo, size };
}

function showUploadPreview(data) {
  clearScene();
  const { mesh, mat, geo } = createMeshFromPreview(data);
  currentMesh = mesh;
  currentMaterial = mat;
  currentGeo = geo;
  setViewActive("perspective");
  viewController.setView("perspective");
  resize();
}

function flushPendingUploadPreview() {
  if (!sceneReady || !pendingUploadPayload?.preview) return;
  try {
    showUploadPreview(pendingUploadPayload.preview);
  } catch (err) {
    if (uploadStatus) uploadStatus.textContent = t("bind.align.previewFail", { msg: err.message });
  }
}

async function ensureSomaReference() {
  if (somaReference) return somaReference;
  const res = await fetch(apiUrl("/api/soma30/reference"));
  if (!res.ok) throw new Error(t("bind.align.noSoma"));
  somaReference = await res.json();
  return somaReference;
}

function showAlignPreview(data) {
  clearScene();
  const { mesh, mat, geo, size } = createMeshFromPreview(data);
  currentMesh = mesh;
  currentMaterial = mat;
  currentGeo = geo;
  alignConfirmed = Boolean(data.align_confirmed);
  if (somaReference) {
    somaOverlay.load(somaReference, { meshSize: size });
  }
  meshTransform.attach(mesh);
  setAlignMode("translate");
  syncAlignInputsFromTransform();
  setViewActive("perspective");
  viewController.setView("perspective");
  setPipelineStep(alignConfirmed ? "bind" : "align");
  bindBtn.disabled = !alignConfirmed;
  alignStatus.textContent = "";
  setViewportInfo("将模型与红色骨骼大致对齐后确认，可提高后续流程精度");
}

function refreshAlignStatus() {
  if (!alignStatus) return;
  if (alignStatus.textContent === "") return;
  alignStatus.textContent = alignConfirmed
    ? t("bind.align.confirmed")
    : t("bind.align.needAlign");
}

function showSkeletonPreview(data) {
  clearScene();
  const { mesh, mat, size } = createMeshFromPreview(data);
  currentMesh = mesh;
  currentMaterial = mat;
  mat.transparent = true;
  mat.opacity = 0.85;
  rigController.loadSkeletonOnly(data, { mesh, material: mat, interactive: false });
  if (data.soma_overlay) {
    somaOverlay.load(data.soma_overlay, { meshSize: size });
  }
  setSkeletonTransformMode("translate");
  setSkeletonTransformSpace("local");
  if (skeletonSelectedBone) {
    skeletonSelectedBone.textContent = "点击黄色 UniRig 骨骼节点进行手动验证和调整。";
  }
  acceptSkeletonBtn.disabled = false;
  confirmStatus.textContent = t("bind.confirm.check");
  setPanelVisible(confirmPanel, true);
  setPanelVisible(somaEditPanel, false);
  setPanelVisible(rigPanel, false);
  setPipelineStep("confirm");
}

function setSkeletonTransformMode(mode) {
  if (!skeletonModeTranslate) return;
  skeletonModeTranslate.classList.toggle("active", mode === "translate");
  skeletonModeRotate.classList.toggle("active", mode === "rotate");
  skeletonModeScale.classList.toggle("active", mode === "scale");
  rigController.setTransformMode(mode);
}

function setSkeletonTransformSpace(space) {
  if (!skeletonSpaceLocal) return;
  skeletonSpaceLocal.classList.toggle("active", space === "local");
  skeletonSpaceWorld.classList.toggle("active", space === "world");
  rigController.setTransformSpace(space);
}

function showSomaEditPreview(data) {
  clearScene();
  const { mesh, mat, size } = createMeshFromPreview(data);
  currentMesh = mesh;
  currentMaterial = mat;
  mat.transparent = true;
  mat.opacity = 0.85;
  somaEditController.load(data, { meshSize: size });
  setSomaTransformMode("translate");
  setSomaTransformSpace("local");
  somaSelectedBone.textContent = "未选择骨骼";
  confirmSomaBtn.disabled = false;
  somaEditStatus.textContent = "";
  setPanelVisible(confirmPanel, false);
  setPanelVisible(somaEditPanel, true);
  setPanelVisible(rigPanel, false);
  setPipelineStep("soma");
  setViewportInfo("微调骨骼位置后确认进入蒙皮");
}

function setWeightToolActive(mode) {
  weightPaintBtn.classList.toggle("active", mode === "paint");
  weightEraseBtn.classList.toggle("active", mode === "erase");
  cancelWeightToolBtn.classList.toggle("active", false);
}

function showBindingPreview(data) {
  clearScene();
  const { mesh, mat } = createMeshFromPreview(data);
  currentMesh = mesh;
  currentMaterial = mat;
  rigController.loadBinding(data, { mesh, material: mat });
  meshOpacity.value = "85";
  rigController.setMeshOpacity(85);
  modeTranslate.classList.add("active");
  modeRotate.classList.remove("active");
  modeScale.classList.remove("active");
  rigController.setTransformMode("translate");
  rigController.resetWeightTools();
  weightBrushRadius.value = "15";
  weightBrushStrength.value = "25";
  rigController.setBrushRadiusScale(Number(weightBrushRadius.value) / 100);
  rigController.setBrushStrength(Number(weightBrushStrength.value) / 100);
  setWeightToolActive(null);
  selectedBone.textContent = "未选择骨骼";
  weightEditStatus.textContent = "";
  confirmSkinBtn.disabled = false;
  setPanelVisible(confirmPanel, false);
  setPanelVisible(somaEditPanel, false);
  setPanelVisible(rigPanel, true);
  setPipelineStep("rig");
  downloadLink.href = apiUrl(`/api/job/${jobId}/download`);
  downloadLink.classList.add("hidden");
  setViewportInfo("点选骨骼进行蒙皮权重修正");
}

function syncIkFinishLabel() {
  const key = ikSourceCharacterId ? "bind.ik.finishExisting" : "bind.ik.saveNew";
  finishIkBtn.setAttribute("data-i18n", key);
  finishIkBtn.textContent = t(key);
}

function activeIkController() {
  return ikSolverMode === "full_body" ? fullBodyIkController : ikController;
}

function isIkPreviewEnabled() {
  return Boolean(ikController.enabled || fullBodyIkController.enabled);
}

function syncIkTransformButtons(resolvedMode = ikTransformMode) {
  ikModeTranslate?.classList.toggle("active", resolvedMode === "translate");
  ikModeRotate?.classList.toggle("active", resolvedMode === "rotate");
  ikSpaceLocal?.classList.toggle("active", ikTransformSpace === "local");
  ikSpaceWorld?.classList.toggle("active", ikTransformSpace === "world");
}

function setIkTransformMode(mode) {
  ikTransformMode = mode === "rotate" ? "rotate" : "translate";
  let resolvedMode = ikTransformMode;
  if (ikPreviewMode === "ik" && ikCompleted) {
    resolvedMode = activeIkController().setTransformMode(ikTransformMode);
  } else {
    rigController.setTransformMode(ikTransformMode);
  }
  syncIkTransformButtons(resolvedMode);
  if (resolvedMode !== ikTransformMode) setViewportInfo(t("bind.ik.poleTranslateOnly"));
  else setViewportInfo(ikPreviewMode === "ik" ? t("bind.ik.transformHint") : t("bind.ik.boneTransformHint"));
}

function setIkTransformSpace(space) {
  ikTransformSpace = space === "local" ? "local" : "world";
  if (ikPreviewMode === "ik" && ikCompleted) activeIkController().setTransformSpace(ikTransformSpace);
  else rigController.setTransformSpace(ikTransformSpace);
  const resolvedMode = ikPreviewMode === "ik" && ikCompleted
    ? activeIkController().transformControls.mode : ikTransformMode;
  syncIkTransformButtons(resolvedMode);
}

function setIkSolverMode(mode) {
  const useFullBody = mode === "full_body" && Boolean(fullBodyIkController.fullBody);
  ikSolverMode = useFullBody ? "full_body" : "two_bone";
  ikController.setEnabled(false);
  fullBodyIkController.setEnabled(false);
  ikTwoBoneSolver?.classList.toggle("active", ikSolverMode === "two_bone");
  ikFullBodySolver?.classList.toggle("active", ikSolverMode === "full_body");
  fullBodySettings?.classList.toggle("hidden", ikSolverMode !== "full_body");
  if (ikPreviewMode === "ik" && ikCompleted) {
    activeIkController().syncControlsFromPose();
    activeIkController().setEnabled(true);
    setIkTransformMode(ikTransformMode);
    setIkTransformSpace(ikTransformSpace);
  }
  ikStatus.textContent = t(ikSolverMode === "full_body" ? "bind.ik.fullBodyActive" : "bind.ik.twoBoneActive");
}

function setIkPreviewMode(mode) {
  const useIkControls = mode === "ik" && ikCompleted;
  ikPreviewMode = useIkControls ? "ik" : "skeleton";
  ikSkeletonMode.classList.toggle("active", !useIkControls);
  ikControlMode.classList.toggle("active", useIkControls);
  ikController.setEnabled(false);
  fullBodyIkController.setEnabled(false);
  if (useIkControls) {
    activeIkController().syncControlsFromPose();
    activeIkController().setEnabled(true);
  } else {
    rigController.setInteractionEnabled(true);
  }
  setIkTransformMode(ikTransformMode);
  setIkTransformSpace(ikTransformSpace);
}

function applyIkProfile(profile) {
  ikController.setEnabled(false);
  ikController.clear();
  fullBodyIkController.setEnabled(false);
  fullBodyIkController.clear();
  ikCompleted = Boolean(profile);
  if (profile) {
    ikController.load(profile);
    const fullBody = profile.full_body || profile.solvers?.full_body;
    if (fullBody) fullBodyIkController.load(profile);
  }
  ikSolverMode = "two_bone";
  ikTransformMode = "translate";
  ikTransformSpace = "world";
  autoIkBtn.classList.toggle("hidden", ikCompleted);
  finishIkBtn.classList.toggle("hidden", !ikCompleted);
  ikPreviewToggle.classList.toggle("hidden", !ikCompleted);
  resetIkPreview.classList.toggle("hidden", !ikCompleted);
  ikSolverToggle?.classList.toggle("hidden", !ikCompleted);
  ikTransformTools?.classList.toggle("hidden", !ikCompleted);
  ikFullBodySolver.disabled = !Boolean(fullBodyIkController.fullBody);
  ikControlMode.disabled = !ikCompleted;
  syncIkFinishLabel();
  setIkSolverMode("two_bone");
  setIkPreviewMode(ikCompleted ? "ik" : "skeleton");
  ikStatus.textContent = ikCompleted ? t("bind.ik.done") : t("bind.ik.ready");
}

function enterIkStep({ sourceCharacterId = null, profile = null } = {}) {
  ikSourceCharacterId = sourceCharacterId;
  rigController.resetWeightTools();
  rigController.setInteractionEnabled(true);
  setPanelVisible(rigPanel, false);
  setPanelVisible(confirmPanel, false);
  setPanelVisible(somaEditPanel, false);
  setPipelineStep("ik");
  applyIkProfile(profile);
}

async function loadExistingCharacterForIk(characterId) {
  if (!characterId) return;
  setAutoRiggingUI(true, 18, t("bind.ik.binding"));
  try {
    const [characterRes, sceneRes] = await Promise.all([
      fetch(apiUrl(`/api/characters/${characterId}`)),
      fetch(apiUrl(`/api/characters/${characterId}/scene`)),
    ]);
    if (!characterRes.ok) throw new Error(await characterRes.text());
    if (!sceneRes.ok) throw new Error(await sceneRes.text());
    const character = await characterRes.json();
    const data = await sceneRes.json();
    if (character.static_asset || data.static_asset) throw new Error("static mesh assets do not support humanoid IK");
    jobId = null;
    window.__meshprepsJobId = null;
    showBindingPreview(data);
    enterIkStep({ sourceCharacterId: characterId, profile: data.ik_profile || null });
    setAutoRiggingUI(false, 100);
  } catch (error) {
    console.error(error);
    setAutoRiggingUI(false);
    setPipelineStep("upload");
    setViewportInfo(t("bind.ik.loadFailed", { msg: error.message }));
  }
}

async function runAutoIk() {
  if (!jobId && !ikSourceCharacterId) return;
  autoIkBtn.disabled = true;
  ikStatus.textContent = t("bind.ik.binding");
  setAutoRiggingUI(true, 36, t("bind.ik.binding"));
  try {
    const endpoint = ikSourceCharacterId
      ? `/api/characters/${ikSourceCharacterId}/ik/auto`
      : `/api/job/${jobId}/ik/auto`;
    const response = await fetch(apiUrl(endpoint), { method: "POST" });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    let profile = result.profile || null;
    if (!profile && ikSourceCharacterId) {
      const sceneResponse = await fetch(apiUrl(`/api/characters/${ikSourceCharacterId}/scene`));
      if (!sceneResponse.ok) throw new Error(await sceneResponse.text());
      profile = (await sceneResponse.json()).ik_profile || null;
    }
    if (!profile) throw new Error("Two-Bone IK profile is missing");
    setAutoRiggingUI(false, 100);
    applyIkProfile(profile);
  } catch (error) {
    console.error(error);
    setAutoRiggingUI(false);
    ikStatus.textContent = t("bind.ik.failed", { msg: error.message });
    setViewportInfo(t("bind.ik.failed", { msg: error.message }));
  } finally {
    autoIkBtn.disabled = false;
  }
}

async function finishIkWorkflow() {
  if (!ikCompleted) return;
  if (ikSourceCharacterId) {
    window.parent.postMessage(
      { type: "motionbuilder-character-updated", character_id: ikSourceCharacterId },
      window.location.origin,
    );
    return;
  }
  if (!jobId) return;
  finishIkBtn.disabled = true;
  try {
    const defaultName = `${t("bind.ik.defaultCharacter")}-${jobId.slice(0, 6)}`;
    const name = window.prompt(t("bind.ik.characterName"), defaultName) || defaultName;
    const assetRes = await fetch(apiUrl(`/api/characters/from-job/${jobId}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!assetRes.ok) throw new Error(await assetRes.text());
    const character = await assetRes.json();
    window.parent.postMessage({ type: "motionbuilder-character-created", character }, window.location.origin);
    resetRigWorkflow();
  } catch (error) {
    console.error(error);
    ikStatus.textContent = t("bind.ik.failed", { msg: error.message });
    finishIkBtn.disabled = false;
  }
}

async function onMeshConfirmed(data) {
  meshId = data.mesh_id;
  const p = data.preview;
  setPipelineStep("bind");
  setViewportInfo("");
  basePreview = { vertices: p.vertices, faces: p.faces };
  bindBtn.disabled = true;
  downloadLink.classList.add("hidden");
  jobStatus.textContent = "";
  confirmStatus.textContent = "";
  somaEditStatus.textContent = "";
  alignStatus.textContent = "";
  alignConfirmed = false;
  jobId = null;
  window.__meshprepsJobId = null;
  try {
    await ensureSomaReference();
    showAlignPreview(p);
  } catch (err) {
    console.error(err);
    if (uploadStatus) uploadStatus.textContent = t("bind.align.enterAlignFail", { msg: err.message });
    setPipelineStep("upload");
    setViewportInfo("无法进入对齐阶段，请重新上传模型");
  }
}

window.addEventListener("meshpreps:uploaded", (ev) => {
  pendingUploadPayload = ev.detail;
  flushPendingUploadPreview();
});

window.addEventListener("meshpreps:upload-confirmed", (ev) => {
  onMeshConfirmed(ev.detail);
});

window.addEventListener("meshpreps:upload-reset", () => {
  pendingUploadPayload = null;
  clearScene();
  setViewportInfo("");
});

function stopPoll() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollJob() {
  if (!jobId) return;
  const res = await fetch(apiUrl(`/api/job/${jobId}`));
  if (!res.ok) return;
  const st = await res.json();
  const progress = Number(st.progress) || 0;
  if (st.automatic && st.phase !== "complete" && st.state !== "error") {
    setAutoRiggingUI(true, progress);
    jobStatus.textContent = "";
  } else {
    jobStatus.textContent = "";
    if (st.phase === "skeleton" && st.state === "running") {
      setAutoRiggingUI(true, progress);
    }
  }

  if (st.phase === "skin" && st.state === "running") {
    setAutoRiggingUI(true, progress);
    somaEditStatus.textContent = "";
    return;
  }

  if (st.phase === "skeleton" && st.state === "done") {
    if (st.automatic) return;
    stopPoll();
    await acceptSkeletonAutomatically(progress);
    return;
  }

  if (st.phase === "soma_edit" && st.automatic) return;

  if (st.phase === "complete" && st.state === "done") {
    stopPoll();
    setAutoRiggingUI(false, 100);
    bindBtn.disabled = false;
    acceptSkeletonBtn.disabled = false;
    confirmSomaBtn.disabled = false;
    const prev = await fetch(apiUrl(`/api/job/${jobId}/preview`));
    if (prev.ok) {
      showBindingPreview(await prev.json());
      jobStatus.textContent = "";
    }
    return;
  }

  if (st.state === "error") {
    console.error("Rigging job failed", st.error || st.message || st);
    stopPoll();
    setAutoRiggingUI(false, progress);
    bindBtn.disabled = false;
    acceptSkeletonBtn.disabled = false;
    confirmSomaBtn.disabled = false;
    jobStatus.textContent = "";
    confirmStatus.textContent = "";
    somaEditStatus.textContent = "";
    setViewportInfo("自动绑定失败，请重新上传后重试");
    setPipelineStep("upload");
  }
}

let bindingStartInFlight = false;

async function startSkeletonBinding() {
  if (!meshId || !alignConfirmed) return;
  if (bindingStartInFlight) return;
  bindingStartInFlight = true;
  bindBtn.disabled = true;
  downloadLink.classList.add("hidden");
  jobStatus.textContent = "";
  confirmStatus.textContent = "";
  setPanelVisible(confirmPanel, false);
  setPanelVisible(somaEditPanel, false);
  setPanelVisible(rigPanel, false);
  setPanelVisible(alignPanel, false);
  setPanelVisible(bindStepSection, true);
  meshTransform.detach();
  somaOverlay.clear();
  setViewportInfo("");
  setAutoRiggingUI(true, 2, "自动骨骼生成中");
  try {
    const res = await fetch(apiUrl(`/api/bind/${meshId}`), { method: "POST" });
    if (!res.ok) {
      const err = await res.text();
      console.error("Unable to start rigging", err);
      jobStatus.textContent = "";
      setAutoRiggingUI(false);
      bindBtn.disabled = !alignConfirmed;
      setPipelineStep("align");
      setViewportInfo("自动绑定启动失败，请检查后重试");
      return;
    }
    const data = await res.json();
    jobId = data.job_id;
    window.__meshprepsJobId = jobId;
    stopPoll();
    pollTimer = setInterval(pollJob, 2000);
    pollJob();
  } catch (err) {
    console.error(err);
    jobStatus.textContent = "";
    setAutoRiggingUI(false);
    bindBtn.disabled = !alignConfirmed;
    setPipelineStep("align");
    setViewportInfo("自动绑定启动失败，请检查后重试");
  } finally {
    bindingStartInFlight = false;
  }
}

bindBtn.addEventListener("click", startSkeletonBinding);

async function acceptSkeletonAutomatically(progress = 0) {
  if (!jobId || skeletonAcceptInFlight) return;
  skeletonAcceptInFlight = true;
  acceptSkeletonBtn.disabled = true;
  confirmStatus.textContent = "";
  setPipelineStep("bind");
  setAutoRiggingUI(true, Math.max(progress, 55));
  try {
    const res = await fetch(apiUrl(`/api/job/${jobId}/skeleton/accept`), { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (!data.preview) throw new Error("SOMA preview is missing");
    showSomaEditPreview(data.preview);
    jobStatus.textContent = "";
  } catch (err) {
    console.error(err);
    setAutoRiggingUI(false);
    setPipelineStep("upload");
    setViewportInfo("骨架确认失败，请重新上传后重试");
  } finally {
    skeletonAcceptInFlight = false;
    acceptSkeletonBtn.disabled = false;
  }
}

acceptSkeletonBtn.addEventListener("click", () => acceptSkeletonAutomatically());

if (skeletonModeTranslate) skeletonModeTranslate.addEventListener("click", () => setSkeletonTransformMode("translate"));
if (skeletonModeRotate) skeletonModeRotate.addEventListener("click", () => setSkeletonTransformMode("rotate"));
if (skeletonModeScale) skeletonModeScale.addEventListener("click", () => setSkeletonTransformMode("scale"));
if (skeletonSpaceLocal) skeletonSpaceLocal.addEventListener("click", () => setSkeletonTransformSpace("local"));
if (skeletonSpaceWorld) skeletonSpaceWorld.addEventListener("click", () => setSkeletonTransformSpace("world"));
if (resetSkeletonBtn) resetSkeletonBtn.addEventListener("click", () => {
  rigController.resetPose();
  if (skeletonSelectedBone) skeletonSelectedBone.textContent = "UniRig 骨架已重置。";
});

confirmSomaBtn.addEventListener("click", async () => {
  if (!jobId) return;
  confirmSomaBtn.disabled = true;
  somaEditStatus.textContent = "";
  setPipelineStep("bind");
  setViewportInfo("");
  setAutoRiggingUI(true, 2, "自动蒙皮推理中");
  try {
    const joints = somaEditController.getJointWorldPositions();
    const bind_matrices = somaEditController.getJointWorldBindMatrices();
    const putRes = await fetch(apiUrl(`/api/job/${jobId}/soma`), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ joints, bind_matrices }),
    });
    if (!putRes.ok) {
      const err = await putRes.text();
      throw new Error(err);
    }
    const res = await fetch(apiUrl(`/api/job/${jobId}/confirm`), { method: "POST" });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }
    stopPoll();
    pollTimer = setInterval(pollJob, 2000);
    pollJob();
  } catch (err) {
    console.error(err);
    somaEditStatus.textContent = "";
    setAutoRiggingUI(false);
    setPipelineStep("soma");
    setViewportInfo("蒙皮计算启动失败，请检查后重试");
    confirmSomaBtn.disabled = false;
  }
});

function setSomaTransformMode(mode) {
  somaModeTranslate.classList.toggle("active", mode === "translate");
  somaModeRotate.classList.toggle("active", mode === "rotate");
  somaModeScale.classList.toggle("active", mode === "scale");
  somaEditController.setTransformMode(mode);
}

function setSomaTransformSpace(space) {
  somaSpaceLocal.classList.toggle("active", space === "local");
  somaSpaceWorld.classList.toggle("active", space === "world");
  somaEditController.setTransformSpace(space);
}

somaModeTranslate.addEventListener("click", () => setSomaTransformMode("translate"));
somaModeRotate.addEventListener("click", () => setSomaTransformMode("rotate"));
somaModeScale.addEventListener("click", () => setSomaTransformMode("scale"));
somaSpaceLocal.addEventListener("click", () => setSomaTransformSpace("local"));
somaSpaceWorld.addEventListener("click", () => setSomaTransformSpace("world"));
resetSomaBtn.addEventListener("click", () => {
  somaEditController.resetPose();
  somaSelectedBone.textContent = "未选择骨骼";
  setViewportInfo("变换已重设");
});

autoAlignBtn.addEventListener("click", async () => {
  if (!meshId) return;
  alignStatus.textContent = "";
  const res = await fetch(apiUrl(`/api/mesh/${meshId}/align/suggest`));
  if (!res.ok) {
    alignStatus.textContent = "";
    setViewportInfo("自动对齐失败，请手动调整后确认");
    return;
  }
  const data = await res.json();
  meshTransform.setMatrixRows(data.matrix);
  syncAlignInputsFromTransform();
  alignStatus.textContent = "";
  setViewportInfo("已应用自动对齐，请检查后确认");
});

resetAlignBtn.addEventListener("click", async () => {
  if (!meshId) return;
  try {
    const res = await fetch(apiUrl(`/api/mesh/${meshId}/align/reset`), { method: "POST" });
    if (!res.ok) {
      alignStatus.textContent = "";
      setViewportInfo("变换重设失败，请重试");
      return;
    }
    await res.json();
    const previewRes = await fetch(apiUrl(`/api/mesh/${meshId}/preview`));
    if (previewRes.ok) {
      const preview = await previewRes.json();
      replaceMeshPreview(preview);
      basePreview = { vertices: preview.vertices, faces: preview.faces };
    } else if (basePreview) {
      replaceMeshPreview(basePreview);
    }
    meshTransform.resetTargetIdentity();
    syncAlignInputsFromTransform();
    alignConfirmed = false;
    bindBtn.disabled = true;
    alignStatus.textContent = "";
    setViewportInfo("变换已重设");
  } catch (err) {
    console.error(err);
    alignStatus.textContent = "";
    setViewportInfo("变换重设失败，请重试");
  }
});

confirmAlignBtn.addEventListener("click", async () => {
  if (!meshId) return;
  confirmAlignBtn.disabled = true;
  alignStatus.textContent = "";
  try {
    const transform = meshTransform.getAlignPayload();
    const res = await fetch(apiUrl(`/api/mesh/${meshId}/align`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transform }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }
    const result = await res.json();
    if (result.preview) {
      replaceMeshPreview(result.preview);
      basePreview = {
        vertices: result.preview.vertices,
        faces: result.preview.faces,
      };
    }
    meshTransform.detach();
    alignConfirmed = Boolean(result.align_confirmed);
    bindBtn.disabled = !alignConfirmed;
    alignStatus.textContent = "";
    if (alignConfirmed) {
      setPipelineStep("bind");
      await startSkeletonBinding();
    } else {
      setViewportInfo("当前对齐无效，请调整后再次确认");
    }
  } catch (err) {
    console.error(err);
    alignStatus.textContent = "";
    setViewportInfo("确认对齐失败，请检查后重试");
  } finally {
    confirmAlignBtn.disabled = false;
  }
});

alignModeTranslate.addEventListener("click", () => setAlignMode("translate"));
alignModeRotate.addEventListener("click", () => setAlignMode("rotate"));
alignModeScale.addEventListener("click", () => setAlignMode("scale"));

for (const input of [alignTx, alignTy, alignTz, alignRx, alignRy, alignRz, alignScale]) {
  input.addEventListener("change", applyAlignInputsToTransform);
}

function setViewActive(name) {
  viewFront.classList.toggle("active", name === "front");
  viewSide.classList.toggle("active", name === "side");
  viewTop.classList.toggle("active", name === "top");
  viewPerspective.classList.toggle("active", name === "perspective");
}

viewFront.addEventListener("click", () => {
  viewController.setView("front");
  setViewActive("front");
});
viewSide.addEventListener("click", () => {
  viewController.setView("side");
  setViewActive("side");
});
viewTop.addEventListener("click", () => {
  viewController.setView("top");
  setViewActive("top");
});
viewPerspective.addEventListener("click", () => {
  viewController.setView("perspective");
  setViewActive("perspective");
});

meshOpacity.addEventListener("input", () => {
  rigController.setMeshOpacity(Number(meshOpacity.value));
});

modeTranslate.addEventListener("click", () => {
  modeTranslate.classList.add("active");
  modeRotate.classList.remove("active");
  modeScale.classList.remove("active");
  rigController.setTransformMode("translate");
});

modeRotate.addEventListener("click", () => {
  modeRotate.classList.add("active");
  modeTranslate.classList.remove("active");
  modeScale.classList.remove("active");
  rigController.setTransformMode("rotate");
});

modeScale.addEventListener("click", () => {
  modeScale.classList.add("active");
  modeTranslate.classList.remove("active");
  modeRotate.classList.remove("active");
  rigController.setTransformMode("scale");
});

resetPose.addEventListener("click", () => {
  rigController.resetPose();
  setViewportInfo("姿态已重设");
});

weightPaintBtn.addEventListener("click", () => {
  if (rigController.selectedIndex < 0) {
    weightEditStatus.textContent = "";
    setViewportInfo("请先点选需要修正的骨骼");
    return;
  }
  rigController.setWeightTool("paint");
  setWeightToolActive("paint");
  weightEditStatus.textContent = "";
  setViewportInfo("绘制模式：增加权重");
});

weightEraseBtn.addEventListener("click", () => {
  if (rigController.selectedIndex < 0) {
    weightEditStatus.textContent = "";
    setViewportInfo("请先点选需要修正的骨骼");
    return;
  }
  rigController.setWeightTool("erase");
  setWeightToolActive("erase");
  weightEditStatus.textContent = "";
  setViewportInfo("擦除模式：降低权重");
});

cancelWeightToolBtn.addEventListener("click", () => {
  rigController.setWeightTool(null);
  setWeightToolActive(null);
  weightEditStatus.textContent = "";
  setViewportInfo("工具取消：点选骨骼来验证姿态");
});

weightBrushRadius.addEventListener("input", () => {
  rigController.setBrushRadiusScale(Number(weightBrushRadius.value) / 100);
});

weightBrushStrength.addEventListener("input", () => {
  rigController.setBrushStrength(Number(weightBrushStrength.value) / 100);
});

autoIkBtn.addEventListener("click", runAutoIk);
ikSkeletonMode.addEventListener("click", () => setIkPreviewMode("skeleton"));
ikControlMode.addEventListener("click", () => setIkPreviewMode("ik"));
ikTwoBoneSolver?.addEventListener("click", () => setIkSolverMode("two_bone"));
ikFullBodySolver?.addEventListener("click", () => setIkSolverMode("full_body"));
ikModeTranslate?.addEventListener("click", () => setIkTransformMode("translate"));
ikModeRotate?.addEventListener("click", () => setIkTransformMode("rotate"));
ikSpaceLocal?.addEventListener("click", () => setIkTransformSpace("local"));
ikSpaceWorld?.addEventListener("click", () => setIkTransformSpace("world"));
fullBodyRootBehavior?.addEventListener("change", () => {
  fullBodyIkController.setRootBehavior(fullBodyRootBehavior.value);
});
fullBodyIterations?.addEventListener("change", () => {
  fullBodyIkController.setIterations(fullBodyIterations.value);
});
resetIkPreview.addEventListener("click", () => {
  activeIkController().reset();
  setIkPreviewMode("ik");
  setViewportInfo(t("bind.ik.transformHint"));
});
finishIkBtn.addEventListener("click", finishIkWorkflow);

confirmSkinBtn.addEventListener("click", async () => {
  if (!jobId) return;
  confirmSkinBtn.disabled = true;
  weightEditStatus.textContent = "";
  setViewportInfo(t("bind.ik.confirmingSkin"));
  try {
    const payload = rigController.getLbsPayload();
    const res = await fetch(apiUrl(`/api/job/${jobId}/skin/complete`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }
    await res.json();
    enterIkStep({ sourceCharacterId: null, profile: null });
  } catch (err) {
    console.error(err);
    weightEditStatus.textContent = "";
    setViewportInfo(t("bind.ik.skinConfirmFailed", { msg: err.message }));
    confirmSkinBtn.disabled = false;
  }
});

function resetRigWorkflow() {
  stopPoll();
  setAutoRiggingUI(false);
  meshId = null;
  jobId = null;
  ikSourceCharacterId = null;
  ikCompleted = false;
  ikController.setEnabled(false);
  ikController.clear();
  fullBodyIkController.setEnabled(false);
  fullBodyIkController.clear();
  ikSolverMode = "two_bone";
  ikPreviewMode = "skeleton";
  ikTransformMode = "translate";
  ikTransformSpace = "world";
  alignConfirmed = false;
  basePreview = null;
  pendingUploadPayload = null;
  skeletonAcceptInFlight = false;
  window.__meshprepsJobId = null;
  bindBtn.disabled = true;
  acceptSkeletonBtn.disabled = false;
  confirmSomaBtn.disabled = false;
  confirmSkinBtn.disabled = false;
  finishIkBtn.disabled = false;
  uploadStatus.textContent = "";
  alignStatus.textContent = "";
  jobStatus.textContent = "";
  confirmStatus.textContent = "";
  somaEditStatus.textContent = "";
  weightEditStatus.textContent = "";
  ikStatus.textContent = "";
  autoIkBtn.classList.remove("hidden");
  finishIkBtn.classList.add("hidden");
  ikPreviewToggle.classList.add("hidden");
  resetIkPreview.classList.add("hidden");
  ikSolverToggle?.classList.add("hidden");
  ikTransformTools?.classList.add("hidden");
  downloadLink.classList.add("hidden");
  setViewportInfo("");
  if (typeof window.__meshprepsSetUploadIdle === "function") {
    window.__meshprepsSetUploadIdle();
  } else {
    clearScene();
  }
  setPipelineStep("upload");
}

setPipelineStep("upload");

window.MKA_I18N?.onChange(() => {
  window.MKA_I18N.applyDom();
  refreshAlignStatus();
  syncIkFinishLabel();
  if (!ikPanel.classList.contains("hidden")) {
    ikStatus.textContent = ikCompleted ? t("bind.ik.done") : t("bind.ik.ready");
    if (isIkPreviewEnabled()) setViewportInfo(t("bind.ik.transformHint"));
    else if (ikCompleted) setViewportInfo(t("bind.ik.boneTransformHint"));
  }
});

window.__meshprepsGetLbsPayload = () => (rigController ? rigController.getLbsPayload() : null);
window.__meshprepsJobId = null;
