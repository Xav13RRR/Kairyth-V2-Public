import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const $ = (id) => document.getElementById(id);
const CONTENT_LAYER = 0;
const HELPER_LAYER = 31;
const CAMERA_HELPER_NEAR = 0.04;
const CAMERA_HELPER_FAR = 1.25;
const CAMERA_COLOR = 0xe4ff4e;
const SELECTED_CAMERA_COLOR = 0x2f7cff;
const DIRECTOR_CUT_PALETTE = ["#e4ff4e", "#49e6ff", "#ff8a3d", "#ca71ff", "#45d483", "#ff5d8f", "#ffd166", "#5f8cff", "#f78cce", "#74d3ae", "#ff6b6b", "#8bd450"];
const BLACKOUT_COLOR = "#111111";
const TIMELINE_LABEL_WIDTH = 124;
const TIMELINE_DRAG_THRESHOLD = 5;
const ASPECT_RATIOS = { "16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1, "21:9": 21 / 9 };
const OUTPUT_RESOLUTIONS = {
  "16:9": { "720p": [1280, 720], "1080p": [1920, 1080] },
  "9:16": { "720p": [720, 1280], "1080p": [1080, 1920] },
  "1:1": { "720p": [720, 720], "1080p": [1080, 1080] },
  "21:9": { "720p": [1680, 720], "1080p": [2520, 1080] },
};

const state = {
  initialized: false,
  projects: [],
  project: null,
  characters: [],
  animations: [],
  terrains: [],
  selected: null,
  selectedCameraId: null,
  frame: 0,
  playing: false,
  lastTick: 0,
  lockDirectorView: false,
  runtime: null,
  actorPackets: new Map(),
  assetPackets: new Map(),
  visuals: new Map(),
  cameraVisuals: new Map(),
  pathVisuals: new Map(),
  thumbnails: new Map(),
  undo: [],
  redo: [],
  saveTimer: 0,
  saving: false,
  pendingSave: false,
  selectedProjectId: null,
  cardMenu: null,
  exportController: null,
  exportJob: null,
};

function tr(zh, en) {
  return window.MotionBuilderI18n?.getLanguage?.() === "en" ? en : zh;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function uid(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function directorSegments() {
  if (!state.project) return [];
  if (!Array.isArray(state.project.camera_cuts)) state.project.camera_cuts = [];
  return state.project.camera_cuts;
}

function isBlackoutSegment(segment) {
  return segment?.kind === "blackout";
}

function cameraSegments() {
  return directorSegments().filter((segment) => !isBlackoutSegment(segment));
}

function segmentLength(segment) {
  return Math.max(1, Math.round(Number(segment?.end_frame || 0) - Number(segment?.start_frame || 0)));
}

function hslColorHex(hue, saturation = 76, lightness = 58) {
  const s = Math.max(0, Math.min(100, saturation)) / 100;
  const l = Math.max(0, Math.min(100, lightness)) / 100;
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const sector = ((Number(hue) % 360) + 360) % 360 / 60;
  const secondary = chroma * (1 - Math.abs(sector % 2 - 1));
  const [red, green, blue] = sector < 1 ? [chroma, secondary, 0]
    : sector < 2 ? [secondary, chroma, 0]
      : sector < 3 ? [0, chroma, secondary]
        : sector < 4 ? [0, secondary, chroma]
          : sector < 5 ? [secondary, 0, chroma]
            : [chroma, 0, secondary];
  const match = l - chroma / 2;
  return `#${[red, green, blue].map((channel) => Math.round((channel + match) * 255).toString(16).padStart(2, "0")).join("")}`;
}

function uniqueSegmentColor(usedColors, seed = 0) {
  const paletteColor = DIRECTOR_CUT_PALETTE.find((color) => !usedColors.has(color.toLowerCase()));
  if (paletteColor) return paletteColor;
  for (let attempt = 0; attempt < 2048; attempt += 1) {
    const band = Math.floor(attempt / 360) % 4;
    const color = hslColorHex(seed * 137.508 + attempt * 47.331, 72 + (band % 2) * 8, 55 + [0, 9, -7, 4][band]);
    if (!usedColors.has(color.toLowerCase())) return color;
  }
  return "#e4ff4e";
}

function nextSegmentColor(excludedId = "") {
  const used = new Set(cameraSegments().filter((segment) => segment.id !== excludedId).map((segment) => String(segment.color || "").toLowerCase()));
  return uniqueSegmentColor(used, cameraSegments().length);
}

function reindexDirectorSegments() {
  directorSegments().forEach((segment, index) => { segment.order = index; });
}

function ensureDirectorSegments() {
  if (!state.project) return false;
  const before = JSON.stringify([state.project.camera_cuts || [], state.project.duration_frames]);
  const rows = directorSegments().map((segment, index) => ({ segment, index }));
  rows.sort((left, right) => {
    const leftOrder = Number.isFinite(Number(left.segment.order)) ? Number(left.segment.order) : left.index;
    const rightOrder = Number.isFinite(Number(right.segment.order)) ? Number(right.segment.order) : right.index;
    return leftOrder - rightOrder || Number(left.segment.start_frame || 0) - Number(right.segment.start_frame || 0);
  });
  state.project.camera_cuts = rows.map(({ segment }) => segment);
  const usedColors = new Set();
  let previousEnd = 0;
  for (let index = 0; index < state.project.camera_cuts.length; index += 1) {
    const segment = state.project.camera_cuts[index];
    segment.kind = segment.kind === "blackout" ? "blackout" : "camera";
    segment.order = index;
    const length = segmentLength(segment);
    segment.start_frame = Math.max(0, Math.round(Number(segment.start_frame) || 0));
    if (segment.start_frame < previousEnd) segment.start_frame = previousEnd;
    segment.end_frame = segment.start_frame + length;
    if (isBlackoutSegment(segment)) {
      segment.camera_id = "";
      segment.color = BLACKOUT_COLOR;
      segment.number ||= `BLACK ${String(index + 1).padStart(2, "0")}`;
      segment.name ||= tr("黑幕", "Blackout");
    } else {
      const requested = String(segment.color || "").toLowerCase();
      segment.color = /^#[0-9a-f]{6}$/i.test(requested) && !usedColors.has(requested) ? requested : uniqueSegmentColor(usedColors, index);
      usedColors.add(segment.color.toLowerCase());
    }
    segment.thumbnail_frame = Math.max(segment.start_frame, Math.min(segment.end_frame - 1, Math.round(Number(segment.thumbnail_frame ?? segment.start_frame))));
    previousEnd = segment.end_frame;
  }
  state.project.duration_frames = Math.max(1, Number(state.project.duration_frames) || 1, previousEnd);
  return before !== JSON.stringify([state.project.camera_cuts, state.project.duration_frames]);
}

function compactDirectorSegments(startFrame = 0) {
  let cursor = Math.max(0, Math.round(Number(startFrame) || 0));
  for (const segment of directorSegments()) {
    const length = segmentLength(segment);
    segment.start_frame = cursor;
    segment.end_frame = cursor + length;
    segment.thumbnail_frame = Math.max(segment.start_frame, Math.min(segment.end_frame - 1, Number(segment.thumbnail_frame ?? segment.start_frame)));
    cursor = segment.end_frame;
  }
  reindexDirectorSegments();
  if (state.project) state.project.duration_frames = Math.max(Number(state.project.duration_frames) || 1, cursor);
}

function shiftDirectorSegments(index, delta) {
  const amount = Math.round(Number(delta) || 0);
  if (!amount) return;
  const rows = directorSegments();
  for (let cursor = Math.max(0, index); cursor < rows.length; cursor += 1) {
    rows[cursor].start_frame += amount;
    rows[cursor].end_frame += amount;
    rows[cursor].thumbnail_frame += amount;
  }
}

function directorInsertIndex(frame = state.frame) {
  const rows = directorSegments();
  if (!rows.length) return 0;
  const value = Math.max(0, Number(frame) || 0);
  const inside = rows.findIndex((segment) => value >= segment.start_frame && value < segment.end_frame);
  if (inside >= 0) return value < (rows[inside].start_frame + rows[inside].end_frame) / 2 ? inside : inside + 1;
  let previous = -1;
  rows.forEach((segment, index) => { if (segment.end_frame <= value) previous = index; });
  return previous + 1;
}

function frameSeconds(frame) {
  return (Number(frame || 0) / Math.max(1, Number(state.project?.fps || 30))).toFixed(3);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.text()) || `${response.status}`);
  const type = response.headers.get("content-type") || "";
  return type.includes("json") ? response.json() : response;
}

function jsonOptions(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

function status(zh, en = zh, isError = false) {
  state.lastStatus = { zh, en, isError };
  const node = $("directorEditorStatus") || $("directorWarehouseStatus");
  if (!node) return;
  node.textContent = tr(zh, en);
  node.classList.toggle("error", Boolean(isError));
}

function projectCountLabel(count) {
  return tr(`${count} 个导演工程`, `${count} director project${count === 1 ? "" : "s"}`);
}

function defaultTransform(position = [0, 0, 0], quaternion = [0, 0, 0, 1]) {
  return { translation: [...position], quaternion: [...quaternion], uniform_scale: 1 };
}

function outputResolution(project = state.project) {
  if (!project) return [1920, 1080];
  if (project.aspect_preset === "custom" || project.resolution_preset === "custom") return project.custom_resolution || [1920, 1080];
  return OUTPUT_RESOLUTIONS[project.aspect_preset]?.[project.resolution_preset] || [1920, 1080];
}

function aspectRatio(project = state.project) {
  if (!project) return 16 / 9;
  const [width, height] = outputResolution(project);
  return ASPECT_RATIOS[project.aspect_preset] || width / Math.max(1, height);
}

function projectSummary(row) {
  const seconds = Number(row.duration_frames || 0) / Math.max(1, Number(row.fps || 30));
  return `${row.aspect_preset || "16:9"} · ${row.resolution_preset || "1080p"}\n${row.camera_count || 0} CAM · ${row.shot_count || 0} SHOT\n${seconds.toFixed(1)} S`;
}

function projectCard(row) {
  const selected = state.selectedProjectId === row.id;
  return `<article class="character-card asset-tech-card director-project-card ${selected ? "selected" : ""}" data-director-card="${escapeHtml(row.id)}">
    <div class="asset-wireframe-slot"><div class="director-card-preview">${escapeHtml(row.shot_count ? `SHOT ${String(row.shot_count).padStart(2, "0")}` : "DIRECTOR")}</div></div>
    <div class="asset-card-head"><h3 class="asset-name">${escapeHtml(row.name)}</h3><span class="asset-kind">.DIRECTOR</span></div>
    <p class="asset-meta">${escapeHtml(projectSummary(row)).replace(/\n/g, "<br>")}</p>
  </article>`;
}

function hideProjectMenu() {
  state.cardMenu?.classList.add("hidden");
  state.selectedProjectId = null;
  document.querySelectorAll("[data-director-card].selected").forEach((card) => card.classList.remove("selected"));
}

function ensureProjectMenu() {
  if (state.cardMenu) return state.cardMenu;
  const menu = document.createElement("div");
  menu.className = "director-project-actions hidden";
  menu.addEventListener("click", (event) => event.stopPropagation());
  document.body.appendChild(menu);
  state.cardMenu = menu;
  return menu;
}

function showProjectMenu(card, row) {
  const menu = ensureProjectMenu();
  menu.innerHTML = `<button data-action="open">${tr("打开工程", "Open")}</button><button data-action="rename">${tr("重命名", "Rename")}</button><button data-action="duplicate">${tr("复制", "Duplicate")}</button><button data-action="snapshot">${tr("版本快照", "Snapshot")}</button><button data-action="delete">${tr("删除", "Delete")}</button>`;
  const rect = card.getBoundingClientRect();
  menu.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - 420))}px`;
  menu.style.top = `${Math.max(8, Math.min(rect.bottom + 7, innerHeight - 60))}px`;
  menu.classList.remove("hidden");
  menu.querySelector("[data-action='open']").onclick = () => openProject(row.id);
  menu.querySelector("[data-action='rename']").onclick = async () => {
    await openProject(row.id);
    $("directorProjectName").focus();
    $("directorProjectName").select();
  };
  menu.querySelector("[data-action='duplicate']").onclick = async () => {
    await api(`/api/director-projects/${encodeURIComponent(row.id)}/duplicate`, jsonOptions("POST", { name: `${row.name} Copy` }));
    await loadProjects();
  };
  menu.querySelector("[data-action='snapshot']").onclick = async () => {
    const name = `${tr("手动快照", "Manual Snapshot")} ${new Date().toLocaleString()}`;
    await api(`/api/director-projects/${encodeURIComponent(row.id)}/snapshots`, jsonOptions("POST", { name, kind: "manual" }));
    status("导演工程快照已保存。", "Director project snapshot saved.");
  };
  menu.querySelector("[data-action='delete']").onclick = async () => {
    if (!confirm(tr("删除该导演工程？此操作不会删除引用的模型和动画。", "Delete this director project? Referenced assets will not be removed."))) return;
    await api(`/api/director-projects/${encodeURIComponent(row.id)}`, { method: "DELETE" });
    await loadProjects();
  };
}

function renderProjects() {
  const grid = $("directorProjectGrid");
  if (!grid) return;
  grid.innerHTML = state.projects.length ? state.projects.map(projectCard).join("") : `<div class="empty">${tr("暂无导演工程", "No director projects")}</div>`;
  grid.querySelectorAll("[data-director-card]").forEach((card) => {
    card.onclick = (event) => {
      event.stopPropagation();
      const row = state.projects.find((item) => item.id === card.dataset.directorCard);
      if (!row) return;
      state.selectedProjectId = row.id;
      grid.querySelectorAll("[data-director-card].selected").forEach((item) => item.classList.remove("selected"));
      card.classList.add("selected");
      showProjectMenu(card, row);
    };
    card.ondblclick = () => openProject(card.dataset.directorCard);
  });
}

async function loadProjects() {
  try {
    state.projects = await api("/api/director-projects");
    hideProjectMenu();
    renderProjects();
    $("directorWarehouseStatus").textContent = state.projects.length ? projectCountLabel(state.projects.length) : "";
  } catch (error) {
    $("directorWarehouseStatus").textContent = error.message;
  }
}

function syncDirectorLanguageUi() {
  const english = window.MotionBuilderI18n?.getLanguage?.() === "en";
  $("directorLanguageZh")?.classList.toggle("active", !english);
  $("directorLanguageEn")?.classList.toggle("active", english);
  $("directorLanguageZh")?.setAttribute("aria-pressed", String(!english));
  $("directorLanguageEn")?.setAttribute("aria-pressed", String(english));
  if ($("directorWarehouseStatus")) $("directorWarehouseStatus").textContent = state.projects.length ? projectCountLabel(state.projects.length) : "";
}

async function loadCatalogs() {
  const [characters, animations, terrains] = await Promise.all([
    api("/api/characters"),
    api("/api/animations"),
    api("/api/terrains"),
  ]);
  state.characters = characters;
  state.animations = animations;
  state.terrains = terrains;
  refreshCatalogInputs();
}

function isStaticAsset(item) {
  return Boolean(item?.static_asset || item?.asset_kind === "static_mesh");
}

function refreshCatalogInputs() {
  const character = $("directorCharacterSelect");
  const riggedCharacters = state.characters.filter((item) => !isStaticAsset(item));
  if (character) {
    const previous = character.value;
    character.innerHTML = riggedCharacters.length
      ? riggedCharacters.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join("")
      : `<option value="">${tr("暂无骨骼网格体", "No skeletal meshes")}</option>`;
    if (riggedCharacters.some((item) => item.id === previous)) character.value = previous;
  }
  refreshDirectorAnimationOptions();
  refreshAssetSelect();
}

function refreshDirectorAnimationOptions() {
  const characterId = $("directorCharacterSelect")?.value || "";
  const animation = $("directorAnimationSelect");
  if (!animation) return;
  const previous = animation.value;
  const rows = state.animations.filter((item) => item.character_id === characterId);
  animation.innerHTML = rows.length
    ? rows.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join("")
    : `<option value="">${tr("该网格体暂无动画", "No animations for this skeletal mesh")}</option>`;
  if (rows.some((item) => item.id === previous)) animation.value = previous;
  refreshDirectorRootMotionOption();
}

function refreshDirectorRootMotionOption() {
  const checkbox = $("directorRootMotionEnabled");
  if (!checkbox) return;
  const animation = state.animations.find((item) => item.id === $("directorAnimationSelect")?.value);
  const available = Boolean(animation?.has_root_motion);
  checkbox.disabled = !available;
  checkbox.checked = available;
  checkbox.closest("label")?.classList.toggle("disabled", !available);
}

function refreshAssetSelect() {
  const select = $("directorAssetSelect");
  if (!select) return;
  const rows = $("directorAssetKind")?.value === "terrain" ? state.terrains : state.characters.filter(isStaticAsset);
  select.innerHTML = rows.length ? rows.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join("") : `<option value="">${tr("暂无资产", "No assets")}</option>`;
}

async function createProject() {
  const name = tr("未命名导演工程", "Untitled Director Project");
  const project = await api("/api/director-projects", jsonOptions("POST", { name, project: {} }));
  await loadProjects();
  await openProject(project.id);
}

function initRuntime() {
  if (state.runtime) return state.runtime;
  const canvas = $("directorCanvas");
  const pipCanvas = $("directorCameraPreview");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x080b11, 1);
  const pipRenderer = new THREE.WebGLRenderer({ canvas: pipCanvas, antialias: true, alpha: false });
  pipRenderer.setPixelRatio(1);
  pipRenderer.outputColorSpace = THREE.SRGBColorSpace;
  pipRenderer.setClearColor(0x080b11, 1);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080b11);
  const editorCamera = new THREE.PerspectiveCamera(42, 16 / 9, 0.01, 500);
  editorCamera.position.set(2.8, 1.8, 4.2);
  editorCamera.layers.enable(HELPER_LAYER);
  const controls = new OrbitControls(editorCamera, canvas);
  controls.target.set(0, 1, 0);
  controls.enableDamping = true;
  const grid = new THREE.GridHelper(20, 40, 0x526274, 0x263342);
  grid.layers.set(HELPER_LAYER);
  scene.add(grid);
  scene.add(new THREE.HemisphereLight(0xe7efff, 0x1b2230, 2.25));
  const key = new THREE.DirectionalLight(0xffffff, 2.8);
  key.position.set(4, 7, 5);
  scene.add(key);
  const gizmo = new TransformControls(editorCamera, canvas);
  gizmo.setMode("translate");
  gizmo.setSpace("world");
  gizmo.getRaycaster().layers.enable(HELPER_LAYER);
  const gizmoHelper = gizmo.getHelper();
  gizmoHelper.traverse((object) => object.layers.set(HELPER_LAYER));
  scene.add(gizmoHelper);
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const runtime = { canvas, pipCanvas, renderer, pipRenderer, scene, editorCamera, controls, grid, key, gizmo, raycaster, pointer, dragging: false, renderCamera: editorCamera };
  gizmo.addEventListener("dragging-changed", (event) => {
    runtime.dragging = event.value;
    if (event.value) runtime.gizmoBefore = projectSnapshot();
    controls.enabled = !event.value && !state.lockDirectorView;
    if (!event.value) commitGizmoTransform();
  });
  gizmo.addEventListener("objectChange", syncSelectedFromVisual);
  canvas.addEventListener("pointerdown", selectFromViewport);
  renderer.setAnimationLoop(tick);
  state.runtime = runtime;
  resizeRuntime();
  return runtime;
}

function resizeRuntime() {
  const runtime = state.runtime;
  if (!runtime || $("directorEditorModal")?.classList.contains("hidden")) return;
  const rect = runtime.canvas.getBoundingClientRect();
  const quality = Number(state.project?.preview_quality || 0.5);
  runtime.renderer.setPixelRatio(Math.max(0.25, Math.min(devicePixelRatio, 2) * quality));
  runtime.renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
  runtime.editorCamera.aspect = rect.width / Math.max(1, rect.height);
  runtime.editorCamera.updateProjectionMatrix();
  const pipRect = runtime.pipCanvas.getBoundingClientRect();
  runtime.pipRenderer.setSize(Math.max(1, pipRect.width), Math.max(1, pipRect.height), false);
}

function fittedViewport(width, height, ratio = aspectRatio()) {
  const safeWidth = Math.max(1, Number(width) || 1);
  const safeHeight = Math.max(1, Number(height) || 1);
  const safeRatio = Math.max(0.01, Number(ratio) || 16 / 9);
  if (safeWidth / safeHeight > safeRatio) {
    const fittedWidth = safeHeight * safeRatio;
    return { x: (safeWidth - fittedWidth) / 2, y: 0, width: fittedWidth, height: safeHeight };
  }
  const fittedHeight = safeWidth / safeRatio;
  return { x: 0, y: (safeHeight - fittedHeight) / 2, width: safeWidth, height: fittedHeight };
}

function syncGuideViewport(viewport = null) {
  const overlay = $("directorGuideOverlay");
  if (!overlay) return;
  const viewportKey = viewport
    ? `${viewport.x.toFixed(2)}:${viewport.y.toFixed(2)}:${viewport.width.toFixed(2)}:${viewport.height.toFixed(2)}`
    : "full";
  if (overlay.dataset.viewportKey === viewportKey) return;
  overlay.dataset.viewportKey = viewportKey;
  if (!viewport) {
    overlay.style.inset = "0";
    overlay.style.width = "auto";
    overlay.style.height = "auto";
    return;
  }
  overlay.style.inset = "auto";
  overlay.style.left = `${viewport.x}px`;
  overlay.style.top = `${viewport.y}px`;
  overlay.style.width = `${viewport.width}px`;
  overlay.style.height = `${viewport.height}px`;
}

function renderMainViewport(runtime, camera, directorView = false, blackout = false) {
  const rect = runtime.canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const previousClearColor = runtime.renderer.getClearColor(new THREE.Color()).clone();
  const previousClearAlpha = runtime.renderer.getClearAlpha();
  if (blackout) runtime.renderer.setClearColor(0x000000, 1);
  runtime.renderer.setScissorTest(false);
  runtime.renderer.setViewport(0, 0, width, height);
  runtime.renderer.clear(true, true, true);
  if (blackout) {
    syncGuideViewport(directorView ? fittedViewport(width, height) : null);
    runtime.renderer.setClearColor(previousClearColor, previousClearAlpha);
    return;
  }
  if (!directorView) {
    syncGuideViewport();
    runtime.renderer.render(runtime.scene, camera);
    return;
  }
  const viewport = fittedViewport(width, height);
  runtime.renderer.setViewport(viewport.x, viewport.y, viewport.width, viewport.height);
  runtime.renderer.setScissor(viewport.x, viewport.y, viewport.width, viewport.height);
  runtime.renderer.setScissorTest(true);
  syncGuideViewport(viewport);
  runtime.renderer.render(runtime.scene, camera);
  runtime.renderer.setScissorTest(false);
  runtime.renderer.setViewport(0, 0, width, height);
}

function clearRuntimeScene() {
  const runtime = state.runtime;
  if (!runtime) return;
  runtime.gizmo.detach();
  for (const visual of [...state.visuals.values(), ...state.cameraVisuals.values(), ...state.pathVisuals.values()]) {
    runtime.scene.remove(visual.group || visual);
    if (visual.helper) {
      runtime.scene.remove(visual.helper);
      visual.helper.dispose?.();
    }
    (visual.group || visual).traverse?.((object) => {
      object.geometry?.dispose?.();
      if (Array.isArray(object.material)) object.material.forEach((material) => { material.map?.dispose?.(); material.dispose?.(); });
      else { object.material?.map?.dispose?.(); object.material?.dispose?.(); }
    });
  }
  state.visuals.clear();
  state.cameraVisuals.clear();
  state.pathVisuals.clear();
  state.actorPackets.clear();
  state.assetPackets.clear();
}

function makeGeometry(sceneData, skinned = false) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute((sceneData.vertices || []).flat(), 3));
  geometry.setIndex((sceneData.faces || []).flat());
  geometry.computeVertexNormals();
  if (skinned) {
    geometry.setAttribute("skinIndex", new THREE.Uint16BufferAttribute((sceneData.lbs_indices || []).flat(), 4));
    geometry.setAttribute("skinWeight", new THREE.Float32BufferAttribute((sceneData.lbs_weights || []).flat(), 4));
  }
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function makeBones(sceneData) {
  const parents = sceneData.parents || [];
  const world = (sceneData.target_bind || []).map((row) => new THREE.Matrix4().set(...row));
  const bones = parents.map((_, index) => {
    const bone = new THREE.Bone();
    bone.name = sceneData.names?.[index] || `Joint_${index}`;
    return bone;
  });
  parents.forEach((parent, index) => {
    const local = parent < 0 ? world[index] : world[parent].clone().invert().multiply(world[index]);
    local.decompose(bones[index].position, bones[index].quaternion, bones[index].scale);
    if (parent >= 0) bones[parent].add(bones[index]);
  });
  return { bones, world };
}

function rootMotionPath(sceneData) {
  return sceneData.virtual_root_path?.length > 1
    ? sceneData.virtual_root_path
    : sceneData.root_motion_path?.length > 1
      ? sceneData.root_motion_path
      : [];
}

function makeRootMotionVisual(sceneData) {
  const path = rootMotionPath(sceneData);
  if (path.length < 2) return null;
  const group = new THREE.Group();
  const points = path.map((point) => new THREE.Vector3(Number(point[0]) || 0, Number(point[1]) || 0, Number(point[2]) || 0));
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0x49e6ff, transparent: true, opacity: 0.92, depthTest: false }),
  );
  line.renderOrder = 8;
  const start = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), new THREE.MeshBasicMaterial({ color: 0xe4ff4e, depthTest: false }));
  const end = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff5c7c, depthTest: false }));
  start.position.copy(points[0]);
  end.position.copy(points.at(-1));
  start.renderOrder = end.renderOrder = 9;
  group.add(line, start, end);
  group.traverse((object) => object.layers.set(HELPER_LAYER));
  return group;
}

function applyItemTransform(group, item) {
  const transform = item.transform || defaultTransform();
  group.position.fromArray(transform.translation || [0, 0, 0]);
  group.quaternion.fromArray(transform.quaternion || [0, 0, 0, 1]).normalize();
  group.scale.setScalar(Number(transform.uniform_scale || 1));
}

async function fetchAllFrames(animationId, count) {
  const frames = [];
  for (let start = 0; start < count; start += 240) {
    const packet = await api(`/api/animations/${encodeURIComponent(animationId)}/frames?from_frame=${start}&to_frame=${Math.min(count, start + 240)}`);
    frames.push(...(packet.world || []));
  }
  return frames;
}

async function loadActorPacket(item) {
  const animation = state.animations.find((row) => row.id === item.source_animation_id);
  const characterId = item.source_asset_id && item.source_asset_id !== "unknown"
    ? item.source_asset_id
    : animation?.character_id;
  if (!characterId) throw new Error(tr("动画没有关联骨骼网格体", "The animation has no associated skeletal mesh"));
  const key = `${characterId}:${item.source_animation_id}`;
  if (state.actorPackets.has(key)) return state.actorPackets.get(key);
  const [animationScene, characterScene] = await Promise.all([
    api(`/api/animations/${encodeURIComponent(item.source_animation_id)}/scene`),
    api(`/api/characters/${encodeURIComponent(characterId)}/scene`),
  ]);
  const scene = {
    ...animationScene,
    vertices: characterScene.vertices || animationScene.vertices,
    faces: characterScene.faces || animationScene.faces,
    lbs_indices: characterScene.lbs_indices || animationScene.lbs_indices,
    lbs_weights: characterScene.lbs_weights || animationScene.lbs_weights,
    target_bind: characterScene.target_bind || characterScene.bind_matrices || animationScene.target_bind,
    parents: characterScene.parents || animationScene.parents,
    names: characterScene.names || animationScene.names,
  };
  const frames = await fetchAllFrames(item.source_animation_id, animationScene.num_frames || 1);
  const jointCount = scene.target_bind?.length || 0;
  const frameJointCount = Array.isArray(frames[0]) ? frames[0].length : 0;
  if (!jointCount || scene.parents?.length !== jointCount || (frameJointCount && frameJointCount !== jointCount)) {
    throw new Error(tr("所选动画与骨骼网格体骨架不兼容", "The selected animation is incompatible with this skeletal mesh"));
  }
  const packet = { scene, frames };
  state.actorPackets.set(key, packet);
  return packet;
}

async function loadAssetPacket(item) {
  const prefix = item.kind === "terrain" ? "/api/terrains" : "/api/characters";
  const key = `${item.kind}:${item.source_asset_id}`;
  if (state.assetPackets.has(key)) return state.assetPackets.get(key);
  const packet = await api(`${prefix}/${encodeURIComponent(item.source_asset_id)}/scene`);
  state.assetPackets.set(key, packet);
  return packet;
}

async function buildSceneItem(item) {
  const runtime = initRuntime();
  const group = new THREE.Group();
  group.name = item.name;
  group.userData.directorType = "item";
  group.userData.directorId = item.id;
  let visual;
  if (item.kind === "skeletal_actor") {
    const packet = await loadActorPacket(item);
    const geometry = makeGeometry(packet.scene, true);
    const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(item.material_override?.color || "#e4ff4e"), roughness: 0.68, metalness: 0.02, side: THREE.DoubleSide });
    const built = makeBones(packet.scene);
    const skeleton = new THREE.Skeleton(built.bones, built.world.map((matrix) => matrix.clone().invert()));
    const mesh = new THREE.SkinnedMesh(geometry, material);
    const rootIndex = (packet.scene.parents || []).findIndex((parent) => parent < 0);
    mesh.add(built.bones[Math.max(0, rootIndex)]);
    mesh.bind(skeleton, new THREE.Matrix4());
    mesh.userData.directorType = "item";
    mesh.userData.directorId = item.id;
    group.add(mesh);
    const rootMotion = makeRootMotionVisual(packet.scene);
    if (rootMotion) group.add(rootMotion);
    visual = { group, mesh, bones: built.bones, packet, item, rootMotion };
  } else {
    const packet = await loadAssetPacket(item);
    const geometry = makeGeometry(packet, false);
    const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(item.material_override?.color || (item.kind === "terrain" ? "#4287c7" : "#e4ff4e")), roughness: item.kind === "terrain" ? 0.92 : 0.66, metalness: 0.01, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.directorType = "item";
    mesh.userData.directorId = item.id;
    group.add(mesh);
    visual = { group, mesh, packet, item };
  }
  applyItemTransform(group, item);
  group.traverse((object) => object.layers.set(CONTENT_LAYER));
  visual.rootMotion?.traverse((object) => object.layers.set(HELPER_LAYER));
  if (visual.rootMotion) visual.rootMotion.visible = Boolean(item.root_motion_enabled);
  runtime.scene.add(group);
  state.visuals.set(item.id, visual);
  return visual;
}

function makeCameraVisual(cameraData) {
  const runtime = initRuntime();
  const camera = new THREE.PerspectiveCamera(cameraData.fov || 42, aspectRatio(), cameraData.near || 0.01, cameraData.far || 100);
  camera.name = cameraData.name;
  camera.userData.directorType = "camera";
  camera.userData.directorId = cameraData.id;
  applyItemTransform(camera, cameraData);
  camera.scale.set(1, 1, 1);
  camera.layers.set(CONTENT_LAYER);
  const helperCamera = new THREE.PerspectiveCamera(camera.fov, camera.aspect, CAMERA_HELPER_NEAR, CAMERA_HELPER_FAR);
  camera.add(helperCamera);
  const helper = new THREE.CameraHelper(helperCamera);
  helper.layers.set(HELPER_LAYER);
  helper.traverse((object) => object.layers.set(HELPER_LAYER));
  helper.material.depthTest = false;
  helper.material.depthWrite = false;
  helper.renderOrder = 18;
  const body = new THREE.Group();
  body.layers.set(HELPER_LAYER);
  body.userData.directorType = "camera";
  body.userData.directorId = cameraData.id;
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.1, 0.22), new THREE.MeshBasicMaterial({ color: CAMERA_COLOR, wireframe: true, depthTest: false }));
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, 0.12, 12), new THREE.MeshBasicMaterial({ color: CAMERA_COLOR, wireframe: true, depthTest: false }));
  lens.rotation.x = Math.PI / 2;
  lens.position.z = -0.16;
  const numberSprite = makeCameraNumberSprite(cameraData);
  body.add(box, lens, numberSprite);
  // Layers are not inherited by children: every camera tool renderable must
  // remain visible to editor/raycast helpers, never to content-only exports.
  body.traverse((object) => object.layers.set(HELPER_LAYER));
  camera.add(body);
  runtime.scene.add(camera, helper);
  const visual = { group: camera, camera, helperCamera, helper, body, box, lens, numberSprite, data: cameraData };
  state.cameraVisuals.set(cameraData.id, visual);
  updateCameraHelper(visual);
  updateCameraSelectionVisuals();
  return visual;
}

function paintCameraNumberSprite(sprite, cameraData, selected = false) {
  if (!sprite) return;
  const canvas = sprite.userData.labelCanvas;
  const context = canvas?.getContext("2d");
  if (!context) return;
  const accent = selected ? "#2f7cff" : "#e4ff4e";
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = selected ? accent : "rgba(8, 10, 13, 0.92)";
  context.fillRect(3, 3, canvas.width - 6, canvas.height - 6);
  context.strokeStyle = accent;
  context.lineWidth = 5;
  context.strokeRect(3, 3, canvas.width - 6, canvas.height - 6);
  context.fillStyle = selected ? "#ffffff" : accent;
  context.font = "800 30px monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(cameraData.number || "CAM", canvas.width / 2, canvas.height / 2 + 1, canvas.width - 18);
  sprite.material.map.needsUpdate = true;
}

function makeCameraNumberSprite(cameraData) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 72;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }));
  sprite.position.set(0, 0.23, 0);
  sprite.scale.set(0.52, 0.146, 1);
  sprite.renderOrder = 20;
  sprite.userData.directorType = "camera";
  sprite.userData.directorId = cameraData.id;
  sprite.userData.labelCanvas = canvas;
  paintCameraNumberSprite(sprite, cameraData, false);
  return sprite;
}

function updateCameraSelectionVisuals() {
  for (const [cameraId, visual] of state.cameraVisuals) {
    const selected = cameraId === state.selectedCameraId;
    const color = selected ? SELECTED_CAMERA_COLOR : CAMERA_COLOR;
    visual.box?.material?.color?.setHex(color);
    visual.lens?.material?.color?.setHex(color);
    const helperColor = new THREE.Color(color);
    visual.helper?.setColors?.(helperColor, helperColor, helperColor, helperColor, helperColor);
    paintCameraNumberSprite(visual.numberSprite, visual.data, selected);
  }
}

function updateCameraHelper(visual) {
  if (!visual?.camera || !visual?.helperCamera || !visual?.helper) return;
  visual.helperCamera.fov = visual.camera.fov;
  visual.helperCamera.aspect = visual.camera.aspect;
  visual.helperCamera.updateProjectionMatrix();
  visual.helperCamera.updateMatrixWorld(true);
  visual.helper.update();
}

function pathCurve(path) {
  const points = (path.points || []).map((point) => new THREE.Vector3().fromArray(point.position));
  if (path.kind === "polyline") {
    return {
      getPointAt(value, target = new THREE.Vector3()) {
        const u = THREE.MathUtils.clamp(value, 0, 1);
        const scaled = u * Math.max(1, points.length - 1);
        const index = Math.min(points.length - 2, Math.floor(scaled));
        return target.copy(points[index]).lerp(points[index + 1], scaled - index);
      },
      getTangentAt(value, target = new THREE.Vector3()) {
        const u = THREE.MathUtils.clamp(value, 0, 1);
        const index = Math.min(points.length - 2, Math.floor(u * Math.max(1, points.length - 1)));
        return target.copy(points[index + 1]).sub(points[index]).normalize();
      },
      getPoints(segments) { return Array.from({ length: segments + 1 }, (_, index) => this.getPointAt(index / segments)); },
    };
  }
  return new THREE.CatmullRomCurve3(points, Boolean(path.closed), "centripetal", 0.5);
}

function rebuildPathVisual(path) {
  const runtime = initRuntime();
  const old = state.pathVisuals.get(path.id);
  if (old) {
    runtime.scene.remove(old.group);
    old.group.traverse((object) => { object.geometry?.dispose?.(); object.material?.dispose?.(); });
  }
  const group = new THREE.Group();
  group.userData.directorType = "path";
  group.userData.directorId = path.id;
  group.layers.set(HELPER_LAYER);
  const curve = pathCurve(path);
  const points = curve.getPoints(Math.max(32, (path.points?.length || 2) * 20));
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: 0x49e6ff, depthTest: false }));
  line.layers.set(HELPER_LAYER);
  group.add(line);
  const markers = (path.points || []).map((point, index) => {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.055, 14, 10), new THREE.MeshBasicMaterial({ color: index === 0 ? 0xe4ff4e : 0x49e6ff, depthTest: false }));
    marker.position.fromArray(point.position);
    marker.userData.directorType = "path-point";
    marker.userData.directorId = path.id;
    marker.userData.pointIndex = index;
    marker.layers.set(HELPER_LAYER);
    group.add(marker);
    return marker;
  });
  runtime.scene.add(group);
  const visual = { group, line, markers, curve, data: path };
  state.pathVisuals.set(path.id, visual);
  return visual;
}

async function rebuildProjectScene(options = {}) {
  clearRuntimeScene();
  if (!state.project) return;
  const requestedFrame = Math.max(0, Math.min(state.project.duration_frames - 1, Number(state.frame) || 0));
  status("正在载入导演场景…", "Loading director scene...");
  for (const item of state.project.scene_items || []) {
    try { await buildSceneItem(item); }
    catch (error) { console.error(error); status(`资产载入失败：${item.name} · ${error.message}`, `Asset failed to load: ${item.name} · ${error.message}`, true); }
  }
  (state.project.cameras || []).forEach(makeCameraVisual);
  if (!cameraItem(state.selectedCameraId)) state.selectedCameraId = state.project.cameras?.[0]?.id || null;
  if (!state.selectedCameraId && state.lockDirectorView) {
    state.lockDirectorView = false;
    if (state.runtime) {
      state.runtime.controls.enabled = !state.runtime.dragging;
      state.runtime.gizmo.getHelper().visible = true;
    }
  }
  updateCameraSelectionVisuals();
  (state.project.camera_paths || []).forEach(rebuildPathVisual);
  setFrame(requestedFrame);
  renderOutliner();
  renderTimeline();
  renderInspector();
  resizeRuntime();
  state.runtime?.scene.updateMatrixWorld(true);
  if (options.frameActors && state.project.scene_items.some((item) => item.kind === "skeletal_actor")) frameActors();
  status("导演场景已就绪。", "Director scene ready.");
}

async function openProject(projectId) {
  try {
    hideProjectMenu();
    state.project = await api(`/api/director-projects/${encodeURIComponent(projectId)}`);
    const migratedSegments = ensureDirectorSegments();
    state.selected = null;
    state.selectedCameraId = state.project.cameras?.[0]?.id || null;
    state.lockDirectorView = false;
    state.frame = 0;
    state.undo = [];
    state.redo = [];
    state.thumbnails.clear();
    $("directorProjectName").value = state.project.name;
    $("directorAspectPreset").value = state.project.aspect_preset;
    $("directorResolutionPreset").value = state.project.resolution_preset;
    $("directorPreviewQuality").value = String(state.project.preview_quality || 0.5);
    $("directorEditorModal").classList.remove("hidden");
    document.body.classList.add("director-editor-open");
    initRuntime();
    state.runtime.controls.enabled = true;
    state.runtime.gizmo.getHelper().visible = true;
    $("directorLockViewBtn").classList.remove("active");
    syncProjectSettingsUi();
    syncDirectorLanguageUi();
    await rebuildProjectScene({ frameActors: true });
    await recoverExportJob();
    if (migratedSegments) scheduleSave();
  } catch (error) {
    status(`打开导演工程失败：${error.message}`, `Could not open director project: ${error.message}`, true);
  }
}

async function closeProject() {
  await flushSave().catch(console.error);
  state.playing = false;
  $("directorPlayBtn").textContent = "▶";
  $("directorEditorModal").classList.add("hidden");
  document.body.classList.remove("director-editor-open");
  state.runtime?.gizmo.detach();
  state.project = null;
  state.selected = null;
  state.selectedCameraId = null;
  state.lockDirectorView = false;
  loadProjects();
}

function projectSnapshot() {
  return state.project ? clone(state.project) : null;
}

function pushUndo(before, label = "Edit") {
  if (!before || !state.project) return;
  state.undo.push({ before, after: projectSnapshot(), label });
  if (state.undo.length > 100) state.undo.shift();
  state.redo.length = 0;
  syncUndoButtons();
}

async function restoreHistory(snapshot) {
  if (!snapshot) return;
  const currentRevision = state.project?.revision;
  state.project = clone(snapshot);
  if (currentRevision != null) state.project.revision = currentRevision;
  $("directorProjectName").value = state.project.name;
  syncProjectSettingsUi();
  await rebuildProjectScene();
  scheduleSave();
}

async function undo() {
  const command = state.undo.pop();
  if (!command) return;
  state.redo.push(command);
  await restoreHistory(command.before);
  syncUndoButtons();
}

async function redo() {
  const command = state.redo.pop();
  if (!command) return;
  state.undo.push(command);
  await restoreHistory(command.after);
  syncUndoButtons();
}

function syncUndoButtons() {
  $("directorUndoBtn").disabled = !state.undo.length;
  $("directorRedoBtn").disabled = !state.redo.length;
}

function scheduleSave() {
  if (!state.project) return;
  state.pendingSave = true;
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => flushSave().catch((error) => status(`自动保存失败：${error.message}`, `Autosave failed: ${error.message}`, true)), 900);
}

async function flushSave() {
  clearTimeout(state.saveTimer);
  if (!state.project || !state.pendingSave || state.saving) return state.project;
  state.saving = true;
  try {
    const row = await api(`/api/director-projects/${encodeURIComponent(state.project.id)}`, jsonOptions("PUT", { project: state.project, expected_revision: state.project.revision }));
    state.project = row;
    state.pendingSave = false;
    $("directorProjectName").value = row.name;
    status("导演工程已自动保存。", "Director project autosaved.");
    return row;
  } finally {
    state.saving = false;
  }
}

function sceneItem(id) { return state.project?.scene_items?.find((item) => item.id === id); }
function cameraItem(id) { return state.project?.cameras?.find((item) => item.id === id); }
function pathItem(id) { return state.project?.camera_paths?.find((item) => item.id === id); }
function cutItem(id) { return state.project?.camera_cuts?.find((item) => item.id === id); }
function markerItem(id) { return state.project?.markers?.find((item) => item.id === id); }
function warpItem(id) { return state.project?.time_warps?.find((item) => item.id === id); }

function selectedData() {
  if (!state.selected) return null;
  if (state.selected.type === "item") return sceneItem(state.selected.id);
  if (state.selected.type === "camera") return cameraItem(state.selected.id);
  if (state.selected.type === "path" || state.selected.type === "path-point") return pathItem(state.selected.id);
  if (state.selected.type === "cut") return cutItem(state.selected.id);
  if (state.selected.type === "marker") return markerItem(state.selected.id);
  if (state.selected.type === "warp") return warpItem(state.selected.id);
  return null;
}

function setSelection(type, id, extra = {}, options = {}) {
  state.selected = id ? { type, id, ...extra } : null;
  if (type === "camera" && cameraItem(id)) state.selectedCameraId = id;
  if (type === "cut" && cutItem(id) && !isBlackoutSegment(cutItem(id))) state.selectedCameraId = cutItem(id).camera_id;
  const runtime = state.runtime;
  runtime?.gizmo.detach();
  let target = null;
  if (type === "item") target = state.visuals.get(id)?.group;
  if (type === "camera") target = state.cameraVisuals.get(id)?.camera;
  if (type === "path-point") target = state.pathVisuals.get(id)?.markers?.[Number(extra.pointIndex || 0)];
  if (target && !selectedData()?.locked) runtime?.gizmo.attach(target);
  updateCameraSelectionVisuals();
  syncCameraViewUi();
  renderOutliner();
  if (options.renderTimeline !== false) renderTimeline();
  renderInspector();
}

function visualForSelection() {
  if (!state.selected) return null;
  if (state.selected.type === "item") return state.visuals.get(state.selected.id)?.group;
  if (state.selected.type === "camera") return state.cameraVisuals.get(state.selected.id)?.camera;
  if (state.selected.type === "path-point") return state.pathVisuals.get(state.selected.id)?.markers?.[Number(state.selected.pointIndex || 0)];
  return null;
}

function syncSelectedFromVisual() {
  const object = visualForSelection();
  const data = selectedData();
  if (!object || !data || !state.selected) return;
  if (state.selected.type === "item" || state.selected.type === "camera") {
    data.transform.translation = object.position.toArray();
    data.transform.quaternion = object.quaternion.toArray();
    if (state.selected.type === "item") data.transform.uniform_scale = object.scale.x;
    updateCameraHelper(state.cameraVisuals.get(data.id));
  } else if (state.selected.type === "path-point") {
    const point = data.points?.[Number(state.selected.pointIndex || 0)];
    if (point) point.position = object.position.toArray();
  }
  renderInspector();
}

function commitGizmoTransform() {
  const runtime = state.runtime;
  if (!runtime?.gizmoBefore || !state.project) return;
  syncSelectedFromVisual();
  const before = runtime.gizmoBefore;
  runtime.gizmoBefore = null;
  if (state.selected?.type === "path-point") rebuildPathVisual(pathItem(state.selected.id));
  pushUndo(before, "Transform");
  scheduleSave();
}

function selectFromViewport(event) {
  const runtime = state.runtime;
  if (!runtime || runtime.dragging || runtime.gizmo.dragging || runtime.gizmo.axis || state.lockDirectorView) return;
  const rect = runtime.canvas.getBoundingClientRect();
  runtime.pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  runtime.raycaster.layers.enableAll();
  runtime.raycaster.params.Line = { threshold: 0.08 };
  runtime.raycaster.setFromCamera(runtime.pointer, runtime.editorCamera);
  const hits = runtime.raycaster.intersectObjects(runtime.scene.children, true);
  for (const hit of hits) {
    let object = hit.object;
    while (object && !object.userData?.directorType) object = object.parent;
    if (!object) continue;
    const type = object.userData.directorType;
    if (type === "item" || type === "camera") return setSelection(type, object.userData.directorId);
    if (type === "path-point") return setSelection(type, object.userData.directorId, { pointIndex: object.userData.pointIndex });
  }
}

function iconForItem(kind) {
  return kind === "skeletal_actor" ? "ACT" : kind === "terrain" ? "TER" : "MESH";
}

function outlinerRow(type, id, name, badge, visible = true, extraClass = "") {
  const selected = state.selected?.type === type && state.selected?.id === id;
  return `<button class="director-outliner-row ${selected ? "selected" : ""} ${extraClass}" data-select-type="${type}" data-select-id="${escapeHtml(id)}"><span class="director-outliner-badge">${badge}</span><span>${escapeHtml(name)}</span><i>${visible ? "●" : "○"}</i></button>`;
}

function renderOutliner() {
  const root = $("directorOutlinerList");
  if (!root || !state.project) return;
  let html = `<h3>${tr("场景物体", "SCENE OBJECTS")}</h3>`;
  html += (state.project.scene_items || []).map((item) => outlinerRow("item", item.id, item.name, iconForItem(item.kind), item.visible)).join("");
  html += `<h3>${tr("摄像机", "CAMERAS")}</h3>`;
  html += (state.project.cameras || []).map((camera) => outlinerRow("camera", camera.id, `${camera.number || "CAM"} · ${camera.name}`, "CAM", true, camera.id === state.selectedCameraId ? "active-camera" : "")).join("");
  html += `<h3>${tr("相机轨道", "CAMERA PATHS")}</h3>`;
  html += (state.project.camera_paths || []).map((path) => outlinerRow("path", path.id, path.name, path.kind === "polyline" ? "LINE" : "SPL")).join("");
  if (!(state.project.scene_items?.length || state.project.cameras?.length || state.project.camera_paths?.length)) html += `<p class="empty">${tr("场景中暂无组件", "The scene is empty")}</p>`;
  root.innerHTML = html;
  root.querySelectorAll("[data-select-type]").forEach((button) => {
    button.onclick = () => setSelection(button.dataset.selectType, button.dataset.selectId);
  });
}

function numberInput(label, field, value, options = "step=\"0.01\"") {
  return `<label>${label}<input type="number" data-field="${field}" value="${Number(value ?? 0)}" ${options}/></label>`;
}

function vectorInputs(label, field, values, step = 0.01) {
  return `<fieldset class="director-vector"><legend>${label}</legend>${["X", "Y", "Z"].map((axis, index) => numberInput(axis, `${field}.${index}`, values?.[index] || 0, `step="${step}"`)).join("")}</fieldset>`;
}

function quaternionEuler(quaternion) {
  const euler = new THREE.Euler().setFromQuaternion(new THREE.Quaternion().fromArray(quaternion || [0, 0, 0, 1]), "YXZ");
  return [euler.x, euler.y, euler.z].map(THREE.MathUtils.radToDeg);
}

function commonTransformInspector(data, allowScale = true) {
  return `${vectorInputs(tr("位置 / 米", "Position / m"), "translation", data.transform?.translation)}${vectorInputs(tr("旋转 / 度", "Rotation / degrees"), "euler", quaternionEuler(data.transform?.quaternion), 0.1)}${allowScale ? numberInput(tr("整体缩放", "Uniform scale"), "uniform_scale", data.transform?.uniform_scale || 1, 'min="0.0001" max="1000" step="0.01"') : ""}`;
}

function targetOptions(selectedId = "") {
  const rows = [
    ...state.project.scene_items.map((item) => ({ id: item.id, name: item.name })),
    ...state.project.cameras.map((camera) => ({ id: camera.id, name: camera.name })),
  ];
  return `<option value="">${tr("未选择", "None")}</option>${rows.map((row) => `<option value="${escapeHtml(row.id)}" ${row.id === selectedId ? "selected" : ""}>${escapeHtml(row.name)}</option>`).join("")}`;
}

function renderInspector() {
  const panel = $("directorInspector");
  const empty = $("directorInspectorEmpty");
  const data = selectedData();
  if (!panel || !empty) return;
  panel.classList.toggle("hidden", !data);
  empty.classList.toggle("hidden", Boolean(data));
  if (!data || !state.selected) return;
  let html = `<div class="director-inspector-head"><strong>${escapeHtml(data.name || state.selected.type)}</strong><button type="button" data-action="delete-selected">${tr("删除", "Delete")}</button></div>`;
  html += `<label>${tr("名称", "Name")}<input data-field="name" value="${escapeHtml(data.name || "")}"/></label>`;
  if (state.selected.type === "item") {
    html += `<label>${tr("颜色", "Color")}<input type="color" data-field="color" value="${escapeHtml(data.material_override?.color || "#e4ff4e")}"/></label>`;
    html += commonTransformInspector(data, true);
    html += `<div class="director-field-grid">${numberInput(tr("出现帧", "Start frame"), "timeline.start_frame", data.timeline?.start_frame || 0, `min="0" max="${state.project.duration_frames - 1}" step="1"`)}${numberInput(tr("结束帧", "End frame"), "timeline.end_frame", data.timeline?.end_frame || state.project.duration_frames, `min="1" max="${state.project.duration_frames}" step="1"`)}${numberInput(tr("播放速率", "Playback rate"), "playback_rate", data.playback_rate || 1, 'min="0.01" max="16" step="0.05"')}</div>`;
    if (data.kind === "skeletal_actor") {
      const hasRootMotion = rootMotionPath(state.visuals.get(data.id)?.packet?.scene || {}).length > 1;
      html += `<label class="check ${hasRootMotion ? "" : "disabled"}"><input type="checkbox" data-field="root_motion_enabled" ${data.root_motion_enabled ? "checked" : ""} ${hasRootMotion ? "" : "disabled"}/>${tr("启用根运动（显示轨迹）", "Enable Root Motion (show trajectory)")}</label>`;
    }
    html += `<label class="check"><input type="checkbox" data-field="visible" ${data.visible ? "checked" : ""}/>${tr("显示", "Visible")}</label>`;
  } else if (state.selected.type === "camera") {
    html += `<div class="director-field-grid"><label>${tr("镜头编号", "Shot number")}<input data-field="number" value="${escapeHtml(data.number || "")}"/></label>${numberInput("FOV", "fov", data.fov || 42, 'min="5" max="170" step="0.1"')}</div>`;
    html += commonTransformInspector(data, false);
    html += `<label>${tr("备注", "Notes")}<textarea data-field="notes">${escapeHtml(data.notes || "")}</textarea></label>`;
    html += `<details open><summary>Look At</summary><label class="check"><input type="checkbox" data-field="look_at.enabled" ${data.look_at?.enabled ? "checked" : ""}/>${tr("启用 Look At", "Enable Look At")}</label><label>${tr("目标类型", "Target type")}<select data-field="look_at.kind"><option value="none" ${data.look_at?.kind === "none" ? "selected" : ""}>${tr("无", "None")}</option><option value="actor" ${data.look_at?.kind === "actor" ? "selected" : ""}>${tr("人物", "Character")}</option><option value="bone" ${data.look_at?.kind === "bone" ? "selected" : ""}>${tr("骨骼", "Bone")}</option><option value="object" ${data.look_at?.kind === "object" ? "selected" : ""}>${tr("场景物体", "Scene Object")}</option><option value="world" ${data.look_at?.kind === "world" ? "selected" : ""}>${tr("空间坐标", "World Position")}</option></select></label><label>${tr("目标", "Target")}<select data-field="look_at.target_id">${targetOptions(data.look_at?.target_id)}</select></label>${data.look_at?.kind === "world" ? vectorInputs(tr("空间坐标", "World Position"), "look_at.world_position", data.look_at.world_position) : ""}<label>${tr("骨骼名称（可选）", "Bone name (optional)")}<input data-field="look_at.bone_name" value="${escapeHtml(data.look_at?.bone_name || "")}"/></label><label class="check"><input type="checkbox" data-field="look_at.keep_level" ${data.look_at?.keep_level ? "checked" : ""}/>${tr("保持水平", "Keep level")}</label></details>`;
    html += `<details><summary>${tr("相机轨道", "Camera path")}</summary><label>${tr("轨道", "Path")}<select data-field="path_id"><option value="">${tr("无", "None")}</option>${state.project.camera_paths.map((path) => `<option value="${path.id}" ${path.id === data.path_id ? "selected" : ""}>${escapeHtml(path.name)}</option>`).join("")}</select></label><label>${tr("朝向模式", "Orientation")}<select data-field="orientation_mode"><option value="free" ${data.orientation_mode === "free" ? "selected" : ""}>Free</option><option value="rail_tangent" ${data.orientation_mode === "rail_tangent" ? "selected" : ""}>Rail Tangent</option><option value="look_at" ${data.orientation_mode === "look_at" ? "selected" : ""}>Look At</option><option value="rail_level" ${data.orientation_mode === "rail_level" ? "selected" : ""}>Rail Level</option></select></label></details>`;
    html += `<details><summary>${tr("手持抖动", "Handheld shake")}</summary><label class="check"><input type="checkbox" data-field="shake.enabled" ${data.shake?.enabled ? "checked" : ""}/>${tr("启用", "Enabled")}</label>${numberInput(tr("位置幅度", "Position amplitude"), "shake.position_amplitude_all", Math.max(...(data.shake?.position_amplitude || [0, 0, 0])), 'min="0" max="2" step="0.001"')}${numberInput(tr("旋转幅度 / 度", "Rotation amplitude / degrees"), "shake.rotation_amplitude_all", Math.max(...(data.shake?.rotation_amplitude || [0, 0, 0]).map(Math.abs).map(THREE.MathUtils.radToDeg)), 'min="0" max="45" step="0.1"')}${numberInput(tr("频率", "Frequency"), "shake.frequency", data.shake?.frequency || 1.5, 'min="0.01" max="60" step="0.1"')}</details>`;
  } else if (state.selected.type === "path" || state.selected.type === "path-point") {
    html += `<label>${tr("类型", "Type")}<select data-field="kind"><option value="catmull_rom" ${data.kind === "catmull_rom" ? "selected" : ""}>Curve</option><option value="polyline" ${data.kind === "polyline" ? "selected" : ""}>Polyline</option></select></label>`;
    html += `<label class="check"><input type="checkbox" data-field="closed" ${data.closed ? "checked" : ""}/>${tr("闭合轨道", "Closed path")}</label>`;
    html += numberInput(tr("轨道时长 / 帧", "Path duration / frames"), "duration_frames", data.duration_frames || 90, 'min="1" step="1"');
    html += `<label>${tr("速度模式", "Speed mode")}<select data-field="speed_mode"><option value="duration" ${data.speed_mode === "duration" ? "selected" : ""}>${tr("按时长", "By duration")}</option><option value="constant" ${data.speed_mode === "constant" ? "selected" : ""}>${tr("恒定速度", "Constant speed")}</option><option value="keys" ${data.speed_mode === "keys" ? "selected" : ""}>${tr("关键帧", "Keyframes")}</option></select></label>${numberInput(tr("速度 / 米每秒", "Speed / m/s"), "speed", data.speed || 1, 'min="0.001" max="1000" step="0.05"')}`;
    html += `<div class="director-path-point-list">${data.points.map((point, index) => `<button type="button" data-path-point="${index}" class="${state.selected.type === "path-point" && state.selected.pointIndex === index ? "active" : ""}">${tr("控制点", "Point")} ${index + 1}</button>`).join("")}</div><button type="button" data-action="add-path-point">${tr("增加控制点", "Add point")}</button>`;
    if (state.selected.type === "path-point") html += vectorInputs(tr("控制点位置", "Point position"), "point_position", data.points[state.selected.pointIndex]?.position || [0, 0, 0]);
  } else if (state.selected.type === "cut") {
    const blackout = isBlackoutSegment(data);
    const durationFrames = segmentLength(data);
    html += `<div class="director-segment-kind ${blackout ? "blackout" : "camera"}">${blackout ? tr("黑幕片段", "BLACKOUT SEGMENT") : "CAMERA CUT"}</div>`;
    html += `<div class="director-field-grid"><label>${tr("片段编号", "Segment number")}<input data-field="number" value="${escapeHtml(data.number || "")}"/></label><label>${tr("片段颜色", "Segment color")}<input type="color" data-field="color" value="${escapeHtml(data.color || (blackout ? BLACKOUT_COLOR : "#e4ff4e"))}" ${blackout ? "disabled" : ""}/></label></div>`;
    if (!blackout) html += `<label>${tr("摄像机", "Camera")}<select data-field="camera_id">${state.project.cameras.map((camera) => `<option value="${camera.id}" ${camera.id === data.camera_id ? "selected" : ""}>${escapeHtml(`${camera.number || "CAM"} · ${camera.name}`)}</option>`).join("")}</select></label>`;
    html += `<div class="director-field-grid">${numberInput(tr("开始帧", "Start frame"), "start_frame", data.start_frame, 'min="0" max="1999999" step="1"')}${numberInput(tr("结束帧", "End frame"), "end_frame", data.end_frame, 'min="1" max="2000000" step="1"')}${numberInput(tr("长度 / 帧", "Length / frames"), "duration_frames", durationFrames, 'min="1" max="2000000" step="1"')}</div>`;
    html += `<div class="director-time-readout"><span>${tr("开始", "Start")}<strong>${frameSeconds(data.start_frame)} s</strong></span><span>${tr("结束", "End")}<strong>${frameSeconds(data.end_frame)} s</strong></span><span>${tr("长度", "Length")}<strong>${frameSeconds(durationFrames)} s</strong></span></div>`;
    html += `<label>${tr("备注", "Notes")}<textarea data-field="notes">${escapeHtml(data.notes || "")}</textarea></label>`;
    const thumb = state.thumbnails.get(data.id);
    if (thumb && !blackout) html += `<img class="director-shot-thumb" src="${thumb}" alt="Camera Cut thumbnail"/>`;
  } else if (state.selected.type === "marker") {
    html += numberInput(tr("帧", "Frame"), "frame", data.frame, `min="0" max="${state.project.duration_frames - 1}" step="1"`);
    html += `<label>${tr("颜色", "Color")}<input type="color" data-field="color" value="${data.color}"/></label><label>${tr("备注", "Notes")}<textarea data-field="notes">${escapeHtml(data.notes || "")}</textarea></label>`;
  } else if (state.selected.type === "warp") {
    html += `<div class="director-field-grid">${numberInput(tr("开始帧", "Start frame"), "start_frame", data.start_frame, `min="0" max="${state.project.duration_frames - 1}" step="1"`)}${numberInput(tr("结束帧", "End frame"), "end_frame", data.end_frame, `min="1" max="${state.project.duration_frames}" step="1"`)}</div><label>${tr("模式", "Mode")}<select data-field="mode"><option value="rate" ${data.mode === "rate" ? "selected" : ""}>${tr("倍率", "Rate")}</option><option value="freeze" ${data.mode === "freeze" ? "selected" : ""}>${tr("定格", "Freeze")}</option><option value="curve" ${data.mode === "curve" ? "selected" : ""}>${tr("曲线", "Curve")}</option></select></label>${numberInput(tr("时间膨胀倍率", "Time dilation"), "rate", data.rate || 1, 'min="0.001" max="16" step="0.05"')}<label class="check"><input type="checkbox" data-field="camera_follows_warp" ${data.camera_follows_warp ? "checked" : ""}/>${tr("相机也应用变速", "Apply warp to camera")}</label>`;
  }
  panel.innerHTML = html;
  bindInspector();
}

function setNestedField(target, path, value) {
  const parts = path.split(".");
  let cursor = target;
  for (let index = 0; index < parts.length - 1; index += 1) cursor = cursor[parts[index]];
  cursor[parts.at(-1)] = value;
}

function editDirectorSegment(segment, field, value) {
  const rows = directorSegments();
  const index = rows.findIndex((row) => row.id === segment.id);
  if (index < 0) return false;
  const previousEnd = index > 0 ? rows[index - 1].end_frame : 0;
  if (field === "start_frame") {
    segment.start_frame = Math.max(previousEnd, Math.min(segment.end_frame - 1, Math.round(Number(value) || 0)));
  } else if (field === "end_frame" || field === "duration_frames") {
    const requestedEnd = field === "duration_frames"
      ? segment.start_frame + Math.max(1, Math.round(Number(value) || 1))
      : Math.max(segment.start_frame + 1, Math.round(Number(value) || segment.start_frame + 1));
    const delta = requestedEnd - segment.end_frame;
    segment.end_frame = requestedEnd;
    shiftDirectorSegments(index + 1, delta);
  } else {
    return false;
  }
  segment.thumbnail_frame = Math.max(segment.start_frame, Math.min(segment.end_frame - 1, Number(segment.thumbnail_frame ?? segment.start_frame)));
  reindexDirectorSegments();
  const lastEnd = rows.at(-1)?.end_frame || 1;
  state.project.duration_frames = Math.max(state.project.duration_frames, lastEnd);
  state.frame = Math.min(state.frame, state.project.duration_frames - 1);
  return true;
}

function applyInspectorField(input) {
  const data = selectedData();
  if (!data || !state.selected) return;
  const field = input.dataset.field;
  if (!field) return;
  const before = projectSnapshot();
  const number = input.type === "number" ? Number(input.value) : null;
  if (field === "translation" || field === "euler") return;
  if (state.selected.type === "cut" && editDirectorSegment(data, field, number)) {
    // Director segment bounds are edited through the shared non-overlap resolver.
  } else if (field.startsWith("translation.")) data.transform.translation[Number(field.split(".")[1])] = number;
  else if (field.startsWith("look_at.world_position.")) data.look_at.world_position[Number(field.split(".").at(-1))] = number;
  else if (field.startsWith("euler.")) {
    const values = [...input.closest("fieldset").querySelectorAll("input")].map((node) => THREE.MathUtils.degToRad(Number(node.value)));
    data.transform.quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(values[0], values[1], values[2], "YXZ")).toArray();
  } else if (field.startsWith("point_position.")) data.points[state.selected.pointIndex].position[Number(field.split(".")[1])] = number;
  else if (field === "color") data.material_override ? data.material_override.color = input.value : data.color = input.value;
  else if (field === "shake.position_amplitude_all") data.shake.position_amplitude = [number, number, number];
  else if (field === "shake.rotation_amplitude_all") data.shake.rotation_amplitude = [number, number, number].map(THREE.MathUtils.degToRad);
  else setNestedField(data, field, input.type === "checkbox" ? input.checked : input.type === "number" ? number : input.value);
  if (state.selected.type === "item") {
    const visual = state.visuals.get(data.id);
    applyItemTransform(visual?.group, data);
    visual?.mesh?.material?.color?.set(data.material_override.color);
    if (data.kind === "skeletal_actor" && field === "root_motion_enabled") applySceneItemFrame(data, state.frame);
  } else if (state.selected.type === "camera") {
    const visual = state.cameraVisuals.get(data.id);
    applyItemTransform(visual?.camera, data);
    if (visual?.camera) { visual.camera.fov = Number(data.fov); visual.camera.updateProjectionMatrix(); updateCameraHelper(visual); }
    if (field === "number" || field === "name") paintCameraNumberSprite(visual?.numberSprite, data, data.id === state.selectedCameraId);
    syncCameraViewUi();
  } else if (state.selected.type === "path" || state.selected.type === "path-point") rebuildPathVisual(data);
  else if (state.selected.type === "cut" && field === "camera_id" && !isBlackoutSegment(data)) state.selectedCameraId = data.camera_id;
  pushUndo(before, `Change ${field}`);
  renderOutliner();
  renderTimeline();
  renderInspector();
  syncProjectSettingsUi();
  syncCameraViewUi();
  scheduleSave();
}

function bindInspector() {
  const panel = $("directorInspector");
  panel.querySelectorAll("[data-field]").forEach((input) => input.addEventListener("change", () => applyInspectorField(input)));
  panel.querySelector("[data-action='delete-selected']")?.addEventListener("click", deleteSelected);
  panel.querySelector("[data-action='add-path-point']")?.addEventListener("click", () => {
    const path = selectedData();
    const before = projectSnapshot();
    const previous = path.points.at(-1)?.position || [0, 1, 0];
    path.points.push({ id: uid("path-point"), position: [previous[0] + 1, previous[1], previous[2]] });
    rebuildPathVisual(path);
    pushUndo(before, "Add path point");
    setSelection("path-point", path.id, { pointIndex: path.points.length - 1 });
    scheduleSave();
  });
  panel.querySelectorAll("[data-path-point]").forEach((button) => button.addEventListener("click", () => setSelection("path-point", state.selected.id, { pointIndex: Number(button.dataset.pathPoint) })));
}

function timelinePercent(frame) {
  return (100 * Number(frame || 0)) / Math.max(1, Number(state.project?.duration_frames || 1));
}

function timelineBlock(row, type, color) {
  const start = Number(row.start_frame ?? row.timeline?.start_frame ?? 0);
  const end = Number(row.end_frame ?? row.timeline?.end_frame ?? state.project.duration_frames);
  const selected = state.selected?.type === type && state.selected.id === row.id;
  const styleType = type === "item" ? "scene" : type;
  const blackout = type === "cut" && isBlackoutSegment(row);
  const thumbnail = type === "cut" && !blackout && state.thumbnails.get(row.id) ? `<img src="${state.thumbnails.get(row.id)}" alt=""/>` : "";
  const camera = type === "cut" && !blackout ? cameraItem(row.camera_id) : null;
  const label = blackout
    ? `${row.number || "BLACK"} · ${tr("黑幕", "Blackout")}`
    : type === "cut"
      ? `${row.number || "CUT"} · ${camera?.number || "CAM"} · ${camera?.name || row.name}`
      : row.number ? `${row.number} · ${row.name}` : row.name;
  return `<button class="director-timeline-block ${styleType} ${blackout ? "blackout" : ""} ${selected ? "selected" : ""}" data-timeline-type="${type}" data-timeline-id="${escapeHtml(row.id)}" data-segment-kind="${blackout ? "blackout" : "camera"}" style="left:calc(${TIMELINE_LABEL_WIDTH}px + (100% - ${TIMELINE_LABEL_WIDTH}px) * ${start / Math.max(1, state.project.duration_frames)});width:calc((100% - ${TIMELINE_LABEL_WIDTH}px) * ${(end - start) / Math.max(1, state.project.duration_frames)});--block-color:${color}" title="${escapeHtml(label)} · ${start + 1}-${end}"><i data-resize="start"></i>${thumbnail}<span>${escapeHtml(label)}</span><i data-resize="end"></i></button>`;
}

function renderTimeline() {
  const root = $("directorTimeline");
  if (!root || !state.project) return;
  if (ensureDirectorSegments()) scheduleSave();
  const playhead = `<i id="directorPlayhead" style="left:calc(124px + (100% - 124px) * ${state.frame / Math.max(1, state.project.duration_frames)})"></i>`;
  const ticks = Array.from({ length: 11 }, (_, index) => `<span class="director-time-tick" style="left:calc(124px + (100% - 124px) * ${index / 10})">${Math.round((state.project.duration_frames - 1) * index / 10) + 1}</span>`).join("");
  const markers = state.project.markers.map((marker) => `<button class="director-timeline-block marker ${state.selected?.type === "marker" && state.selected.id === marker.id ? "selected" : ""}" data-timeline-type="marker" data-timeline-id="${marker.id}" style="left:calc(124px + (100% - 124px) * ${marker.frame / Math.max(1, state.project.duration_frames)});--marker-color:${marker.color}" title="${escapeHtml(marker.name)}"></button>`).join("");
  const cuts = directorSegments().map((cut) => timelineBlock(cut, "cut", cut.color || (isBlackoutSegment(cut) ? BLACKOUT_COLOR : "#e4ff4e"))).join("");
  const cameraRows = state.project.cameras.map((camera) => {
    const frames = [...new Set(Object.values(camera.keys || {}).flat().map((key) => key.frame))].sort((a, b) => a - b);
    const keys = frames.map((frame) => `<button class="director-camera-key" data-timeline-type="camera" data-timeline-id="${camera.id}" data-key-frame="${frame}" style="left:calc(124px + (100% - 124px) * ${frame / Math.max(1, state.project.duration_frames)})" title="${escapeHtml(camera.name)} · F${frame + 1}"></button>`).join("");
    return `<div class="director-timeline-row camera-row"><strong title="${escapeHtml(camera.name)}">${escapeHtml(camera.number || "CAM")} · ${escapeHtml(camera.name)}</strong>${keys}</div>`;
  }).join("");
  const sceneRows = state.project.scene_items.length ? state.project.scene_items.map((item) => `<div class="director-timeline-row"><strong title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</strong>${timelineBlock({ ...item, ...item.timeline }, "item", item.material_override?.color || "#49e6ff")}</div>`).join("") : `<div class="director-timeline-row"><strong>SCENE</strong></div>`;
  const warps = state.project.time_warps.map((warp) => timelineBlock(warp, "warp", "#ca71ff")).join("");
  root.innerHTML = `${playhead}<div class="director-time-ruler">${ticks}${markers}</div><div class="director-timeline-row" data-director-track="true"><strong>CAM CUT</strong>${cuts || `<em>${tr("添加 Camera Cut 以驱动导演视角", "Add Camera Cuts to drive the director view")}</em>`}</div>${cameraRows}${sceneRows}<div class="director-timeline-row"><strong>TIME</strong>${warps}</div>`;
  root.querySelectorAll("[data-timeline-type]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      if (node.dataset.suppressClick === "true") { node.dataset.suppressClick = ""; return; }
      setSelection(node.dataset.timelineType, node.dataset.timelineId);
      if (node.dataset.keyFrame != null) setFrame(Number(node.dataset.keyFrame));
    });
    if (node.classList.contains("director-timeline-block") && node.dataset.timelineType !== "marker") bindTimelineDrag(node);
  });
  root.onpointerdown = timelineSeek;
  syncBlackoutButton();
}

function timelineSeek(event) {
  if (event.target.closest("[data-timeline-type]")) return;
  setFrame(Math.min(state.project.duration_frames - 1, timelineFrameAtClientX(event.clientX)));
}

function timelineTrackMetrics() {
  const rect = $("directorTimeline").getBoundingClientRect();
  return { rect, left: rect.left + TIMELINE_LABEL_WIDTH, width: Math.max(1, rect.width - TIMELINE_LABEL_WIDTH) };
}

function timelineFrameAtClientX(clientX) {
  const metrics = timelineTrackMetrics();
  const ratio = Math.max(0, Math.min(1, (clientX - metrics.left) / metrics.width));
  return Math.round(ratio * state.project.duration_frames);
}

function updateDirectorSegmentNodePositions() {
  const root = $("directorTimeline");
  if (!root) return;
  root.querySelectorAll('[data-timeline-type="cut"]').forEach((node) => {
    const segment = cutItem(node.dataset.timelineId);
    if (!segment) return;
    node.style.left = `calc(${TIMELINE_LABEL_WIDTH}px + (100% - ${TIMELINE_LABEL_WIDTH}px) * ${segment.start_frame / Math.max(1, state.project.duration_frames)})`;
    node.style.width = `calc((100% - ${TIMELINE_LABEL_WIDTH}px) * ${(segment.end_frame - segment.start_frame) / Math.max(1, state.project.duration_frames)})`;
  });
}

function directorDropIndex(clientX, draggedId) {
  const frame = timelineFrameAtClientX(clientX);
  const remaining = directorSegments().filter((segment) => segment.id !== draggedId);
  const index = remaining.findIndex((segment) => frame < (segment.start_frame + segment.end_frame) / 2);
  return index < 0 ? remaining.length : index;
}

function showDirectorDropIndicator(clientX, draggedId) {
  const row = $("directorTimeline")?.querySelector('.director-timeline-row[data-director-track="true"]');
  if (!row) return;
  let indicator = row.querySelector(".director-timeline-drop-indicator");
  if (!indicator) {
    indicator = document.createElement("i");
    indicator.className = "director-timeline-drop-indicator";
    row.append(indicator);
  }
  const remaining = directorSegments().filter((segment) => segment.id !== draggedId);
  const index = directorDropIndex(clientX, draggedId);
  const frame = index < remaining.length ? remaining[index].start_frame : (remaining.at(-1)?.end_frame || 0);
  indicator.style.left = `calc(${TIMELINE_LABEL_WIDTH}px + (100% - ${TIMELINE_LABEL_WIDTH}px) * ${frame / Math.max(1, state.project.duration_frames)})`;
}

function bindTimelineDrag(node) {
  node.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
    event.preventDefault();
    const type = node.dataset.timelineType;
    const id = node.dataset.timelineId;
    const data = type === "item" ? sceneItem(id)?.timeline : selectedTimelineData(type, id);
    if (!data) return;
    setSelection(type, id, {}, { renderTimeline: false });
    node.parentElement?.querySelectorAll(".director-timeline-block.selected").forEach((row) => row.classList.toggle("selected", row === node));
    const before = projectSnapshot();
    const startX = event.clientX;
    const start = data.start_frame;
    const end = data.end_frame;
    const mode = event.target.dataset.resize || "move";
    const metrics = timelineTrackMetrics();
    const segmentIndex = type === "cut" ? directorSegments().findIndex((segment) => segment.id === id) : -1;
    const downstream = type === "cut" ? directorSegments().slice(segmentIndex + 1).map((segment) => ({ segment, start: segment.start_frame, end: segment.end_frame, thumbnail: segment.thumbnail_frame })) : [];
    const initialDuration = state.project.duration_frames;
    let dragged = false;
    let lastClientX = startX;
    const move = (moveEvent) => {
      lastClientX = moveEvent.clientX;
      if (!dragged && Math.abs(lastClientX - startX) < TIMELINE_DRAG_THRESHOLD) return;
      dragged = true;
      node.classList.add("dragging");
      const delta = Math.round((lastClientX - startX) / metrics.width * state.project.duration_frames);
      if (type === "cut" && mode === "move") {
        showDirectorDropIndicator(lastClientX, id);
        return;
      }
      if (mode === "start") {
        const minimum = type === "cut" && segmentIndex > 0 ? directorSegments()[segmentIndex - 1].end_frame : 0;
        data.start_frame = Math.max(minimum, Math.min(end - 1, start + delta));
      } else if (mode === "end") {
        data.end_frame = Math.max(start + 1, end + delta);
        if (type === "cut") {
          downstream.forEach((row) => {
            row.segment.start_frame = row.start + delta;
            row.segment.end_frame = row.end + delta;
            row.segment.thumbnail_frame = row.thumbnail + delta;
          });
          state.project.duration_frames = Math.max(initialDuration, directorSegments().at(-1)?.end_frame || 1);
        } else data.end_frame = Math.min(state.project.duration_frames, data.end_frame);
      }
      else {
        const length = end - start;
        data.start_frame = Math.max(0, Math.min(state.project.duration_frames - length, start + delta));
        data.end_frame = data.start_frame + length;
      }
      if (type === "cut") updateDirectorSegmentNodePositions();
      else {
        node.style.left = `calc(${TIMELINE_LABEL_WIDTH}px + (100% - ${TIMELINE_LABEL_WIDTH}px) * ${data.start_frame / Math.max(1, state.project.duration_frames)})`;
        node.style.width = `calc((100% - ${TIMELINE_LABEL_WIDTH}px) * ${(data.end_frame - data.start_frame) / Math.max(1, state.project.duration_frames)})`;
      }
    };
    const up = () => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      removeEventListener("pointercancel", up);
      node.classList.remove("dragging");
      $("directorTimeline")?.querySelector(".director-timeline-drop-indicator")?.remove();
      if (dragged && type === "cut" && mode === "move") {
        const rows = directorSegments();
        const originalIndex = rows.findIndex((segment) => segment.id === id);
        const [segment] = rows.splice(originalIndex, 1);
        rows.splice(directorDropIndex(lastClientX, id), 0, segment);
        compactDirectorSegments();
      }
      if (dragged) {
        node.dataset.suppressClick = "true";
        pushUndo(before, type === "cut" && mode === "move" ? "Reorder director segment" : "Edit timeline range");
        syncProjectSettingsUi();
        scheduleSave();
      }
      renderTimeline();
      renderInspector();
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up, { once: true });
    addEventListener("pointercancel", up, { once: true });
  });
}

function selectedTimelineData(type, id) {
  if (type === "cut") return cutItem(id);
  if (type === "warp") return warpItem(id);
  return null;
}

function sampleKeys(keys, frame, fallback, quaternion = false) {
  if (!keys?.length) return Array.isArray(fallback) ? [...fallback] : fallback;
  if (frame <= keys[0].frame) return clone(keys[0].value);
  if (frame >= keys.at(-1).frame) return clone(keys.at(-1).value);
  const rightIndex = keys.findIndex((key) => key.frame >= frame);
  const left = keys[rightIndex - 1];
  const right = keys[rightIndex];
  if (left.interpolation === "step") return clone(left.value);
  const alpha = (frame - left.frame) / Math.max(1, right.frame - left.frame);
  if (quaternion) return new THREE.Quaternion().fromArray(left.value).slerp(new THREE.Quaternion().fromArray(right.value), alpha).normalize().toArray();
  if (Array.isArray(left.value)) return left.value.map((value, index) => THREE.MathUtils.lerp(Number(value), Number(right.value[index]), alpha));
  return THREE.MathUtils.lerp(Number(left.value), Number(right.value), alpha);
}

function effectiveSceneFrame(frame) {
  let value = frame;
  for (const warp of state.project.time_warps || []) {
    if (frame < warp.start_frame || frame >= warp.end_frame) continue;
    if (warp.mode === "freeze") value = warp.source_frame;
    else if (warp.mode === "rate") value = warp.start_frame + (frame - warp.start_frame) * Number(warp.rate || 1);
    else if (warp.keys?.length) value = sampleKeys(warp.keys, frame, frame);
  }
  return value;
}

function effectiveCameraFrame(frame) {
  let value = frame;
  for (const warp of state.project.time_warps || []) {
    if (!warp.camera_follows_warp || frame < warp.start_frame || frame >= warp.end_frame) continue;
    if (warp.mode === "freeze") value = warp.source_frame;
    else if (warp.mode === "rate") value = warp.start_frame + (frame - warp.start_frame) * Number(warp.rate || 1);
    else if (warp.keys?.length) value = sampleKeys(warp.keys, frame, frame);
  }
  return value;
}

function sourceFrameForItem(item, frame) {
  const sourceLength = Math.max(1, item.source_range.end_frame - item.source_range.start_frame);
  let value = item.source_range.start_frame + (effectiveSceneFrame(frame) - item.timeline.start_frame) * Number(item.playback_rate || 1);
  if (item.end_behavior === "loop") value = item.source_range.start_frame + ((value - item.source_range.start_frame) % sourceLength + sourceLength) % sourceLength;
  else value = Math.max(item.source_range.start_frame, Math.min(item.source_range.end_frame - 1, value));
  return Math.round(value);
}

function applyActorRootMotion(item, sceneData, world, source) {
  const virtualRoots = sceneData.virtual_root_transforms || [];
  if (virtualRoots.length) {
    const currentIndex = Math.max(0, Math.min(virtualRoots.length - 1, source));
    const baseIndex = Math.max(0, Math.min(virtualRoots.length - 1, Number(item.source_range?.start_frame) || 0));
    const rootRow = virtualRoots[item.root_motion_enabled ? currentIndex : baseIndex];
    if (rootRow) {
      const root = new THREE.Matrix4().set(...rootRow);
      return world.map((matrix) => root.clone().multiply(matrix));
    }
  }
  const path = sceneData.root_motion_path || [];
  if (!item.root_motion_enabled && path.length) {
    const currentIndex = Math.max(0, Math.min(path.length - 1, source));
    const baseIndex = Math.max(0, Math.min(path.length - 1, Number(item.source_range?.start_frame) || 0));
    const current = path[currentIndex] || [0, 0, 0];
    const base = path[baseIndex] || [0, 0, 0];
    const offset = new THREE.Vector3(
      Number(current[0] || 0) - Number(base[0] || 0),
      Number(current[1] || 0) - Number(base[1] || 0),
      Number(current[2] || 0) - Number(base[2] || 0),
    );
    world.forEach((matrix) => matrix.setPosition(new THREE.Vector3().setFromMatrixPosition(matrix).sub(offset)));
  }
  return world;
}

function applyActorFrame(item, visual, frame) {
  const inside = frame >= item.timeline.start_frame && frame < item.timeline.end_frame;
  visual.group.visible = Boolean(item.visible && inside);
  for (const range of item.visibility_ranges || []) if (frame >= range.start_frame && frame < range.end_frame) visual.group.visible = Boolean(range.visible);
  if (!inside || !visual.bones?.length) return;
  const source = sourceFrameForItem(item, frame);
  const rows = visual.packet.frames[Math.max(0, Math.min(visual.packet.frames.length - 1, source))];
  if (!rows) return;
  const world = applyActorRootMotion(item, visual.packet.scene, rows.map((row) => new THREE.Matrix4().set(...row)), source);
  visual.bones.forEach((bone, index) => {
    const parent = visual.packet.scene.parents[index];
    const local = parent < 0 ? world[index] : world[parent].clone().invert().multiply(world[index]);
    local.decompose(bone.position, bone.quaternion, bone.scale);
  });
  visual.mesh.skeleton.update();
  if (visual.rootMotion) visual.rootMotion.visible = Boolean(item.root_motion_enabled);
}

function applySceneItemFrame(item, frame) {
  const visual = state.visuals.get(item.id);
  if (!visual) return;
  const transform = {
    translation: sampleKeys(item.transform_keys?.translation, frame, item.transform.translation),
    quaternion: sampleKeys(item.transform_keys?.rotation, frame, item.transform.quaternion, true),
    uniform_scale: sampleKeys(item.transform_keys?.scale, frame, item.transform.uniform_scale),
  };
  applyItemTransform(visual.group, { transform });
  if (item.kind === "skeletal_actor") applyActorFrame(item, visual, frame);
  else {
    visual.group.visible = Boolean(item.visible && frame >= item.timeline.start_frame && frame < item.timeline.end_frame);
    for (const range of item.visibility_ranges || []) if (frame >= range.start_frame && frame < range.end_frame) visual.group.visible = Boolean(range.visible);
  }
}

function seededNoise(seed, frame, channel) {
  const value = Math.sin((Number(seed || 1) * 17.17 + frame * 12.9898 + channel * 78.233)) * 43758.5453;
  return (value - Math.floor(value)) * 2 - 1;
}

function sampledCurveLength(curve) {
  const points = curve.getPoints(100);
  let length = 0;
  for (let index = 1; index < points.length; index += 1) length += points[index].distanceTo(points[index - 1]);
  return Math.max(0.001, length);
}

function lookAtTarget(cameraData) {
  const look = cameraData.look_at || {};
  if (!look.enabled) return null;
  if (look.kind === "world") return new THREE.Vector3().fromArray(look.world_position || [0, 1, 0]);
  const visual = state.visuals.get(look.target_id) || state.cameraVisuals.get(look.target_id);
  if (!visual) return null;
  if (look.kind === "bone" && visual.bones?.length) {
    const bone = visual.bones.find((item) => item.name === look.bone_name) || visual.bones[0];
    return bone.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3().fromArray(look.offset || [0, 0, 0]));
  }
  return (visual.group || visual.camera).getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3().fromArray(look.offset || [0, 0, 0]));
}

function evaluateCamera(cameraData, frame) {
  const visual = state.cameraVisuals.get(cameraData.id);
  if (!visual) return;
  const camera = visual.camera;
  frame = effectiveCameraFrame(frame);
  camera.position.fromArray(sampleKeys(cameraData.keys?.translation, frame, cameraData.transform.translation));
  camera.quaternion.fromArray(sampleKeys(cameraData.keys?.rotation, frame, cameraData.transform.quaternion, true)).normalize();
  camera.fov = Math.max(5, Math.min(170, Number(sampleKeys(cameraData.keys?.fov, frame, cameraData.fov))));
  const path = pathItem(cameraData.path_id);
  const pathVisual = path && state.pathVisuals.get(path.id);
  if (pathVisual) {
    const fallbackU = path.speed_mode === "constant"
      ? Math.max(0, Math.min(1, frame / Math.max(1, state.project.fps) * Number(path.speed || 1) / sampledCurveLength(pathVisual.curve)))
      : Math.max(0, Math.min(1, frame / Math.max(1, path.duration_frames)));
    const u = Math.max(0, Math.min(1, Number(sampleKeys(cameraData.keys?.path_position, frame, fallbackU))));
    camera.position.copy(pathVisual.curve.getPointAt(u));
    if (cameraData.orientation_mode === "rail_tangent" || cameraData.orientation_mode === "rail_level") {
      const tangent = pathVisual.curve.getTangentAt(u).normalize();
      if (cameraData.orientation_mode === "rail_level") tangent.y = 0;
      camera.lookAt(camera.position.clone().add(tangent.normalize()));
    }
  }
  const target = lookAtTarget(cameraData);
  if (target && (cameraData.orientation_mode === "look_at" || cameraData.look_at?.enabled)) {
    if (cameraData.look_at.keep_level) target.y = camera.position.y;
    const targetQuaternion = new THREE.Quaternion();
    const matrix = new THREE.Matrix4().lookAt(camera.position, target, camera.up);
    targetQuaternion.setFromRotationMatrix(matrix);
    camera.quaternion.slerp(targetQuaternion, Number(cameraData.look_at.strength ?? 1));
  }
  const shake = cameraData.shake || {};
  const inShakeRange = !shake.end_frame || (frame >= shake.start_frame && frame <= shake.end_frame);
  if (shake.enabled && inShakeRange) {
    const phase = frame / Math.max(1, state.project.fps) * Number(shake.frequency || 1.5);
    const framePhase = phase * 20;
    const positionAmplitude = shake.position_amplitude || [0, 0, 0];
    const rotationAmplitude = shake.rotation_amplitude || [0, 0, 0];
    camera.position.add(new THREE.Vector3(...positionAmplitude.map((value, index) => value * seededNoise(shake.seed, framePhase, index))));
    camera.quaternion.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotationAmplitude.map((value, index) => value * seededNoise(shake.seed, framePhase, index + 3)))));
  }
  camera.aspect = aspectRatio();
  camera.updateProjectionMatrix();
  updateCameraHelper(visual);
}

function activeCut(frame = state.frame) {
  const segment = activeDirectorSegment(frame);
  return segment && !isBlackoutSegment(segment) ? segment : null;
}

function activeDirectorSegment(frame = state.frame) {
  return directorSegments().find((segment) => frame >= segment.start_frame && frame < segment.end_frame) || null;
}

function cutCamera(frame = state.frame) {
  const cut = activeCut(frame);
  return state.cameraVisuals.get(cut?.camera_id)?.camera || null;
}

function selectedCamera() {
  return state.cameraVisuals.get(state.selectedCameraId)?.camera || null;
}

function previewCamera(frame = state.frame) {
  return selectedCamera() || cutCamera(frame) || state.cameraVisuals.values().next().value?.camera || null;
}

function sequenceCamera(frame = state.frame) {
  return cutCamera(frame);
}

function cameraDataForVisual(camera) {
  return camera ? [...state.cameraVisuals.values()].find((visual) => visual.camera === camera)?.data || null : null;
}

function syncCameraViewUi() {
  if (!state.project) return;
  const camera = previewCamera();
  const data = cameraDataForVisual(camera);
  const cameraLabel = data ? `${data.number || "CAM"} · ${data.name || tr("机位", "Camera")}` : tr("无机位", "No Camera");
  const previewLabel = $("directorCameraPreviewLabel");
  if (previewLabel) {
    previewLabel.textContent = cameraLabel;
    previewLabel.classList.toggle("hidden", !data);
  }
  const button = $("directorLockViewBtn");
  if (button) {
    button.textContent = data ? `${tr("导演视角", "Director View")} · ${data.number || "CAM"}` : tr("导演视角", "Director View");
    button.title = data ? `${tr("切换至选中机位", "Switch to selected camera")}：${cameraLabel}` : tr("当前没有可用机位", "No camera is available");
    button.disabled = !data;
  }
  const viewportLabel = $("directorViewportLabel");
  if (viewportLabel) viewportLabel.textContent = state.lockDirectorView
    ? `${tr("导演视角", "Director View")} · ${cameraLabel}`
    : tr("编辑视角", "Editor View");
}

function setFrame(frame, options = {}) {
  if (!state.project) return;
  state.frame = Math.max(0, Math.min(state.project.duration_frames - 1, Math.round(Number(frame) || 0)));
  state.project.scene_items.forEach((item) => applySceneItemFrame(item, state.frame));
  state.project.cameras.forEach((camera) => evaluateCamera(camera, state.frame));
  $("directorTimelineRange").value = state.frame;
  const seconds = state.frame / Math.max(1, state.project.fps);
  $("directorFrameLabel").textContent = `${state.frame + 1} / ${state.project.duration_frames} · ${new Date(seconds * 1000).toISOString().slice(14, 23)}`;
  const playhead = $("directorPlayhead");
  if (playhead) playhead.style.left = `calc(124px + (100% - 124px) * ${state.frame / Math.max(1, state.project.duration_frames)})`;
  syncCameraViewUi();
  if (!options.quiet) renderInspector();
}

function tick(time) {
  const runtime = state.runtime;
  if (!runtime || !state.project || $("directorEditorModal")?.classList.contains("hidden")) return;
  if (state.playing) {
    if (!state.lastTick) state.lastTick = time;
    const frames = ((time - state.lastTick) / 1000) * state.project.fps;
    if (frames >= 1) {
      setFrame((state.frame + Math.max(1, Math.floor(frames))) % state.project.duration_frames, { quiet: true });
      state.lastTick = time;
    }
  }
  runtime.controls.update();
  const selectedPreviewCamera = previewCamera();
  const segment = activeDirectorSegment();
  const selectedBlackout = !state.playing && state.selected?.type === "cut" && isBlackoutSegment(cutItem(state.selected.id));
  const directorBlackout = state.lockDirectorView && (selectedBlackout || (state.playing && (!segment || isBlackoutSegment(segment))));
  const directorCamera = state.playing ? sequenceCamera() : selectedPreviewCamera;
  runtime.renderCamera = state.lockDirectorView && directorCamera ? directorCamera : runtime.editorCamera;
  runtime.renderCamera.layers.set(CONTENT_LAYER);
  if (!state.lockDirectorView) runtime.renderCamera.layers.enable(HELPER_LAYER);
  renderMainViewport(runtime, runtime.renderCamera, state.lockDirectorView && Boolean(directorCamera || directorBlackout), directorBlackout);
  if (selectedPreviewCamera) {
    selectedPreviewCamera.layers.set(CONTENT_LAYER);
    runtime.pipRenderer.render(runtime.scene, selectedPreviewCamera);
  }
}

async function addActor() {
  const characterId = $("directorCharacterSelect").value;
  const animationId = $("directorAnimationSelect").value;
  const character = state.characters.find((row) => row.id === characterId && !isStaticAsset(row));
  const animation = state.animations.find((row) => row.id === animationId && row.character_id === characterId);
  if (!character) return status("请选择骨骼网格体。", "Select a skeletal mesh.", true);
  if (!animation) return status("请选择该网格体上的动画。", "Select an animation on this skeletal mesh.", true);
  const before = projectSnapshot();
  const frames = Math.max(1, Number(animation.frames || animation.num_frames || state.project.duration_frames));
  const item = {
    id: uid("item"), name: animation.name, kind: "skeletal_actor", source_asset_id: character.id,
    source_animation_id: animation.id, source_revision: String(animation.revision || ""), transform: defaultTransform(),
    material_override: { color: "#e4ff4e", mode: "replace" }, timeline: { start_frame: 0, end_frame: state.project.duration_frames },
    source_range: { start_frame: 0, end_frame: frames }, root_motion_enabled: Boolean(animation.has_root_motion && $("directorRootMotionEnabled").checked), playback_rate: 1, end_behavior: "loop", visible: true, locked: false,
    transform_keys: { translation: [], rotation: [], scale: [] }, visibility_ranges: [],
  };
  state.project.scene_items.push(item);
  try { await buildSceneItem(item); }
  catch (error) { state.project.scene_items.pop(); return status(`动画载入失败：${error.message}`, `Animation failed to load: ${error.message}`, true); }
  pushUndo(before, "Add actor");
  setSelection("item", item.id);
  setFrame(state.frame);
  state.runtime?.scene.updateMatrixWorld(true);
  scheduleSave();
}

async function addAsset() {
  const kind = $("directorAssetKind").value;
  const assetId = $("directorAssetSelect").value;
  const rows = kind === "terrain" ? state.terrains : state.characters;
  const asset = rows.find((row) => row.id === assetId);
  if (!asset) return status("请选择场景资产。", "Select a scene asset.", true);
  const before = projectSnapshot();
  const item = {
    id: uid("item"), name: asset.name, kind, source_asset_id: asset.id, source_animation_id: "", source_revision: String(asset.revision || ""),
    transform: defaultTransform(), material_override: { color: kind === "terrain" ? "#4287c7" : "#e4ff4e", mode: "replace" },
    timeline: { start_frame: 0, end_frame: state.project.duration_frames }, source_range: { start_frame: 0, end_frame: 1 }, playback_rate: 1,
    end_behavior: "hold", visible: true, locked: false, transform_keys: { translation: [], rotation: [], scale: [] }, visibility_ranges: [],
  };
  state.project.scene_items.push(item);
  try { await buildSceneItem(item); }
  catch (error) { state.project.scene_items.pop(); return status(`资产载入失败：${error.message}`, `Asset failed to load: ${error.message}`, true); }
  pushUndo(before, "Add scene asset");
  setSelection("item", item.id);
  scheduleSave();
}

function addCamera() {
  const before = projectSnapshot();
  const index = state.project.cameras.length + 1;
  const camera = {
    id: uid("camera"), number: `CAM ${String(index).padStart(2, "0")}`, name: `${tr("机位", "Camera")} ${index}`, notes: "",
    transform: defaultTransform([2.8 + index * 0.25, 1.6, 3.8], new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0.55, 0)).toArray()),
    fov: 42, near: 0.01, far: 500, look_at: { enabled: false, kind: "none", target_id: "", bone_name: "", world_position: [0, 1, 0], offset: [0, 0, 0], strength: 1, keep_level: false },
    path_id: "", orientation_mode: "free", keys: { translation: [], rotation: [], fov: [], path_position: [] },
    shake: { enabled: false, position_amplitude: [0.005, 0.005, 0.005], rotation_amplitude: [0.003, 0.003, 0.003], frequency: 1.5, seed: index, start_frame: 0, end_frame: 0, blend_in_frames: 0, blend_out_frames: 0 },
  };
  state.project.cameras.push(camera);
  state.selectedCameraId = camera.id;
  makeCameraVisual(camera);
  pushUndo(before, "Add camera");
  setSelection("camera", camera.id);
  scheduleSave();
}

function addCameraPath() {
  const before = projectSnapshot();
  const index = state.project.camera_paths.length + 1;
  const path = { id: uid("path"), name: `${tr("相机轨道", "Camera Path")} ${index}`, kind: "catmull_rom", closed: false, points: [
    { id: uid("path-point"), position: [-2, 1.5, 3] }, { id: uid("path-point"), position: [0, 1.5, 2.2] }, { id: uid("path-point"), position: [2, 1.5, 3] },
  ], duration_frames: Math.min(90, state.project.duration_frames), speed_mode: "duration", speed: 1 };
  state.project.camera_paths.push(path);
  rebuildPathVisual(path);
  pushUndo(before, "Add camera path");
  setSelection("path", path.id);
  scheduleSave();
}

function deleteSelected() {
  if (!state.selected || !state.project) return;
  const before = projectSnapshot();
  const { type, id } = state.selected;
  if (type === "item") state.project.scene_items = state.project.scene_items.filter((row) => row.id !== id);
  else if (type === "camera") {
    state.project.cameras = state.project.cameras.filter((row) => row.id !== id);
    state.project.camera_cuts = state.project.camera_cuts.filter((row) => row.camera_id !== id);
    compactDirectorSegments();
    if (state.selectedCameraId === id) state.selectedCameraId = state.project.cameras[0]?.id || null;
  } else if (type === "path") {
    state.project.camera_paths = state.project.camera_paths.filter((row) => row.id !== id);
    state.project.cameras.forEach((camera) => { if (camera.path_id === id) camera.path_id = ""; });
  } else if (type === "path-point") {
    const path = pathItem(id);
    if (path?.points?.length <= 2) return status("相机轨道至少需要两个控制点。", "A camera path needs at least two points.", true);
    path.points.splice(state.selected.pointIndex, 1);
  } else if (type === "cut") {
    const rows = directorSegments();
    const index = rows.findIndex((row) => row.id === id);
    if (index < 0) return;
    const removedLength = segmentLength(rows[index]);
    rows.splice(index, 1);
    shiftDirectorSegments(index, -removedLength);
    reindexDirectorSegments();
    const next = rows[index] || rows[index - 1] || null;
    state.selected = next ? { type: "cut", id: next.id } : null;
    if (next && !isBlackoutSegment(next)) state.selectedCameraId = next.camera_id;
    pushUndo(before, "Delete director segment");
    renderTimeline();
    renderInspector();
    syncCameraViewUi();
    syncBlackoutButton();
    scheduleSave();
    return;
  }
  else if (type === "marker") state.project.markers = state.project.markers.filter((row) => row.id !== id);
  else if (type === "warp") state.project.time_warps = state.project.time_warps.filter((row) => row.id !== id);
  state.selected = null;
  pushUndo(before, "Delete");
  rebuildProjectScene();
  scheduleSave();
}

function syncBlackoutButton() {
  const button = $("directorAddBlackoutBtn");
  if (!button) return;
  const enabled = cameraSegments().length > 0;
  button.disabled = !enabled;
  button.title = enabled ? tr("在导演轴中添加纯黑片段", "Add a pure-black segment to the director track") : tr("至少需要一个 Camera Cut", "At least one Camera Cut is required");
}

function insertDirectorSegment(kind = "camera") {
  const blackout = kind === "blackout";
  if (!blackout && !state.project.cameras.length) return status("请先增加机位。", "Add a camera first.", true);
  if (blackout && !cameraSegments().length) return status("至少需要一个 Camera Cut 才能添加黑幕。", "Add at least one Camera Cut before adding a blackout.", true);
  const before = projectSnapshot();
  const selectedCut = state.selected?.type === "cut" ? cutItem(state.selected.id) : null;
  const camera = cameraItem(state.selected?.type === "camera" ? state.selected.id : "") || cameraItem(selectedCut?.camera_id) || cameraItem(state.selectedCameraId) || state.project.cameras[0];
  const rows = directorSegments();
  const insertAt = directorInsertIndex(state.frame);
  const length = Math.max(1, Math.round(Number(state.project.fps || 30) * 3));
  const start = insertAt > 0 ? rows[insertAt - 1].end_frame : 0;
  shiftDirectorSegments(insertAt, length);
  const sequenceNumber = blackout ? rows.filter(isBlackoutSegment).length + 1 : cameraSegments().length + 1;
  const segment = {
    id: uid(blackout ? "blackout" : "shot"), kind: blackout ? "blackout" : "camera", order: insertAt,
    number: blackout ? `BLACK ${String(sequenceNumber).padStart(2, "0")}` : `S${String(sequenceNumber).padStart(3, "0")}`,
    name: blackout ? tr("黑幕", "Blackout") : `${tr("镜头", "Shot")} ${sequenceNumber}`,
    notes: "", camera_id: blackout ? "" : camera.id, color: blackout ? BLACKOUT_COLOR : nextSegmentColor(),
    start_frame: start, end_frame: start + length, blend_frames: 0, thumbnail_frame: start, time_warp_id: "",
  };
  rows.splice(insertAt, 0, segment);
  reindexDirectorSegments();
  state.project.duration_frames = Math.max(state.project.duration_frames, rows.at(-1)?.end_frame || 1);
  pushUndo(before, blackout ? "Add blackout segment" : "Add Camera Cut");
  setSelection("cut", segment.id);
  syncProjectSettingsUi();
  syncBlackoutButton();
  scheduleSave();
}

function addCut() { insertDirectorSegment("camera"); }

function addBlackout() { insertDirectorSegment("blackout"); }

function addMarker() {
  const before = projectSnapshot();
  const marker = { id: uid("marker"), frame: state.frame, name: `${tr("标记", "Marker")} ${state.project.markers.length + 1}`, notes: "", color: "#e4ff4e", kind: "general" };
  state.project.markers.push(marker);
  pushUndo(before, "Add marker");
  setSelection("marker", marker.id);
  scheduleSave();
}

function addWarp() {
  const before = projectSnapshot();
  const warp = { id: uid("warp"), name: `${tr("局部变速", "Local Time Warp")} ${state.project.time_warps.length + 1}`, start_frame: state.frame, end_frame: Math.min(state.project.duration_frames, state.frame + Math.round(state.project.fps * 2)), mode: "rate", rate: 0.5, source_frame: state.frame, keys: [], camera_follows_warp: false };
  if (warp.end_frame <= warp.start_frame) warp.start_frame = Math.max(0, warp.end_frame - 1);
  state.project.time_warps.push(warp);
  pushUndo(before, "Add time warp");
  setSelection("warp", warp.id);
  scheduleSave();
}

function upsertKey(keys, frame, value) {
  const row = { frame, value: clone(value), interpolation: "linear" };
  const index = keys.findIndex((key) => key.frame === frame);
  if (index >= 0) keys[index] = row;
  else keys.push(row);
  keys.sort((a, b) => a.frame - b.frame);
}

function addCameraKey() {
  let camera = cameraItem(state.selected?.type === "camera" ? state.selected.id : "");
  if (!camera) camera = cameraItem(activeCut()?.camera_id);
  if (!camera) return status("请选择摄像机。", "Select a camera.", true);
  const visual = state.cameraVisuals.get(camera.id)?.camera;
  const before = projectSnapshot();
  upsertKey(camera.keys.translation, state.frame, visual.position.toArray());
  upsertKey(camera.keys.rotation, state.frame, visual.quaternion.toArray());
  upsertKey(camera.keys.fov, state.frame, visual.fov);
  if (camera.path_id) {
    const existing = sampleKeys(camera.keys.path_position, state.frame, state.frame / Math.max(1, pathItem(camera.path_id)?.duration_frames || state.project.duration_frames));
    upsertKey(camera.keys.path_position, state.frame, existing);
  }
  pushUndo(before, "Add camera key");
  setSelection("camera", camera.id);
  status(`已在第 ${state.frame + 1} 帧记录相机关键帧。`, `Camera key recorded at frame ${state.frame + 1}.`);
  scheduleSave();
}

function deleteCurrentKey() {
  const camera = cameraItem(state.selected?.type === "camera" ? state.selected.id : "") || cameraItem(activeCut()?.camera_id);
  if (!camera) return;
  const before = projectSnapshot();
  Object.values(camera.keys).forEach((keys) => {
    const index = keys.findIndex((key) => key.frame === state.frame);
    if (index >= 0) keys.splice(index, 1);
  });
  pushUndo(before, "Delete camera key");
  renderTimeline();
  scheduleSave();
}

function boxForVisuals(visuals) {
  const box = new THREE.Box3();
  visuals.forEach((visual) => box.expandByObject(visual.group || visual.camera || visual));
  return box;
}

function frameBox(box, camera = (state.lockDirectorView ? previewCamera() : state.runtime?.editorCamera)) {
  if (!camera || box.isEmpty()) return;
  const directorCameraData = [...state.cameraVisuals.values()].find((visual) => visual.camera === camera)?.data;
  const before = directorCameraData ? projectSnapshot() : null;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(0.25, size.length() * 0.55);
  const distance = radius / Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
  const direction = camera.position.clone().sub(directorCameraData ? center : state.runtime.controls.target).normalize();
  if (direction.lengthSq() < 0.001) direction.set(1, 0.5, 1).normalize();
  camera.position.copy(center).add(direction.multiplyScalar(distance * 1.2));
  camera.near = Math.max(0.001, distance / 1000);
  camera.far = Math.max(100, distance * 20);
  camera.updateProjectionMatrix();
  if (directorCameraData) {
    camera.lookAt(center);
    directorCameraData.transform.translation = camera.position.toArray();
    directorCameraData.transform.quaternion = camera.quaternion.toArray();
    if (directorCameraData.keys.translation.length || directorCameraData.keys.rotation.length) {
      upsertKey(directorCameraData.keys.translation, state.frame, directorCameraData.transform.translation);
      upsertKey(directorCameraData.keys.rotation, state.frame, directorCameraData.transform.quaternion);
    }
    updateCameraHelper(state.cameraVisuals.get(directorCameraData.id));
    pushUndo(before, "Auto frame camera");
    scheduleSave();
  } else {
    state.runtime.controls.target.copy(center);
    state.runtime.controls.update();
  }
}

function frameSelected() {
  const visual = visualForSelection();
  if (visual) frameBox(boxForVisuals([{ group: visual }]));
}

function frameActors() {
  frameBox(boxForVisuals(state.project.scene_items.filter((item) => item.kind === "skeletal_actor").map((item) => state.visuals.get(item.id)).filter(Boolean)));
}

function syncProjectSettingsUi() {
  if (!state.project) return;
  $("directorTimelineRange").max = state.project.duration_frames - 1;
  $("directorFps").value = state.project.fps;
  $("directorDuration").value = state.project.duration_frames;
  $("directorAspectPreset").value = state.project.aspect_preset;
  $("directorResolutionPreset").value = state.project.resolution_preset;
  $("directorPreviewQuality").value = String(state.project.preview_quality);
  $("directorGuideSafe").checked = Boolean(state.project.guides.safe_frame);
  $("directorGuideCenter").checked = Boolean(state.project.guides.center);
  $("directorGuideThirds").checked = Boolean(state.project.guides.thirds);
  $("directorExportDepth").checked = Boolean(state.project.export_settings.include_depth);
  $("directorExportNormal").checked = Boolean(state.project.export_settings.include_normal);
  $("directorExportSegmentation").checked = Boolean(state.project.export_settings.include_segmentation);
  const overlay = $("directorGuideOverlay");
  overlay.classList.toggle("show-safe", Boolean(state.project.guides.safe_frame));
  overlay.classList.toggle("show-center", Boolean(state.project.guides.center));
  overlay.classList.toggle("show-thirds", Boolean(state.project.guides.thirds));
  const ratio = aspectRatio();
  overlay.style.setProperty("--director-aspect", ratio);
  $("directorCameraPreview").style.aspectRatio = String(ratio);
  resizeRuntime();
  setFrame(state.frame, { quiet: true });
}

async function createSnapshot() {
  if (!state.project) return;
  await flushSave();
  const name = `${tr("版本", "Version")} ${new Date().toLocaleString()}`;
  await api(`/api/director-projects/${state.project.id}/snapshots`, jsonOptions("POST", { name, kind: "manual" }));
  status("版本快照已保存。", "Version snapshot saved.");
}

function canvasBlob(canvas, type = "image/png", quality = 0.95) {
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Canvas export failed")), type, quality));
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ensureExportRuntime(width, height) {
  if (!state.exportRuntime) {
    const canvas = document.createElement("canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x080b11, 1);
    state.exportRuntime = { canvas, renderer };
  }
  state.exportRuntime.renderer.setPixelRatio(1);
  state.exportRuntime.renderer.setSize(width, height, false);
  return state.exportRuntime;
}

function prepareRenderCamera(camera, width, height) {
  camera.aspect = width / Math.max(1, height);
  camera.layers.set(CONTENT_LAYER);
  camera.updateProjectionMatrix();
  return camera;
}

function segmentationMaterial(item) {
  const actor = item?.kind === "skeletal_actor";
  return new THREE.MeshBasicMaterial({ color: actor ? 0xff3b30 : item?.kind === "terrain" ? 0x2f7cff : 0x34c759, side: THREE.DoubleSide, toneMapped: false });
}

async function renderOutputBlob(frame, passName = "rgb", dimensions = outputResolution(), cameraOverride = null) {
  const [width, height] = dimensions;
  const runtime = ensureExportRuntime(width, height);
  setFrame(frame, { quiet: true });
  const segment = activeDirectorSegment(frame);
  const blackout = cameraOverride === false || (!cameraOverride && (!segment || isBlackoutSegment(segment)));
  const camera = cameraOverride || sequenceCamera(frame) || state.runtime.editorCamera;
  prepareRenderCamera(camera, width, height);
  const previousOverride = state.runtime.scene.overrideMaterial;
  const materialRows = [];
  let temporaryOverride = null;
  try {
    if (blackout) {
      runtime.renderer.setClearColor(0x000000, 1);
      runtime.renderer.clear(true, true, true);
      return await canvasBlob(runtime.canvas);
    }
    if (passName === "depth") temporaryOverride = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, side: THREE.DoubleSide });
    if (passName === "normal") temporaryOverride = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
    if (temporaryOverride) state.runtime.scene.overrideMaterial = temporaryOverride;
    if (passName === "segmentation") {
      for (const [id, visual] of state.visuals) {
        const item = sceneItem(id);
        visual.group.traverse((object) => {
          if (!object.isMesh) return;
          materialRows.push([object, object.material]);
          object.material = segmentationMaterial(item);
        });
      }
      runtime.renderer.setClearColor(0x000000, 1);
    }
    runtime.renderer.render(state.runtime.scene, camera);
    return await canvasBlob(runtime.canvas);
  } finally {
    temporaryOverride?.dispose();
    state.runtime.scene.overrideMaterial = previousOverride;
    materialRows.forEach(([object, material]) => { object.material.dispose(); object.material = material; });
    runtime.renderer.setClearColor(0x080b11, 1);
  }
}

async function screenshotExport() {
  if (!state.project) return;
  try {
    status("正在生成无工具组件截图…", "Rendering clean screenshot...");
    const selectedBlackout = state.lockDirectorView && state.selected?.type === "cut" && isBlackoutSegment(cutItem(state.selected.id));
    const currentCamera = selectedBlackout ? false : state.lockDirectorView ? previewCamera(state.frame) : state.runtime.editorCamera;
    const blob = await renderOutputBlob(state.frame, "rgb", outputResolution(), currentCamera);
    downloadBlob(blob, `${state.project.name}_F${String(state.frame + 1).padStart(5, "0")}.png`);
    status("截图已导出。", "Screenshot exported.");
  } catch (error) { status(`截图导出失败：${error.message}`, `Screenshot export failed: ${error.message}`, true); }
}

async function uploadBlob(url, blob, filename, signal) {
  const form = new FormData();
  form.append("file", blob, filename);
  return api(url, { method: "PUT", body: form, signal });
}

function exportPasses() {
  const passes = ["rgb"];
  if ($("directorExportDepth").checked) passes.push("depth");
  if ($("directorExportNormal").checked) passes.push("normal");
  if ($("directorExportSegmentation").checked) passes.push("segmentation");
  return passes;
}

function updateExportProgress(job, message = "") {
  state.exportJob = job;
  const panel = $("directorExportProgress");
  panel.classList.remove("hidden");
  const progress = Math.max(0, Math.min(1, Number(job?.progress || 0)));
  panel.style.setProperty("--progress", `${progress * 100}%`);
  panel.querySelector("span").textContent = message || tr(`导出：${job.stage || job.state}`, `Export: ${job.stage || job.state}`);
  panel.querySelector("small").textContent = `${Math.round(progress * 100)}% · ${job.completed_frames || 0} / ${job.total_frames || state.project?.duration_frames || 0}`;
  $("directorResumeExportBtn").classList.toggle("hidden", !["cancelled", "failed"].includes(job.state));
  $("directorCancelExportBtn").disabled = ["completed", "failed", "cancelled"].includes(job.state);
}

async function newExportJob(kind = "video") {
  await flushSave();
  const settings = {
    ...state.project.export_settings,
    kind,
    passes: exportPasses(),
    format: "mp4",
    width: outputResolution()[0], height: outputResolution()[1], fps: state.project.fps,
  };
  const job = await api(`/api/director-projects/${state.project.id}/exports`, jsonOptions("POST", { settings }));
  updateExportProgress(job);
  return job;
}

async function missingFrames(job, passName) {
  return api(`/api/director-projects/${state.project.id}/exports/${job.id}/missing-frames?pass_name=${encodeURIComponent(passName)}`);
}

async function renderJobFrames(job, passes = exportPasses()) {
  const controller = new AbortController();
  state.exportController = controller;
  const originalFrame = state.frame;
  const totalOperations = passes.length * state.project.duration_frames;
  let completedOperations = 0;
  try {
    for (const passName of passes) {
      const checkpoint = await missingFrames(job, passName);
      completedOperations += checkpoint.completed_frames;
      for (const frame of checkpoint.missing_frames) {
        if (controller.signal.aborted) throw new DOMException("Export cancelled", "AbortError");
        const blob = await renderOutputBlob(frame, passName);
        job = await uploadBlob(`/api/director-projects/${state.project.id}/exports/${job.id}/frames/${passName}/${frame}`, blob, `${String(frame).padStart(8, "0")}.png`, controller.signal);
        completedOperations += 1;
        const localProgress = Math.min(0.84, completedOperations / Math.max(1, totalOperations) * 0.84);
        updateExportProgress({ ...job, progress: localProgress }, tr(`渲染 ${passName.toUpperCase()} · 第 ${frame + 1} 帧`, `Rendering ${passName.toUpperCase()} · frame ${frame + 1}`));
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }
    return job;
  } finally {
    state.exportController = null;
    setFrame(originalFrame);
  }
}

async function pollExport(jobId) {
  while (state.project) {
    const job = await api(`/api/director-projects/${state.project.id}/exports/${jobId}`);
    updateExportProgress(job);
    if (["completed", "failed", "cancelled"].includes(job.state)) return job;
    await new Promise((resolve) => setTimeout(resolve, 650));
  }
  throw new Error("Director project closed");
}

async function runVideoExport(existingJob = null, options = {}) {
  if (!state.project) return null;
  let job = existingJob;
  try {
    job ||= await newExportJob(options.kind || "video");
    job = await renderJobFrames(job, options.passes || exportPasses());
    updateExportProgress({ ...job, state: "encoding", stage: "encoding", progress: 0.86 });
    await api(`/api/director-projects/${state.project.id}/exports/${job.id}/encode`, { method: "POST" });
    job = await pollExport(job.id);
    if (job.state === "failed") throw new Error(job.error || "Video encoding failed");
    if (job.state === "completed" && !options.silentDownload) location.href = `/api/director-projects/${state.project.id}/exports/${job.id}/download`;
    return job;
  } catch (error) {
    if (error.name === "AbortError") {
      status("导出已取消，可从检查点继续。", "Export cancelled; it can resume from its checkpoint.");
      if (job) {
        try { updateExportProgress(await api(`/api/director-projects/${state.project.id}/exports/${job.id}`)); } catch {}
      }
    } else {
      status(`导出失败：${error.message}`, `Export failed: ${error.message}`, true);
    }
    if (job && error.name !== "AbortError") {
      try { updateExportProgress(await api(`/api/director-projects/${state.project.id}/exports/${job.id}`, jsonOptions("PATCH", { state: "failed", stage: "failed", error: error.message }))); } catch {}
    }
    return null;
  }
}

async function cancelExport() {
  const job = state.exportJob;
  if (!job || !state.project) return;
  state.exportController?.abort();
  updateExportProgress(await api(`/api/director-projects/${state.project.id}/exports/${job.id}/cancel`, { method: "POST" }));
  status("导出已取消，已完成的帧会保留。", "Export cancelled; completed frames are retained.");
}

async function resumeExport() {
  if (!state.exportJob || !state.project) return;
  let job = await api(`/api/director-projects/${state.project.id}/exports/${state.exportJob.id}/resume`, { method: "POST" });
  updateExportProgress(job);
  await runVideoExport(job);
}

async function recoverExportJob() {
  if (!state.project) return;
  try {
    const jobs = await api(`/api/director-projects/${state.project.id}/exports`);
    const job = jobs.find((row) => ["rendering", "encoding", "packaging", "cancelled", "failed"].includes(row.state));
    if (!job) return;
    if (["encoding", "packaging"].includes(job.state)) updateExportProgress(await pollExport(job.id));
    else updateExportProgress(job, tr("发现可恢复的导出任务", "A recoverable export was found"));
  } catch (error) { console.warn(error); }
}

async function generateShotThumbnail(cut, force = false) {
  if (!force && state.thumbnails.has(cut.id)) return state.thumbnails.get(cut.id);
  const original = state.frame;
  const width = 320;
  const height = Math.round(width / aspectRatio());
  const blob = await renderOutputBlob(cut.thumbnail_frame ?? cut.start_frame, "rgb", [width, height]);
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
  state.thumbnails.set(cut.id, dataUrl);
  setFrame(original, { quiet: true });
  return dataUrl;
}

async function refreshThumbnails() {
  status("正在刷新 Camera Cut 缩略图…", "Refreshing Camera Cut thumbnails...");
  for (const cut of state.project.camera_cuts) await generateShotThumbnail(cut, true);
  renderTimeline();
  renderInspector();
  status("镜头缩略图已刷新。", "Shot thumbnails refreshed.");
}

async function makeContactSheet() {
  const cuts = state.project.camera_cuts;
  if (!cuts.length) throw new Error(tr("请先添加 Camera Cut。", "Add Camera Cuts first."));
  const columns = Math.min(4, cuts.length);
  const cellWidth = 480;
  const imageHeight = Math.round(cellWidth / aspectRatio());
  const cellHeight = imageHeight + 86;
  const canvas = document.createElement("canvas");
  canvas.width = columns * cellWidth;
  canvas.height = Math.ceil(cuts.length / columns) * cellHeight;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0b0d10";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let index = 0; index < cuts.length; index += 1) {
    const cut = cuts[index];
    const url = await generateShotThumbnail(cut);
    const bitmap = await createImageBitmap(await (await fetch(url)).blob());
    const x = index % columns * cellWidth;
    const y = Math.floor(index / columns) * cellHeight;
    context.drawImage(bitmap, x, y, cellWidth, imageHeight);
    context.fillStyle = "#e4ff4e";
    context.font = "700 22px sans-serif";
    context.fillText(`${cut.number || `S${index + 1}`} · ${cut.name}`, x + 14, y + imageHeight + 30);
    context.fillStyle = "#ffffff";
    context.font = "16px sans-serif";
    context.fillText(`F${cut.start_frame + 1}—F${cut.end_frame} · ${isBlackoutSegment(cut) ? tr("黑幕", "Blackout") : cameraItem(cut.camera_id)?.name || "Camera"}`, x + 14, y + imageHeight + 58);
  }
  return canvasBlob(canvas);
}

async function exportContactSheet(download = true, job = null) {
  try {
    status("正在生成分镜联系表…", "Building storyboard contact sheet...");
    const blob = await makeContactSheet();
    if (job) await uploadBlob(`/api/director-projects/${state.project.id}/exports/${job.id}/artifacts/contact-sheet.png`, blob, "contact-sheet.png");
    if (download) downloadBlob(blob, `${state.project.name}_ContactSheet.png`);
    status("分镜联系表已导出。", "Contact sheet exported.");
    return blob;
  } catch (error) { status(error.message, error.message, true); return null; }
}

function referenceMetadata() {
  return {
    schema_version: 2, project: state.project.name, fps: state.project.fps, duration_frames: state.project.duration_frames,
    aspect: state.project.aspect_preset, resolution: outputResolution(), generated_at: new Date().toISOString(),
    shots: state.project.camera_cuts.map((cut) => {
      const camera = cameraItem(cut.camera_id);
      return { id: cut.id, kind: isBlackoutSegment(cut) ? "blackout" : "camera", order: cut.order, color: cut.color, number: cut.number, name: cut.name, notes: cut.notes, start_frame: cut.start_frame, end_frame: cut.end_frame, duration_frames: segmentLength(cut), thumbnail_frame: cut.thumbnail_frame, camera: camera ? { id: camera.id, number: camera.number, name: camera.name, fov: camera.fov, transform: camera.transform, keys: camera.keys, look_at: camera.look_at, path_id: camera.path_id, shake: camera.shake } : null };
    }),
    markers: state.project.markers,
  };
}

async function exportReferencePack() {
  if (!state.project.camera_cuts.length) return status("请先添加 Camera Cut。", "Add Camera Cuts first.", true);
  let job = null;
  try {
    job = await newExportJob("reference_pack");
    job = await renderJobFrames(job, exportPasses());
    for (const cut of state.project.camera_cuts) {
      const shotName = encodeURIComponent(cut.number || cut.id);
      const first = await renderOutputBlob(cut.start_frame, "rgb");
      const last = await renderOutputBlob(Math.max(cut.start_frame, cut.end_frame - 1), "rgb");
      const thumbnail = await renderOutputBlob(cut.thumbnail_frame ?? cut.start_frame, "rgb", [640, Math.round(640 / aspectRatio())]);
      await uploadBlob(`/api/director-projects/${state.project.id}/exports/${job.id}/artifacts/shots/${shotName}/first.png`, first, "first.png");
      await uploadBlob(`/api/director-projects/${state.project.id}/exports/${job.id}/artifacts/shots/${shotName}/last.png`, last, "last.png");
      await uploadBlob(`/api/director-projects/${state.project.id}/exports/${job.id}/artifacts/shots/${shotName}/thumbnail.png`, thumbnail, "thumbnail.png");
    }
    const metadata = new Blob([JSON.stringify(referenceMetadata(), null, 2)], { type: "application/json" });
    await uploadBlob(`/api/director-projects/${state.project.id}/exports/${job.id}/artifacts/camera-timeline.json`, metadata, "camera-timeline.json");
    await exportContactSheet(false, job);
    await api(`/api/director-projects/${state.project.id}/exports/${job.id}/encode`, { method: "POST" });
    job = await pollExport(job.id);
    if (job.state !== "completed") throw new Error(job.error || "Reference package export failed");
    await api(`/api/director-projects/${state.project.id}/exports/${job.id}/package`, { method: "POST" });
    location.href = `/api/director-projects/${state.project.id}/exports/${job.id}/download`;
    status("参考包已导出。", "Reference package exported.");
  } catch (error) {
    status(`参考包导出失败：${error.message}`, `Reference package export failed: ${error.message}`, true);
    if (job) updateExportProgress({ ...job, state: "failed", error: error.message });
  }
}

function setGizmoMode(mode) {
  const resolved = ["translate", "rotate", "scale"].includes(mode) ? mode : "translate";
  state.runtime?.gizmo.setMode(resolved);
  $("directorMoveBtn").classList.toggle("active", resolved === "translate");
  $("directorRotateBtn").classList.toggle("active", resolved === "rotate");
  $("directorScaleBtn").classList.toggle("active", resolved === "scale");
}

function toggleDirectorView() {
  if (!previewCamera()) return;
  state.lockDirectorView = !state.lockDirectorView;
  $("directorLockViewBtn").classList.toggle("active", state.lockDirectorView);
  state.runtime.controls.enabled = !state.lockDirectorView && !state.runtime.dragging;
  state.runtime.gizmo.getHelper().visible = !state.lockDirectorView;
  syncCameraViewUi();
}

function bindProjectSettings() {
  for (const id of ["directorAspectPreset", "directorResolutionPreset", "directorPreviewQuality"]) {
    $(id).addEventListener("change", () => {
      const before = projectSnapshot();
      state.project.aspect_preset = $("directorAspectPreset").value;
      state.project.resolution_preset = $("directorResolutionPreset").value;
      state.project.preview_quality = Number($("directorPreviewQuality").value);
      pushUndo(before, "Project output settings");
      syncProjectSettingsUi();
      scheduleSave();
    });
  }
  $("directorFps").addEventListener("input", () => {
    if (!$("directorFps").value) return;
    const before = projectSnapshot();
    state.project.fps = Math.max(1, Math.min(120, Number($("directorFps").value) || 30));
    pushUndo(before, "Change frame rate");
    syncProjectSettingsUi();
    scheduleSave();
  });
  $("directorDuration").addEventListener("input", () => {
    if (!$("directorDuration").value) return;
    const before = projectSnapshot();
    const requestedDuration = Math.max(1, Math.min(2_000_000, Math.round(Number($("directorDuration").value) || 300)));
    const segmentMinimum = directorSegments().at(-1)?.end_frame || 1;
    const duration = Math.max(requestedDuration, segmentMinimum);
    state.project.duration_frames = duration;
    state.project.scene_items.forEach((item) => {
      item.timeline.start_frame = Math.min(item.timeline.start_frame, Math.max(0, duration - 1));
      item.timeline.end_frame = Math.max(item.timeline.start_frame + 1, Math.min(duration, item.timeline.end_frame));
    });
    directorSegments().forEach((cut) => { cut.thumbnail_frame = Math.max(cut.start_frame, Math.min(cut.end_frame - 1, cut.thumbnail_frame)); });
    state.project.time_warps.forEach((warp) => {
      warp.start_frame = Math.min(warp.start_frame, Math.max(0, duration - 1));
      warp.end_frame = Math.max(warp.start_frame + 1, Math.min(duration, warp.end_frame));
    });
    state.project.markers.forEach((marker) => { marker.frame = Math.min(marker.frame, duration - 1); });
    state.frame = Math.min(state.frame, duration - 1);
    pushUndo(before, "Change duration");
    syncProjectSettingsUi();
    renderTimeline();
    scheduleSave();
  });
  const guideMap = { directorGuideSafe: "safe_frame", directorGuideCenter: "center", directorGuideThirds: "thirds" };
  Object.entries(guideMap).forEach(([id, key]) => $(id).addEventListener("change", () => {
    state.project.guides[key] = $(id).checked;
    syncProjectSettingsUi();
    scheduleSave();
  }));
  const exportMap = { directorExportDepth: "include_depth", directorExportNormal: "include_normal", directorExportSegmentation: "include_segmentation" };
  Object.entries(exportMap).forEach(([id, key]) => $(id).addEventListener("change", () => {
    state.project.export_settings[key] = $(id).checked;
    scheduleSave();
  }));
}

function bindDirectorEvents() {
  $("directorCreateBtn").onclick = createProject;
  $("directorRefreshBtn").onclick = () => Promise.all([loadProjects(), loadCatalogs()]);
  $("directorCloseBtn").onclick = closeProject;
  $("directorSaveBtn").onclick = async () => { state.pendingSave = true; await flushSave(); status("导演工程已保存。", "Director project saved."); };
  $("directorSnapshotBtn").onclick = createSnapshot;
  $("directorUndoBtn").onclick = undo;
  $("directorRedoBtn").onclick = redo;
  $("directorLanguageZh").onclick = () => window.MotionBuilderI18n?.setLanguage?.("zh");
  $("directorLanguageEn").onclick = () => window.MotionBuilderI18n?.setLanguage?.("en");
  $("directorCharacterSelect").onchange = refreshDirectorAnimationOptions;
  $("directorAnimationSelect").onchange = refreshDirectorRootMotionOption;
  $("directorAssetKind").onchange = refreshAssetSelect;
  $("directorAddActorBtn").onclick = addActor;
  $("directorAddAssetBtn").onclick = addAsset;
  $("directorAddCameraBtn").onclick = addCamera;
  $("directorAddPathBtn").onclick = addCameraPath;
  $("directorMoveBtn").onclick = () => setGizmoMode("translate");
  $("directorRotateBtn").onclick = () => setGizmoMode("rotate");
  $("directorScaleBtn").onclick = () => setGizmoMode("scale");
  $("directorLockViewBtn").onclick = toggleDirectorView;
  $("directorFrameSelectedBtn").onclick = frameSelected;
  $("directorFrameActorsBtn").onclick = frameActors;
  $("directorPlayBtn").onclick = () => {
    state.playing = !state.playing;
    state.lastTick = 0;
    $("directorPlayBtn").textContent = state.playing ? "⏸" : "▶";
  };
  $("directorTimelineRange").oninput = (event) => setFrame(event.target.value, { quiet: true });
  $("directorAddCutBtn").onclick = addCut;
  $("directorAddBlackoutBtn").onclick = addBlackout;
  $("directorAddMarkerBtn").onclick = addMarker;
  $("directorAddWarpBtn").onclick = addWarp;
  $("directorAddCameraKeyBtn").onclick = addCameraKey;
  $("directorDeleteKeyBtn").onclick = deleteCurrentKey;
  $("directorRefreshThumbnailsBtn").onclick = () => refreshThumbnails().catch((error) => status(error.message, error.message, true));
  $("directorScreenshotBtn").onclick = screenshotExport;
  $("directorVideoExportBtn").onclick = () => runVideoExport();
  $("directorContactSheetBtn").onclick = () => exportContactSheet();
  $("directorReferencePackBtn").onclick = exportReferencePack;
  $("directorCancelExportBtn").onclick = cancelExport;
  $("directorResumeExportBtn").onclick = resumeExport;
  $("directorProjectName").addEventListener("change", (event) => {
    if (!state.project || !event.target.value.trim()) return;
    const before = projectSnapshot();
    state.project.name = event.target.value.trim();
    pushUndo(before, "Rename project");
    scheduleSave();
  });
  bindProjectSettings();
  document.querySelector(".tab[data-tab='director']")?.addEventListener("click", () => Promise.all([loadProjects(), loadCatalogs()]));
  document.addEventListener("click", (event) => { if (!event.target.closest?.("[data-director-card],.director-project-actions")) hideProjectMenu(); });
  addEventListener("resize", resizeRuntime);
  addEventListener("keydown", (event) => {
    if (!state.project || $("directorEditorModal").classList.contains("hidden") || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); }
    else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") { event.preventDefault(); redo(); }
    else if (event.key === "Delete") deleteSelected();
    else if (event.key.toLowerCase() === "w") setGizmoMode("translate");
    else if (event.key.toLowerCase() === "e") setGizmoMode("rotate");
    else if (event.key.toLowerCase() === "r") setGizmoMode("scale");
    else if (event.key === " ") { event.preventDefault(); $("directorPlayBtn").click(); }
  });
  addEventListener("motionbuilder:languagechange", () => {
    if (state.project) { renderOutliner(); renderTimeline(); renderInspector(); syncProjectSettingsUi(); syncCameraViewUi(); }
    renderProjects();
    syncDirectorLanguageUi();
    if (state.lastStatus) status(state.lastStatus.zh, state.lastStatus.en, state.lastStatus.isError);
  });
  setInterval(async () => {
    if (!state.project || $("directorEditorModal").classList.contains("hidden")) return;
    try {
      await flushSave();
      await api(`/api/director-projects/${state.project.id}/snapshots`, jsonOptions("POST", { name: `Autosave ${new Date().toISOString()}`, kind: "autosave" }));
    } catch (error) { console.warn("Director autosave snapshot failed", error); }
  }, 5 * 60 * 1000);
}

async function initDirectorStudio() {
  if (state.initialized || !$("directorProjectGrid")) return;
  state.initialized = true;
  bindDirectorEvents();
  syncUndoButtons();
  syncDirectorLanguageUi();
  try { await Promise.all([loadProjects(), loadCatalogs()]); }
  catch (error) { status(`导演仓库初始化失败：${error.message}`, `Director warehouse initialization failed: ${error.message}`, true); }
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initDirectorStudio, { once: true });
else initDirectorStudio();
