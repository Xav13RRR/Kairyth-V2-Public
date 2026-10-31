import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const $ = (id) => document.getElementById(id);
const state = {
  terrains: [],
  scenes: new Map(),
  selectedId: "",
  gaitTerrainId: "",
  gaitPlayer: null,
  finalPlayer: null,
  editor: null,
  somaScene: null,
  terrainPreview: null,
};

function language() {
  return window.MotionBuilderI18n?.getLanguage?.() === "en" ? "en" : "zh";
}

function copy(zh, en) {
  return language() === "en" ? en : zh;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.text()) || String(response.status));
  return response.headers.get("content-type")?.includes("json") ? response.json() : response;
}

function bilingual(zh, en, tag = "span") {
  return `<${tag} data-terrain-zh="${escapeHtml(zh)}" data-terrain-en="${escapeHtml(en)}"></${tag}>`;
}

function translateUi(root = document) {
  const lang = language();
  root.querySelectorAll?.("[data-terrain-zh]").forEach((element) => {
    element.textContent = element.dataset[lang === "en" ? "terrainEn" : "terrainZh"] || "";
  });
  const tab = $("terrainLibraryTab");
  if (tab) tab.textContent = copy("地形仓库", "Terrain Library");
  if (state.editor?.sceneData && $("terrainEditorTitle")) {
    $("terrainEditorTitle").textContent = `${copy("地形编辑器", "Terrain Editor")} · ${state.editor.sceneData.name}`;
    $("terrainEditorStatus").textContent = copy(
      "地形已载入。左键使用笔刷，按住 Shift 反向雕刻。",
      "Terrain loaded. Use LMB to paint; hold Shift to invert sculpting.",
    );
  }
  populateGaitTerrainSelect();
  if (state.finalPlayer?.terrainObject && $("finalTerrainToggle")) {
    $("finalTerrainToggle").textContent = state.finalPlayer.terrainObject.visible
      ? copy("隐藏地形", "Hide Terrain")
      : copy("显示地形", "Show Terrain");
  }
  renderTerrainGrid();
}

function installShell() {
  const tabs = document.querySelector(".tabs");
  const languageSwitch = tabs?.querySelector(".studio-language-switch");
  const rigTab = tabs?.querySelector('.tab[data-tab="rig"]');
  if (tabs && languageSwitch && !$("terrainLibraryTab")) {
    const button = document.createElement("button");
    button.id = "terrainLibraryTab";
    button.className = "tab";
    button.dataset.tab = "terrain";
    button.textContent = copy("地形仓库", "Terrain Library");
    button.addEventListener("click", () => loadTerrains());
    tabs.insertBefore(button, rigTab || languageSwitch);
  }

  const main = document.querySelector("main");
  const assets = $("tab-assets");
  if (main && assets && !$("tab-terrain")) {
    const section = document.createElement("section");
    section.id = "tab-terrain";
    section.className = "tabpage terrain-library-page";
    section.innerHTML = `
      <div class="page-title asset-title animation-title terrain-library-title"><div><h1>${bilingual("地形创建与编辑", "Terrain Creation & Editing")}</h1></div></div>
      <div class="asset-action-row animation-action-row terrain-library-actions"><button id="newTerrainBtn" class="primary" type="button">${bilingual("新建地形", "New Terrain")}</button><p id="terrainLibraryStatus" class="status"></p></div>
      <div class="page-title asset-title asset-library-title animation-library-title terrain-library-title"><div><h1>${bilingual("地形资产仓库管理", "Terrain Asset Library Management")}</h1></div></div>
      <div id="terrainGrid" class="card-grid"></div>`;
    assets.insertAdjacentElement("afterend", section);
  }

  if (!$("terrainCreateModal")) {
    document.body.insertAdjacentHTML("beforeend", `
      <div id="terrainCreateModal" class="preview-modal hidden">
        <div class="preview-modal-card terrain-create-card">
          <div class="editor-head"><strong>${bilingual("新建地形", "New Terrain")}</strong><button id="closeTerrainCreate" type="button">${bilingual("关闭", "Close")}</button></div>
          <form id="terrainCreateForm" class="terrain-create-form">
            <label>${bilingual("地形名称", "Terrain Name")}<input id="terrainCreateName" maxlength="80" value="New Terrain" required /></label>
            <div class="row"><label>${bilingual("X 尺寸 / 米", "X Size / m")}<input id="terrainCreateSizeX" type="number" min="0.5" max="4096" step="0.5" value="8" /></label><label>${bilingual("Z 尺寸 / 米", "Z Size / m")}<input id="terrainCreateSizeZ" type="number" min="0.5" max="4096" step="0.5" value="8" /></label></div>
            <label>${bilingual("高度图分辨率", "Heightmap Resolution")}<select id="terrainCreateResolution"><option>33</option><option>65</option><option selected>129</option><option>257</option><option>513</option></select></label>
            <button class="primary" type="submit">${bilingual("创建并开始编辑", "Create & Start Editing")}</button>
            <p id="terrainCreateStatus" class="status"></p>
          </form>
        </div>
      </div>`);
  }

  if (!$("terrainEditorModal")) {
    document.body.insertAdjacentHTML("beforeend", `
      <div id="terrainEditorModal" class="preview-modal hidden">
        <div class="preview-modal-card terrain-editor-card">
          <div class="editor-head"><strong id="terrainEditorTitle">Terrain Editor</strong><button id="closeTerrainEditor" type="button">${bilingual("关闭", "Close")}</button></div>
          <div class="terrain-editor-layout">
            <aside class="terrain-tool-rail">
              <h3>${bilingual("造型工具", "Sculpt Tools")}</h3>
              <div id="terrainToolGrid" class="terrain-tool-grid">
                <button class="active" data-terrain-tool="sculpt" type="button">${bilingual("雕刻", "Sculpt")}</button>
                <button data-terrain-tool="sculpt_invert" type="button">${bilingual("雕刻（反向）", "Sculpt (Invert)")}</button>
                <button data-terrain-tool="smooth" type="button">${bilingual("平滑", "Smooth")}</button>
                <button data-terrain-tool="flatten" type="button">${bilingual("压平", "Flatten")}</button>
                <button data-terrain-tool="ramp" type="button">${bilingual("坡道", "Ramp")}</button>
                <button data-terrain-tool="noise" type="button">${bilingual("噪声", "Noise")}</button>
                <button data-terrain-tool="erase" type="button">${bilingual("擦除", "Erase")}</button>
                <button data-terrain-tool="overall" type="button">${bilingual("总体调节", "Overall Adjust")}</button>
              </div>
              <div id="terrainOverallPanel" class="terrain-overall-panel hidden">
                <div class="terrain-overall-modes"><button id="terrainOverallTranslate" class="active" type="button">${bilingual("平移", "Translate")}</button><button id="terrainOverallRotate" type="button">${bilingual("旋转", "Rotate")}</button><button id="terrainOverallScale" type="button">${bilingual("缩放", "Scale")}</button></div>
                <div class="terrain-overall-values"><label>X / m<input id="terrainOverallX" type="number" step="0.05" value="0" /></label><label>Y / m<input id="terrainOverallY" type="number" step="0.05" value="0" /></label><label>Z / m<input id="terrainOverallZ" type="number" step="0.05" value="0" /></label><label>${bilingual("朝向 / 度", "Yaw / deg")}<input id="terrainOverallYaw" type="number" step="1" value="0" /></label><label>${bilingual("统一比例", "Uniform Scale")}<input id="terrainOverallScaleValue" type="number" min="0.01" max="100" step="0.05" value="1" /></label></div>
                <button id="terrainOverallReset" type="button">${bilingual("复位总体调节", "Reset Overall Adjust")}</button>
                <small>${bilingual("旋转约束为 Y 轴，缩放保持统一比例。", "Rotation is Y-axis constrained; scale remains uniform.")}</small>
              </div>
              <div id="terrainBrushPanel">
                <h3>${bilingual("笔刷", "Brush")}</h3>
                <label><span class="terrain-value-line">${bilingual("半径 / 米", "Radius / m")}<output id="terrainBrushRadiusValue">0.60</output></span><input id="terrainBrushRadius" type="range" min="0.1" max="3" step="0.05" value="0.6" /></label>
                <label><span class="terrain-value-line">${bilingual("强度", "Strength")}<output id="terrainBrushStrengthValue">0.35</output></span><input id="terrainBrushStrength" type="range" min="0.01" max="1" step="0.01" value="0.35" /></label>
                <label><span class="terrain-value-line">${bilingual("衰减", "Falloff")}<output id="terrainBrushFalloffValue">0.55</output></span><input id="terrainBrushFalloff" type="range" min="0" max="1" step="0.01" value="0.55" /></label>
              </div>
              <div class="terrain-history-actions"><button id="terrainUndo" type="button">${bilingual("撤销", "Undo")}</button><button id="terrainRedo" type="button">${bilingual("重做", "Redo")}</button></div>
              <h3 style="margin-top:20px">${bilingual("参考显示", "References")}</h3>
              <div class="terrain-view-options"><button id="terrainPlaneToggle" class="active" type="button">${bilingual("基准平面", "Reference Plane")}</button><button id="terrainSomaToggle" class="active" type="button">${bilingual("标准 SOMA", "Standard SOMA")}</button></div>
              <p class="terrain-editor-status" id="terrainEditorStatus"></p>
              <div class="terrain-editor-actions"><button id="terrainSave" class="primary" type="button">${bilingual("保存地形", "Save Terrain")}</button><button id="terrainReset" type="button">${bilingual("恢复已保存状态", "Revert to Saved")}</button></div>
            </aside>
            <div class="terrain-editor-viewport" id="terrainEditorViewport"><canvas id="terrainEditorCanvas"></canvas><div class="terrain-editor-hud"><span>Y-UP · METERS</span><span>${bilingual("左键造型 · Alt+拖拽环绕 · Shift 反向", "LMB sculpt · Alt+drag orbit · Shift invert")}</span></div></div>
          </div>
        </div>
      </div>`);
  }

  installGaitControls();
  installFinalTerrainToggle();
  bindUi();
  translateUi();
}

function installGaitControls() {
  const form = $("gaitIKForm");
  if (!form || $("gaitIKTerrainPanel")) return;
  const parameterGrid = form.querySelector(".gait-ik-parameter-grid");
  const panel = document.createElement("div");
  panel.id = "gaitIKTerrainPanel";
  panel.className = "gait-terrain-panel";
  panel.innerHTML = `
    <div class="gait-terrain-heading"><strong>${bilingual("地形贴合", "Terrain Grounding")}</strong><span id="gaitIKTerrainState" class="axis-pill">PLANE Y=0</span></div>
    <label>${bilingual("地形资产", "Terrain Asset")}<select id="gaitIKTerrain"><option value="">${copy("基准平面", "Reference Plane")}</option></select></label>
    <div class="gait-terrain-actions"><button id="gaitIKAddTerrain" type="button">${bilingual("添加地形", "Add Terrain")}</button><button id="gaitIKRemoveTerrain" type="button">${bilingual("移除地形", "Remove Terrain")}</button><button id="gaitIKMoveStart" type="button">${bilingual("移动初始位置", "Move Start Position")}</button></div>
    <div id="gaitIKStartPanel" class="gait-start-panel hidden">
      <div class="gait-start-modes"><button id="gaitIKStartTranslate" class="active" type="button">${bilingual("平移", "Translate")}</button><button id="gaitIKStartRotate" type="button">${bilingual("旋转", "Rotate")}</button><button id="gaitIKStartReset" type="button">${bilingual("重置", "Reset")}</button></div>
      <div class="gait-start-values"><label>X / m<input id="gaitIKStartX" type="number" step="0.05" value="0" /></label><label>Z / m<input id="gaitIKStartZ" type="number" step="0.05" value="0" /></label><label>${bilingual("朝向 / 度", "Yaw / deg")}<input id="gaitIKStartYaw" type="number" step="1" value="0" /></label></div>
      <span id="gaitIKStartGround" class="gait-start-ground">GROUND Y = 0.000 m</span>
    </div>`;
  form.insertBefore(panel, parameterGrid);
  parameterGrid?.insertAdjacentHTML("beforeend", `<label>${bilingual("地形采样平滑帧", "Terrain Sampling Smoothing")}<input id="gaitIKTerrainSmooth" type="number" min="1" max="61" step="2" value="5" /></label><label>${bilingual("最大可贴合坡度 / 度", "Maximum Walkable Slope / deg")}<input id="gaitIKMaxSlope" type="number" min="1" max="89" step="1" value="55" /></label>`);
  const commit = $("commitGaitIKBtn");
  commit?.insertAdjacentHTML("beforebegin", `<label class="check gait-terrain-toggle"><input id="gaitIKIncludeTerrain" type="checkbox" disabled /> ${bilingual("随新动画保存地形快照", "Save Terrain Snapshot with New Animation")}</label>`);
}

function installFinalTerrainToggle() {
  const preview = document.querySelector(".final-preview-stage");
  if (!preview || $("finalTerrainToggle")) return;
  const button = document.createElement("button");
  button.id = "finalTerrainToggle";
  button.type = "button";
  button.className = "final-terrain-toggle hidden";
  button.textContent = copy("隐藏地形", "Hide Terrain");
  preview.appendChild(button);
}

function bindUi() {
  if ($("newTerrainBtn")?.dataset.bound) return;
  $("newTerrainBtn").dataset.bound = "1";
  $("newTerrainBtn").onclick = () => $("terrainCreateModal").classList.remove("hidden");
  $("closeTerrainCreate").onclick = () => $("terrainCreateModal").classList.add("hidden");
  $("terrainCreateForm").onsubmit = createTerrain;
  $("closeTerrainEditor").onclick = closeEditor;
  $("terrainSave").onclick = saveTerrain;
  $("terrainReset").onclick = revertTerrain;
  $("terrainUndo").onclick = undoTerrain;
  $("terrainRedo").onclick = redoTerrain;
  $("terrainPlaneToggle").onclick = () => toggleReference("plane");
  $("terrainSomaToggle").onclick = () => toggleReference("soma");
  $("terrainOverallTranslate").onclick = () => setTerrainOverallMode("translate");
  $("terrainOverallRotate").onclick = () => setTerrainOverallMode("rotate");
  $("terrainOverallScale").onclick = () => setTerrainOverallMode("scale");
  $("terrainOverallReset").onclick = resetTerrainOverall;
  ["terrainOverallX", "terrainOverallY", "terrainOverallZ", "terrainOverallYaw", "terrainOverallScaleValue"].forEach((id) => $(id).onchange = updateTerrainOverallFromInputs);
  document.querySelectorAll("[data-terrain-tool]").forEach((button) => {
    button.onclick = () => setTerrainTool(button.dataset.terrainTool);
  });
  for (const key of ["Radius", "Strength", "Falloff"]) {
    const input = $(`terrainBrush${key}`), output = $(`terrainBrush${key}Value`);
    input.oninput = () => { output.value = Number(input.value).toFixed(2); updateBrushRing(); };
  }
  $("gaitIKAddTerrain").onclick = applySelectedGaitTerrain;
  $("gaitIKRemoveTerrain").onclick = removeGaitTerrain;
  $("gaitIKMoveStart").onclick = toggleGaitStartPanel;
  $("gaitIKStartTranslate").onclick = () => setGaitTransformMode("translate");
  $("gaitIKStartRotate").onclick = () => setGaitTransformMode("rotate");
  $("gaitIKStartReset").onclick = resetGaitStart;
  ["gaitIKStartX", "gaitIKStartZ", "gaitIKStartYaw"].forEach((id) => $(id).oninput = updateGaitStartFromInputs);
  $("finalTerrainToggle").onclick = toggleFinalTerrain;
  document.addEventListener("click", (event) => { if (state.selectedId && !event.target.closest?.("[data-terrain-card],#terrainActionMenu,#terrainEditorModal,#terrainCreateModal")) clearTerrainSelection(); });
  window.addEventListener("motionbuilder:languagechange", () => translateUi());
}

async function loadTerrains() {
  try {
    state.terrains = await api("/api/terrains");
    populateGaitTerrainSelect();
    renderTerrainGrid();
    window.dispatchEvent(new CustomEvent("motionbuilder:terrainschange", { detail: { terrains: state.terrains } }));
    return state.terrains;
  } catch (error) {
    if ($("terrainLibraryStatus")) $("terrainLibraryStatus").textContent = error.message;
    return [];
  }
}

function syncTerrainFocusClass() {
  const active = Boolean(state.selectedId);
  document.documentElement.classList.toggle("terrain-focus-active", active);
  document.body.classList.toggle("terrain-focus-active", active);
}

function terrainActionMenu() {
  let menu = $("terrainActionMenu");
  if (menu) return menu;
  menu = document.createElement("div"); menu.id = "terrainActionMenu"; menu.className = "asset-action-menu terrain-action-menu hidden";
  menu.onclick = (event) => event.stopPropagation(); document.body.appendChild(menu); return menu;
}

function disposeTerrainPreview() {
  const preview = state.terrainPreview; if (!preview) return;
  preview.renderer.setAnimationLoop(null); preview.renderer.dispose();
  preview.group.traverse((object) => { object.geometry?.dispose?.(); object.material?.dispose?.(); });
  state.terrainPreview = null;
}

function clearTerrainSelection(render = true) {
  if (!state.selectedId && !document.body.classList.contains("terrain-focus-active")) return;
  state.selectedId = ""; disposeTerrainPreview(); terrainActionMenu().classList.add("hidden"); syncTerrainFocusClass();
  if (render) renderTerrainGrid();
}

function showTerrainActionMenu(card, terrain) {
  const menu = terrainActionMenu(), rect = card.getBoundingClientRect();
  menu.innerHTML = `<button data-terrain-edit="${terrain.id}">${copy("编辑", "Edit")}</button><button data-terrain-rename="${terrain.id}">${copy("重命名", "Rename")}</button><button data-terrain-delete="${terrain.id}">${copy("删除", "Delete")}</button>`;
  menu.style.left = `${Math.max(12, rect.left)}px`; menu.style.top = `${Math.min(rect.bottom + 8, innerHeight - 64)}px`; menu.classList.remove("hidden");
  const menuWidth = menu.getBoundingClientRect().width; menu.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - menuWidth - 12))}px`;
  menu.querySelector("[data-terrain-edit]").onclick = () => { clearTerrainSelection(); openEditor(terrain.id); };
  menu.querySelector("[data-terrain-rename]").onclick = () => renameTerrain(terrain.id);
  menu.querySelector("[data-terrain-delete]").onclick = () => deleteTerrain(terrain.id);
}

function selectTerrainCard(id) {
  const terrain = state.terrains.find((row) => row.id === id); if (!terrain) return;
  state.selectedId = id; syncTerrainFocusClass(); renderTerrainGrid();
}

function renderTerrainGrid() {
  const grid = $("terrainGrid"); if (!grid) return;
  disposeTerrainPreview(); terrainActionMenu().classList.add("hidden");
  if (state.selectedId && !state.terrains.some((terrain) => terrain.id === state.selectedId)) state.selectedId = "";
  syncTerrainFocusClass();
  if (!state.terrains.length) {
    grid.innerHTML = `<div class="empty">${copy("暂无地形。点击“新建地形”开始。", "No terrain assets yet. Choose New Terrain to begin.")}</div>`;
    return;
  }
  grid.innerHTML = state.terrains.map((terrain) => {
    const selected = terrain.id === state.selectedId, preview = selected ? `<canvas class="asset-wireframe-canvas terrain-expanded-preview" data-terrain-preview-canvas="${terrain.id}"></canvas>` : "";
    return `<article class="character-card asset-tech-card terrain-tech-card asset-static ${selected ? "selected" : ""}" data-terrain-card="${terrain.id}">
      <div class="asset-wireframe-slot">${preview}</div>
      <div class="asset-card-head"><h3 class="asset-name">${escapeHtml(terrain.name)}</h3><span class="asset-kind">.TERRAIN</span></div>
      <p class="asset-meta">${Number(terrain.size_x).toFixed(1)} × ${Number(terrain.size_z).toFixed(1)} m<br>${terrain.resolution_x} × ${terrain.resolution_z} HEIGHTFIELD</p>
    </article>`;
  }).join("");
  grid.querySelectorAll("[data-terrain-card]").forEach((card) => card.onclick = (event) => { event.stopPropagation(); selectTerrainCard(card.dataset.terrainCard); });
  if (state.selectedId) requestAnimationFrame(() => {
    const terrain = state.terrains.find((row) => row.id === state.selectedId), card = document.querySelector(`[data-terrain-card="${CSS.escape(state.selectedId)}"]`);
    if (terrain && card) showTerrainActionMenu(card, terrain);
    renderSelectedTerrainPreview();
  });
}

function fitTerrainPreviewCamera(preview, width, height) {
  const sphere = preview.geometry.boundingSphere; if (!sphere || sphere.isEmpty()) return;
  const aspect = width / Math.max(1, height), halfVertical = THREE.MathUtils.degToRad(preview.camera.fov * .5), halfHorizontal = Math.atan(Math.tan(halfVertical) * aspect), limitingHalfFov = Math.max(.08, Math.min(halfVertical, halfHorizontal)), distance = Math.max(.8, sphere.radius / Math.sin(limitingHalfFov) * 1.18);
  const direction = new THREE.Vector3(1, Math.SQRT2, 1).normalize(); preview.camera.position.copy(sphere.center).addScaledVector(direction, distance); preview.camera.near = Math.max(.01, distance - sphere.radius * 1.5); preview.camera.far = Math.max(100, distance + sphere.radius * 3); preview.camera.aspect = aspect; preview.camera.lookAt(sphere.center); preview.camera.updateProjectionMatrix();
}

async function renderSelectedTerrainPreview() {
  const id = state.selectedId, canvas = id && document.querySelector(`[data-terrain-preview-canvas="${CSS.escape(id)}"]`); if (!id || !canvas || !THREE) return;
  try {
    const sceneData = await terrainScene(id); if (state.selectedId !== id || !canvas.isConnected) return;
    disposeTerrainPreview(); const geometry = buildHeightfieldGeometry(sceneData); geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true }); renderer.setClearColor(0x000000, 0); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(38, 1, .01, 100), group = new THREE.Group();
    const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x49677c, roughness: .86, metalness: .04, side: THREE.DoubleSide }));
    const wireGeometry = new THREE.WireframeGeometry(geometry), wire = new THREE.LineSegments(wireGeometry, new THREE.LineBasicMaterial({ color: 0xe4ff4e, transparent: true, opacity: .48 }));
    group.add(surface, wire); scene.add(group, new THREE.HemisphereLight(0xeaf6ff, 0x101722, 2.7)); const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(4, 7, 5); scene.add(key);
    const preview = { id, canvas, renderer, scene, camera, group, geometry, wireGeometry, width: 0, height: 0 }; state.terrainPreview = preview;
    renderer.setAnimationLoop((time) => { if (state.terrainPreview !== preview) return; const rect = canvas.getBoundingClientRect(), width = Math.max(1, Math.round(rect.width)), height = Math.max(1, Math.round(rect.height)); if (width !== preview.width || height !== preview.height) { preview.width = width; preview.height = height; renderer.setSize(width, height, false); fitTerrainPreviewCamera(preview, width, height); } group.rotation.y = time * .00032; renderer.render(scene, camera); });
  } catch (error) { if ($("terrainLibraryStatus")) $("terrainLibraryStatus").textContent = error.message; }
}

async function createTerrain(event) {
  event.preventDefault();
  const status = $("terrainCreateStatus");
  status.textContent = copy("正在创建地形…", "Creating terrain…");
  try {
    const row = await api("/api/terrains", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: $("terrainCreateName").value,
        size_x: +$("terrainCreateSizeX").value,
        size_z: +$("terrainCreateSizeZ").value,
        resolution: +$("terrainCreateResolution").value,
      }),
    });
    $("terrainCreateModal").classList.add("hidden");
    status.textContent = "";
    await loadTerrains();
    await openEditor(row.id);
  } catch (error) { status.textContent = error.message; }
}

async function renameTerrain(id) {
  const terrain = state.terrains.find((row) => row.id === id);
  const name = prompt(copy("新的地形名称", "New terrain name"), terrain?.name || "");
  if (!name?.trim()) return;
  await api(`/api/terrains/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: name.trim() }) });
  await loadTerrains();
}

async function deleteTerrain(id) {
  if (!confirm(copy("删除该地形资产？已随动画保存的快照不会被删除。", "Delete this terrain asset? Snapshots saved with animations will remain."))) return;
  await api(`/api/terrains/${id}`, { method: "DELETE" });
  state.scenes.delete(id);
  if (state.selectedId === id) state.selectedId = "";
  if (state.gaitTerrainId === id) removeGaitTerrain();
  await loadTerrains();
}

async function terrainScene(id, force = false) {
  if (!id) return null;
  if (force || !state.scenes.has(id)) state.scenes.set(id, await api(`/api/terrains/${id}/scene`));
  return state.scenes.get(id);
}

function buildHeightfieldGeometry(sceneData) {
  const heights = sceneData.heights || [], rows = heights.length, columns = heights[0]?.length || 0;
  const positions = new Float32Array(rows * columns * 3), indices = [];
  for (let z = 0; z < rows; z++) for (let x = 0; x < columns; x++) {
    const index = z * columns + x;
    positions[index * 3] = -sceneData.size_x * .5 + x * sceneData.size_x / Math.max(1, columns - 1);
    positions[index * 3 + 1] = Number(heights[z][x]);
    positions[index * 3 + 2] = -sceneData.size_z * .5 + z * sceneData.size_z / Math.max(1, rows - 1);
    if (x < columns - 1 && z < rows - 1) {
      const a = index, b = index + 1, c = index + columns, d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices); geometry.computeVertexNormals();
  geometry.userData.rows = rows; geometry.userData.columns = columns;
  return geometry;
}

function terrainTransform(value = {}) {
  const translation = Array.isArray(value.translation) && value.translation.length === 3 ? value.translation.map(Number) : [0, 0, 0];
  return { translation, yaw_degrees: Number(value.yaw_degrees) || 0, uniform_scale: Math.max(.01, Number(value.uniform_scale) || 1) };
}

function cloneTerrainTransform(value) {
  const resolved = terrainTransform(value); return { ...resolved, translation: resolved.translation.slice() };
}

function terrainGroup(sceneData, editor = false) {
  const group = new THREE.Group(), geometry = buildHeightfieldGeometry(sceneData);
  const material = new THREE.MeshStandardMaterial({ color: editor ? 0x506779 : 0x3c5362, roughness: .92, metalness: .02, side: THREE.DoubleSide, transparent: !editor, opacity: editor ? 1 : .82 });
  const mesh = new THREE.Mesh(geometry, material); mesh.receiveShadow = true; mesh.userData.terrainSurface = true;
  const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geometry), new THREE.LineBasicMaterial({ color: editor ? 0x7ba5bf : 0x7694a8, transparent: true, opacity: editor ? .22 : .16 }));
  group.add(mesh, wire); group.userData = { mesh, wire, sceneData };
  const transform = terrainTransform(sceneData.transform);
  group.position.fromArray(transform.translation); group.rotation.y = THREE.MathUtils.degToRad(transform.yaw_degrees); group.scale.setScalar(transform.uniform_scale);
  return group;
}

async function openEditor(id) {
  const sceneData = await terrainScene(id, true);
  $("terrainEditorModal").classList.remove("hidden");
  $("terrainEditorTitle").textContent = `${copy("地形编辑器", "Terrain Editor")} · ${sceneData.name}`;
  initEditor3D();
  loadEditorScene(sceneData);
  requestAnimationFrame(resizeEditor);
}

function initEditor3D() {
  if (state.editor) return;
  const canvas = $("terrainEditorCanvas"), renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1115);
  const camera = new THREE.PerspectiveCamera(42, 1, .01, 500); camera.position.set(6, 4.5, 7);
  const controls = new OrbitControls(camera, canvas); controls.target.set(0, 0, 0); controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xdbe8ff, 0x172238, 2.35));
  const key = new THREE.DirectionalLight(0xffffff, 2.8); key.position.set(4, 8, 5); key.castShadow = true; scene.add(key);
  const plane = new THREE.GridHelper(24, 48, 0x46657a, 0x242b31); plane.position.y = -.002; scene.add(plane);
  const brushRing = new THREE.Mesh(new THREE.RingGeometry(.96, 1, 64), new THREE.MeshBasicMaterial({ color: 0x74c9ff, side: THREE.DoubleSide, transparent: true, opacity: .8, depthTest: false }));
  brushRing.rotation.x = -Math.PI / 2; brushRing.visible = false; brushRing.renderOrder = 8; scene.add(brushRing);
  const overallControl = new TransformControls(camera, canvas); overallControl.setSpace("world"); overallControl.setSize(.82); scene.add(overallControl.getHelper()); overallControl.getHelper().visible = false;
  const editor = { renderer, scene, camera, controls, plane, soma: null, brushRing, overallControl, overallMode: "translate", group: null, mesh: null, wire: null, sceneData: null, heights: null, saved: null, transform: terrainTransform(), savedTransform: terrainTransform(), tool: "sculpt", undo: [], redo: [], painting: false, changed: false, flattenHeight: 0, rampStart: null, raycaster: new THREE.Raycaster(), pointer: new THREE.Vector2() };
  state.editor = editor;
  overallControl.addEventListener("dragging-changed", (event) => { controls.enabled = !event.value; if (event.value) pushTerrainUndo(); else { editor.changed = true; updateTerrainHistoryUi(); } });
  overallControl.addEventListener("objectChange", syncTerrainOverallFromControl);
  canvas.addEventListener("pointerdown", beginBrush);
  canvas.addEventListener("pointermove", moveBrush);
  canvas.addEventListener("pointerleave", endBrush);
  window.addEventListener("pointerup", endBrush);
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
  loadSomaReference();
}

function loadEditorScene(sceneData) {
  const editor = state.editor;
  if (editor.group) {
    editor.scene.remove(editor.group);
    editor.group.traverse((object) => { object.geometry?.dispose?.(); object.material?.dispose?.(); });
  }
  editor.sceneData = sceneData;
  editor.heights = sceneData.heights.map((row) => row.map(Number));
  editor.saved = editor.heights.map((row) => row.slice());
  editor.transform = terrainTransform(sceneData.transform); editor.savedTransform = cloneTerrainTransform(editor.transform);
  editor.undo = []; editor.redo = []; editor.changed = false; editor.overallControl.detach();
  editor.group = terrainGroup({ ...sceneData, heights: editor.heights, transform: editor.transform }, true);
  editor.mesh = editor.group.userData.mesh; editor.wire = editor.group.userData.wire; editor.scene.add(editor.group);
  const extent = Math.max(sceneData.size_x, sceneData.size_z, 2);
  editor.camera.position.set(extent * .72, Math.max(2.6, extent * .48), extent * .82);
  editor.controls.target.set(0, (sceneData.min_height + sceneData.max_height) * .5, 0); editor.controls.update();
  setTerrainTool("sculpt"); updateTerrainHistoryUi();
  $("terrainEditorStatus").textContent = copy("地形已载入。左键使用笔刷，按住 Shift 反向雕刻。", "Terrain loaded. Use LMB to paint; hold Shift to invert sculpting.");
}

async function loadSomaReference() {
  try {
    state.somaScene ||= await api("/api/kimodo-runtime/default-scene");
    if (!state.editor || state.editor.soma) return;
    const packet = state.somaScene, matrices = packet.target_bind || packet.bind_matrices || [], positions = [];
    for (let index = 0; index < matrices.length; index++) {
      const parent = packet.parents?.[index];
      if (parent == null || parent < 0) continue;
      const a = new THREE.Vector3().setFromMatrixPosition(new THREE.Matrix4().set(...matrices[parent]));
      const b = new THREE.Vector3().setFromMatrixPosition(new THREE.Matrix4().set(...matrices[index]));
      positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const soma = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xffc45e, transparent: true, opacity: .78, depthTest: false }));
    soma.renderOrder = 7; state.editor.soma = soma; state.editor.scene.add(soma);
  } catch (error) {
    if ($("terrainEditorStatus")) $("terrainEditorStatus").textContent = copy("标准 SOMA 参考加载失败。", "Standard SOMA reference failed to load.");
  }
}

function terrainSnapshot(editor = state.editor) {
  return { heights: editor.heights.map((row) => row.slice()), transform: cloneTerrainTransform(editor.transform) };
}

function restoreTerrainSnapshot(snapshot) {
  const editor = state.editor; if (!editor || !snapshot) return;
  editor.heights = snapshot.heights.map((row) => row.slice()); updateEditorGeometry(); applyTerrainOverallTransform(snapshot.transform); editor.changed = true;
}

function pushTerrainUndo() {
  const editor = state.editor; if (!editor) return;
  editor.undo.push(terrainSnapshot(editor)); if (editor.undo.length > 30) editor.undo.shift(); editor.redo = []; updateTerrainHistoryUi();
}

function applyTerrainOverallTransform(value, syncInputs = true) {
  const editor = state.editor; if (!editor?.group) return;
  editor.transform = cloneTerrainTransform(value); editor.group.position.fromArray(editor.transform.translation); editor.group.rotation.set(0, THREE.MathUtils.degToRad(editor.transform.yaw_degrees), 0); editor.group.scale.setScalar(editor.transform.uniform_scale); editor.group.updateMatrixWorld(true);
  if (syncInputs) syncTerrainOverallInputs();
}

function syncTerrainOverallInputs() {
  const transform = state.editor?.transform; if (!transform) return;
  $("terrainOverallX").value = transform.translation[0].toFixed(3); $("terrainOverallY").value = transform.translation[1].toFixed(3); $("terrainOverallZ").value = transform.translation[2].toFixed(3); $("terrainOverallYaw").value = transform.yaw_degrees.toFixed(2); $("terrainOverallScaleValue").value = transform.uniform_scale.toFixed(3);
}

function syncTerrainOverallFromControl() {
  const editor = state.editor; if (!editor?.group) return;
  if (editor.overallMode === "rotate") editor.group.rotation.set(0, editor.group.rotation.y, 0);
  if (editor.overallMode === "scale") {
    const axis = editor.overallControl.axis, value = axis === "X" ? editor.group.scale.x : axis === "Y" ? editor.group.scale.y : axis === "Z" ? editor.group.scale.z : (editor.group.scale.x + editor.group.scale.y + editor.group.scale.z) / 3;
    editor.group.scale.setScalar(Math.max(.01, Math.min(100, value)));
  }
  editor.transform = { translation: editor.group.position.toArray(), yaw_degrees: THREE.MathUtils.radToDeg(editor.group.rotation.y), uniform_scale: editor.group.scale.x }; syncTerrainOverallInputs();
}

function setTerrainOverallMode(mode) {
  const editor = state.editor; if (!editor) return;
  editor.overallMode = ["rotate", "scale"].includes(mode) ? mode : "translate"; editor.overallControl.setMode(editor.overallMode); editor.overallControl.showX = editor.overallMode !== "rotate"; editor.overallControl.showY = true; editor.overallControl.showZ = editor.overallMode !== "rotate";
  ["Translate", "Rotate", "Scale"].forEach((name) => $("terrainOverall" + name).classList.toggle("active", editor.overallMode === name.toLowerCase()));
}

function updateTerrainOverallFromInputs() {
  const editor = state.editor; if (!editor) return; pushTerrainUndo();
  applyTerrainOverallTransform({ translation: [Number($("terrainOverallX").value) || 0, Number($("terrainOverallY").value) || 0, Number($("terrainOverallZ").value) || 0], yaw_degrees: Number($("terrainOverallYaw").value) || 0, uniform_scale: Math.max(.01, Math.min(100, Number($("terrainOverallScaleValue").value) || 1)) }); editor.changed = true;
}

function resetTerrainOverall() {
  const editor = state.editor; if (!editor) return; pushTerrainUndo(); applyTerrainOverallTransform(terrainTransform()); editor.changed = true; $("terrainEditorStatus").textContent = copy("总体调节已复位。", "Overall adjustment reset.");
}

function setTerrainTool(tool) {
  if (!state.editor) return;
  const editor = state.editor; editor.tool = tool;
  document.querySelectorAll("[data-terrain-tool]").forEach((button) => button.classList.toggle("active", button.dataset.terrainTool === tool));
  const overall = tool === "overall"; $("terrainOverallPanel").classList.toggle("hidden", !overall); $("terrainBrushPanel").classList.toggle("hidden", overall); editor.brushRing.visible = false;
  if (overall && editor.group) { editor.overallControl.attach(editor.group); editor.overallControl.getHelper().visible = true; setTerrainOverallMode(editor.overallMode); syncTerrainOverallInputs(); }
  else { editor.overallControl.detach(); editor.overallControl.getHelper().visible = false; }
}

function canvasHit(event) {
  const editor = state.editor, rect = editor.renderer.domElement.getBoundingClientRect();
  editor.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
  editor.raycaster.setFromCamera(editor.pointer, editor.camera);
  return editor.raycaster.intersectObject(editor.mesh, false)[0] || null;
}

function beginBrush(event) {
  if (!state.editor?.mesh || state.editor.tool === "overall" || event.button !== 0 || event.altKey) return;
  const hit = canvasHit(event); if (!hit) return;
  const point = state.editor.group.worldToLocal(hit.point.clone());
  event.preventDefault(); state.editor.painting = true; state.editor.controls.enabled = false; pushTerrainUndo();
  state.editor.flattenHeight = sampleEditorHeight(point.x, point.z);
  state.editor.rampStart = { x: point.x, z: point.z, height: state.editor.flattenHeight };
  applyBrush(point, event.shiftKey);
}

function moveBrush(event) {
  if (!state.editor?.mesh || state.editor.tool === "overall") { if (state.editor?.brushRing) state.editor.brushRing.visible = false; return; }
  const hit = canvasHit(event);
  if (!hit) { state.editor.brushRing.visible = false; return; }
  state.editor.brushRing.visible = true;
  state.editor.brushRing.position.set(hit.point.x, hit.point.y + .01, hit.point.z);
  updateBrushRing();
  if (state.editor.painting) applyBrush(state.editor.group.worldToLocal(hit.point.clone()), event.shiftKey);
}

function endBrush() {
  if (!state.editor?.painting) return;
  state.editor.painting = false; state.editor.controls.enabled = true; state.editor.changed = true; updateTerrainHistoryUi();
}

function sampleEditorHeight(x, z) {
  const editor = state.editor, rows = editor.heights.length, columns = editor.heights[0].length;
  const gx = Math.max(0, Math.min(columns - 1, (x / editor.sceneData.size_x + .5) * (columns - 1)));
  const gz = Math.max(0, Math.min(rows - 1, (z / editor.sceneData.size_z + .5) * (rows - 1)));
  return editor.heights[Math.round(gz)][Math.round(gx)];
}

function applyBrush(point, invert) {
  const editor = state.editor, heights = editor.heights, rows = heights.length, columns = heights[0].length;
  const radius = +$("terrainBrushRadius").value, strength = +$("terrainBrushStrength").value, falloff = +$("terrainBrushFalloff").value;
  const centerX = (point.x / editor.sceneData.size_x + .5) * (columns - 1), centerZ = (point.z / editor.sceneData.size_z + .5) * (rows - 1);
  const cellX = editor.sceneData.size_x / Math.max(1, columns - 1), cellZ = editor.sceneData.size_z / Math.max(1, rows - 1);
  const rangeX = Math.ceil(radius / cellX), rangeZ = Math.ceil(radius / cellZ), source = heights.map((row) => row.slice());
  const ramp = editor.rampStart, rampDx = point.x - (ramp?.x || 0), rampDz = point.z - (ramp?.z || 0), rampLength2 = Math.max(1e-6, rampDx * rampDx + rampDz * rampDz), pointHeight = sampleEditorHeight(point.x, point.z);
  for (let z = Math.max(0, Math.floor(centerZ - rangeZ)); z <= Math.min(rows - 1, Math.ceil(centerZ + rangeZ)); z++) {
    for (let x = Math.max(0, Math.floor(centerX - rangeX)); x <= Math.min(columns - 1, Math.ceil(centerX + rangeX)); x++) {
      const worldX = -editor.sceneData.size_x * .5 + x * cellX, worldZ = -editor.sceneData.size_z * .5 + z * cellZ;
      const distance = Math.hypot(worldX - point.x, worldZ - point.z); if (distance > radius) continue;
      const normalized = 1 - distance / radius, weight = Math.pow(normalized, 1 + falloff * 4), blend = Math.min(1, strength * weight);
      if (editor.tool === "smooth") {
        let sum = 0, count = 0;
        for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) if (source[z + dz]?.[x + dx] != null) { sum += source[z + dz][x + dx]; count++; }
        heights[z][x] += (sum / Math.max(1, count) - heights[z][x]) * blend;
      } else if (editor.tool === "flatten") heights[z][x] += (editor.flattenHeight - heights[z][x]) * blend;
      else if (editor.tool === "erase") heights[z][x] += (0 - heights[z][x]) * blend;
      else if (editor.tool === "noise") heights[z][x] += (Math.random() * 2 - 1) * .045 * blend * (invert ? -1 : 1);
      else if (editor.tool === "ramp" && ramp) {
        const projection = ((worldX - ramp.x) * rampDx + (worldZ - ramp.z) * rampDz) / rampLength2;
        const target = ramp.height + Math.max(0, Math.min(1, projection)) * (pointHeight - ramp.height);
        heights[z][x] += (target - heights[z][x]) * blend;
      } else {
        let direction = editor.tool === "sculpt_invert" ? -1 : 1; if (invert) direction *= -1;
        heights[z][x] += .055 * blend * direction;
      }
    }
  }
  updateEditorGeometry();
}

function updateEditorGeometry() {
  const editor = state.editor, positions = editor.mesh.geometry.attributes.position.array, rows = editor.heights.length, columns = editor.heights[0].length;
  for (let z = 0; z < rows; z++) for (let x = 0; x < columns; x++) positions[(z * columns + x) * 3 + 1] = editor.heights[z][x];
  editor.mesh.geometry.attributes.position.needsUpdate = true; editor.mesh.geometry.computeVertexNormals();
  editor.wire.geometry.dispose(); editor.wire.geometry = new THREE.WireframeGeometry(editor.mesh.geometry);
}

function updateBrushRing() {
  if (!state.editor) return;
  const radius = +$("terrainBrushRadius").value * (state.editor.transform?.uniform_scale || 1);
  state.editor.brushRing.scale.set(radius, radius, radius);
}

function updateTerrainHistoryUi() {
  if (!state.editor) return;
  $("terrainUndo").disabled = !state.editor.undo.length; $("terrainRedo").disabled = !state.editor.redo.length;
}

function undoTerrain() {
  const editor = state.editor; if (!editor?.undo.length) return;
  editor.redo.push(terrainSnapshot(editor)); restoreTerrainSnapshot(editor.undo.pop()); updateTerrainHistoryUi();
}

function redoTerrain() {
  const editor = state.editor; if (!editor?.redo.length) return;
  editor.undo.push(terrainSnapshot(editor)); restoreTerrainSnapshot(editor.redo.pop()); updateTerrainHistoryUi();
}

function toggleReference(kind) {
  const editor = state.editor, object = kind === "plane" ? editor?.plane : editor?.soma, button = kind === "plane" ? $("terrainPlaneToggle") : $("terrainSomaToggle");
  if (!object) return; object.visible = !object.visible; button.classList.toggle("active", object.visible);
}

async function saveTerrain() {
  const editor = state.editor; if (!editor?.sceneData) return;
  $("terrainSave").disabled = true; $("terrainEditorStatus").textContent = copy("正在保存高度图…", "Saving heightmap…");
  try {
    const row = await api(`/api/terrains/${editor.sceneData.id}/heightfield`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ heights: editor.heights, transform: editor.transform, expected_revision: editor.sceneData.revision }) });
    editor.sceneData = { ...editor.sceneData, ...row, heights: editor.heights, transform: cloneTerrainTransform(editor.transform) }; editor.saved = editor.heights.map((line) => line.slice()); editor.savedTransform = cloneTerrainTransform(editor.transform); editor.changed = false; editor.undo = []; editor.redo = [];
    state.scenes.set(row.id, editor.sceneData); updateTerrainHistoryUi(); await loadTerrains();
    $("terrainEditorStatus").textContent = copy(`地形已保存 · 修订 ${row.revision}`, `Terrain saved · Revision ${row.revision}`);
  } catch (error) { $("terrainEditorStatus").textContent = error.message; }
  finally { $("terrainSave").disabled = false; }
}

function revertTerrain() {
  const editor = state.editor; if (!editor?.saved) return;
  editor.heights = editor.saved.map((row) => row.slice()); applyTerrainOverallTransform(editor.savedTransform); editor.undo = []; editor.redo = []; editor.changed = false; updateEditorGeometry(); updateTerrainHistoryUi();
  $("terrainEditorStatus").textContent = copy("已恢复到最近保存状态。", "Reverted to the most recently saved state.");
}

function closeEditor() {
  if (state.editor?.changed && !confirm(copy("当前地形有未保存的修改，仍要关闭？", "This terrain has unsaved edits. Close anyway?"))) return;
  $("terrainEditorModal").classList.add("hidden");
}

function resizeEditor() {
  const editor = state.editor; if (!editor) return;
  const rect = editor.renderer.domElement.getBoundingClientRect(); editor.renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false); editor.camera.aspect = rect.width / Math.max(1, rect.height); editor.camera.updateProjectionMatrix();
}

function populateGaitTerrainSelect() {
  const select = $("gaitIKTerrain"); if (!select) return;
  const previous = select.value || state.gaitTerrainId;
  select.innerHTML = `<option value="">${copy("基准平面", "Reference Plane")}</option>${state.terrains.map((terrain) => `<option value="${terrain.id}">${escapeHtml(terrain.name)} · ${terrain.size_x}×${terrain.size_z}m</option>`).join("")}`;
  if (state.terrains.some((terrain) => terrain.id === previous)) select.value = previous;
}

async function applySelectedGaitTerrain() {
  const id = $("gaitIKTerrain").value;
  if (!id) { removeGaitTerrain(); return; }
  try {
    const sceneData = await terrainScene(id);
    state.gaitTerrainId = id; $("gaitIKTerrainState").textContent = `TERRAIN · REV ${sceneData.revision}`;
    $("gaitIKIncludeTerrain").disabled = false;
    syncPlayerTerrain(state.gaitPlayer, sceneData);
    updateGaitStartFromInputs();
  } catch (error) { $("gaitIKStatus").textContent = error.message; }
}

function removeGaitTerrain() {
  state.gaitTerrainId = ""; if ($("gaitIKTerrain")) $("gaitIKTerrain").value = "";
  if ($("gaitIKTerrainState")) $("gaitIKTerrainState").textContent = "PLANE Y=0";
  if ($("gaitIKIncludeTerrain")) { $("gaitIKIncludeTerrain").checked = false; $("gaitIKIncludeTerrain").disabled = true; }
  syncPlayerTerrain(state.gaitPlayer, null); updateGaitStartFromInputs();
}

function toggleGaitStartPanel() {
  const panel = $("gaitIKStartPanel"), hidden = panel.classList.toggle("hidden");
  if (state.gaitPlayer?.terrainStartControl) {
    state.gaitPlayer.terrainStartControl.enabled = !hidden;
    state.gaitPlayer.terrainStartControl.getHelper().visible = !hidden && !state.gaitPlayer.terrainSolved;
    state.gaitPlayer.terrainStartProxy.visible = !hidden && !state.gaitPlayer.terrainSolved;
  }
}

function setupGaitTransform(player) {
  if (!player || player.terrainStartControl) return;
  const proxy = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.RingGeometry(.2, .23, 48), new THREE.MeshBasicMaterial({ color: 0xffc45e, side: THREE.DoubleSide, depthTest: false })); ring.rotation.x = -Math.PI / 2;
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), .55, 0xffc45e, .12, .06); proxy.add(ring, arrow); player.scene.add(proxy);
  const control = new TransformControls(player.camera, player.canvas); player.scene.add(control.getHelper()); control.attach(proxy); control.setMode("translate"); control.showY = false;
  control.addEventListener("dragging-changed", (event) => { player.controls.enabled = !event.value; });
  control.addEventListener("objectChange", () => {
    if (player.terrainStartUpdating) return;
    proxy.rotation.x = 0; proxy.rotation.z = 0; proxy.position.y = gaitGroundHeight(proxy.position.x, proxy.position.z);
    $("gaitIKStartX").value = proxy.position.x.toFixed(3); $("gaitIKStartZ").value = proxy.position.z.toFixed(3); $("gaitIKStartYaw").value = THREE.MathUtils.radToDeg(proxy.rotation.y).toFixed(1);
    applyGaitStartToCharacter(); updateGaitGroundLabel(proxy.position.y);
  });
  player.terrainStartProxy = proxy; player.terrainStartControl = control; player.terrainSolved = false;
  const visible = !$("gaitIKStartPanel")?.classList.contains("hidden"); control.getHelper().visible = visible; proxy.visible = visible;
}

function gaitGroundHeight(x, z) {
  if (!state.gaitTerrainId) return 0;
  const sceneData = state.scenes.get(state.gaitTerrainId), heights = sceneData?.heights;
  if (!heights?.length) return 0;
  const transform = terrainTransform(sceneData.transform), dx = x - transform.translation[0], dz = z - transform.translation[2], yaw = THREE.MathUtils.degToRad(transform.yaw_degrees), c = Math.cos(yaw), s = Math.sin(yaw);
  const localX = (c * dx - s * dz) / transform.uniform_scale, localZ = (s * dx + c * dz) / transform.uniform_scale;
  const rows = heights.length, columns = heights[0].length, gx = (localX / sceneData.size_x + .5) * (columns - 1), gz = (localZ / sceneData.size_z + .5) * (rows - 1);
  if (gx < 0 || gx > columns - 1 || gz < 0 || gz > rows - 1) return 0;
  const x0 = Math.floor(gx), z0 = Math.floor(gz), x1 = Math.min(columns - 1, x0 + 1), z1 = Math.min(rows - 1, z0 + 1), tx = gx - x0, tz = gz - z0;
  const h0 = heights[z0][x0] * (1 - tx) + heights[z0][x1] * tx, h1 = heights[z1][x0] * (1 - tx) + heights[z1][x1] * tx;
  return transform.translation[1] + (h0 * (1 - tz) + h1 * tz) * transform.uniform_scale;
}

function updateGaitGroundLabel(height) {
  if ($("gaitIKStartGround")) $("gaitIKStartGround").textContent = `GROUND Y = ${Number(height).toFixed(3)} m · AUTO SNAP`;
}

function updateGaitStartFromInputs() {
  const player = state.gaitPlayer, x = +$("gaitIKStartX")?.value || 0, z = +$("gaitIKStartZ")?.value || 0, yaw = +$("gaitIKStartYaw")?.value || 0, y = gaitGroundHeight(x, z);
  updateGaitGroundLabel(y);
  if (!player?.terrainStartProxy) return;
  player.terrainStartUpdating = true; player.terrainStartProxy.position.set(x, y, z); player.terrainStartProxy.rotation.set(0, THREE.MathUtils.degToRad(yaw), 0); player.terrainStartUpdating = false;
  applyGaitStartToCharacter();
}

function applyGaitStartToCharacter() {
  const player = state.gaitPlayer; if (!player || player.terrainSolved) return;
  const x = +$("gaitIKStartX")?.value || 0, z = +$("gaitIKStartZ")?.value || 0, yaw = THREE.MathUtils.degToRad(+$("gaitIKStartYaw")?.value || 0), y = gaitGroundHeight(x, z);
  if (player.mesh) { player.mesh.position.set(x, y, z); player.mesh.rotation.set(0, yaw, 0); }
  else if (player.bones?.[0]) { player.bones[0].position.x += x; player.bones[0].position.y += y; player.bones[0].position.z += z; }
}

function setGaitTransformMode(mode) {
  const control = state.gaitPlayer?.terrainStartControl; if (!control) return;
  control.setMode(mode); control.showX = mode === "translate"; control.showY = mode === "rotate"; control.showZ = mode === "translate";
  $("gaitIKStartTranslate").classList.toggle("active", mode === "translate"); $("gaitIKStartRotate").classList.toggle("active", mode === "rotate");
}

function resetGaitStart() {
  $("gaitIKStartX").value = "0"; $("gaitIKStartZ").value = "0"; $("gaitIKStartYaw").value = "0"; updateGaitStartFromInputs();
}

function gaitPayload() {
  const x = +$("gaitIKStartX")?.value || 0, z = +$("gaitIKStartZ")?.value || 0;
  return { terrain_id: state.gaitTerrainId, initial_translation: [x, gaitGroundHeight(x, z), z], initial_yaw_degrees: +$("gaitIKStartYaw")?.value || 0, terrain_smoothing_frames: +$("gaitIKTerrainSmooth")?.value || 5, max_slope_degrees: +$("gaitIKMaxSlope")?.value || 55 };
}

function includeTerrain() { return Boolean(state.gaitTerrainId && $("gaitIKIncludeTerrain")?.checked); }

function configureGaitPlayer(player, options = {}) {
  state.gaitPlayer = player; if (!player) return;
  setupGaitTransform(player); player.terrainSolved = Boolean(options.solved);
  const visible = !player.terrainSolved && !$("gaitIKStartPanel")?.classList.contains("hidden");
  player.terrainStartControl.getHelper().visible = visible; player.terrainStartProxy.visible = visible;
  if (player.terrainSolved) {
    if (player.mesh) { player.mesh.position.set(0, 0, 0); player.mesh.rotation.set(0, 0, 0); }
  } else updateGaitStartFromInputs();
}

async function prepareGait(player) {
  state.gaitPlayer = player; await loadTerrains(); setupGaitTransform(player);
  if (state.gaitTerrainId) syncPlayerTerrain(player, await terrainScene(state.gaitTerrainId));
  configureGaitPlayer(player, { solved: false });
}

function disposeTerrainObject(player) {
  if (!player?.terrainObject) return;
  player.scene.remove(player.terrainObject); player.terrainObject.traverse((object) => { object.geometry?.dispose?.(); object.material?.dispose?.(); }); player.terrainObject = null;
}

function syncPlayerTerrain(player, sceneData, options = {}) {
  if (!player) return;
  disposeTerrainObject(player);
  if (sceneData?.heights?.length) {
    player.terrainObject = terrainGroup(sceneData, false); player.scene.add(player.terrainObject);
  }
  if (options.final) {
    state.finalPlayer = player; const button = $("finalTerrainToggle"), present = Boolean(player.terrainObject);
    button?.classList.toggle("hidden", !present); if (button) button.textContent = copy("隐藏地形", "Hide Terrain");
  }
}

function toggleFinalTerrain() {
  const object = state.finalPlayer?.terrainObject; if (!object) return;
  object.visible = !object.visible; $("finalTerrainToggle").textContent = object.visible ? copy("隐藏地形", "Hide Terrain") : copy("显示地形", "Show Terrain");
}

window.addEventListener("resize", resizeEditor);
installShell();
loadTerrains();

window.TerrainWorkspace = Object.freeze({
  list: () => state.terrains.slice(),
  load: loadTerrains,
  scene: terrainScene,
  gaitPayload,
  includeTerrain,
  prepareGait,
  configureGaitPlayer,
  syncPlayerTerrain,
});
