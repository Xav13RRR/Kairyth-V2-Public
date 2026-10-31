const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

export function uniformScaleMultiplier(startScale, currentScale) {
  let factor = 1;
  let strongestChange = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const start = Number(startScale?.[axis]);
    const current = Number(currentScale?.[axis]);
    if (!Number.isFinite(start) || !Number.isFinite(current) || Math.abs(start) <= 1e-8) continue;
    const ratio = current / start;
    if (!Number.isFinite(ratio) || ratio <= 0) continue;
    const change = Math.abs(Math.log(ratio));
    if (change > strongestChange) {
      strongestChange = change;
      factor = ratio;
    }
  }
  return clamp(factor, 0.001, 1000);
}

export function createFinalSceneTools(options) {
  const data = {
    mode: "timeline",
    sceneData: null,
    objects: [],
    constraints: [],
    visuals: new Map(),
    targets: new Map(),
    assetCache: new Map(),
    terrains: [],
    selectedObjectId: "",
    selectedConstraintId: "",
    selectedConstraintKeyFrame: null,
    selectedIKControl: null,
    selectingConstraintIK: false,
    runtime: null,
    gizmo: null,
    raycaster: null,
    pointer: null,
    pointerHandler: null,
    uniformScaleDrag: null,
  };
  const $ = (id) => document.getElementById(id);
  const tr = (zh, en) => document.documentElement.lang.toLowerCase().startsWith("en") ? en : zh;
  const uid = (prefix) => prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
  const number = (id, fallback = 0) => {
    const value = Number($(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  };
  const frameCount = () => Math.max(1, Number(data.sceneData?.num_frames || 1));
  const selectedObject = () => data.objects.find((item) => item.id === data.selectedObjectId) || null;
  const selectedConstraint = () => data.constraints.find((item) => item.id === data.selectedConstraintId) || null;
  const status = (zh, en) => options.setStatus?.(tr(zh, en));

  function ensureUi() {
    if ($("sceneImportModeBtn")) return;
    const selector = $("editorModeSelector");
    const panels = $("editorModePanels");
    const viewport = $("viewport");
    if (!selector || !panels || !viewport) return;
    selector.insertAdjacentHTML("beforeend", [
      '<button id="sceneImportModeBtn" data-editor-mode="scene"></button>',
      '<button id="ikLockModeBtn" data-editor-mode="iklock"></button>',
    ].join(""));
    panels.insertAdjacentHTML("beforeend", [
      '<div id="sceneImportPanel" class="editor-mode-panel final-scene-panel hidden">',
      '  <h2 id="scenePanelTitle"></h2>',
      '  <label><span id="sceneAssetKindLabel"></span><select id="sceneAssetKind"><option value="static_mesh"></option><option value="terrain"></option></select></label>',
      '  <label><span id="sceneAssetLabel"></span><select id="sceneAssetSelect"></select></label>',
      '  <button id="sceneAddAssetBtn" class="primary" type="button"></button>',
      '  <div class="final-scene-divider"></div>',
      '  <label><span id="sceneObjectNameLabel"></span><input id="sceneObjectName" maxlength="80"/></label>',
      '  <div class="mode-row"><button id="sceneMoveBtn" class="mode-btn active" type="button"></button><button id="sceneRotateBtn" class="mode-btn" type="button"></button><button id="sceneScaleBtn" class="mode-btn" type="button"></button></div>',
      '  <label><span id="sceneParentTypeLabel"></span><select id="sceneParentType"><option value="world"></option><option value="bone"></option><option value="ik_lock"></option></select></label>',
      '  <label><span id="sceneParentLabel"></span><select id="sceneParentSelect"></select></label>',
      '  <label class="check"><input id="sceneDefaultVisible" type="checkbox" checked/><span id="sceneDefaultVisibleLabel"></span></label>',
      '  <div class="row"><label><span id="sceneVisibilityStartLabel"></span><input id="sceneVisibilityStart" type="number" min="0" value="0"/></label><label><span id="sceneVisibilityEndLabel"></span><input id="sceneVisibilityEnd" type="number" min="1" value="1"/></label></div>',
      '  <label><span id="sceneVisibilityActionLabel"></span><select id="sceneVisibilityAction"><option value="false"></option><option value="true"></option></select></label>',
      '  <div class="mode-row"><button id="sceneAddVisibilityBtn" type="button"></button><button id="sceneClearVisibilityBtn" type="button"></button></div>',
      '  <div id="sceneVisibilityList" class="edit-list compact"></div>',
      '  <div class="mode-row"><button id="sceneSaveObjectBtn" class="primary" type="button"></button><button id="sceneDeleteObjectBtn" class="editor-danger" type="button"></button></div>',
      '  <button id="sceneExitBtn" type="button"></button>',
      '</div>',
      '<div id="ikLockPanel" class="editor-mode-panel final-scene-panel hidden">',
      '  <h2 id="ikLockPanelTitle"></h2>',
      '  <label><span id="ikLockNameLabel"></span><input id="ikLockName" maxlength="80"/></label>',
      '  <label><span id="ikLockSolverLabel"></span><select id="ikLockSolver"><option value="two_bone">Two-Bone IK</option><option value="pbik">PBIK</option><option value="jacobian_dls">Jacobian DLS</option><option value="fabrik">FABRIK</option></select></label>',
      '  <label><span id="ikLockEffectorLabel"></span><select id="ikLockEffector"><option value="LeftFoot">Left Foot</option><option value="RightFoot">Right Foot</option><option value="LeftHand">Left Hand</option><option value="RightHand">Right Hand</option><option value="Head">Head</option></select></label>',
      '  <p id="ikLockControlInfo" class="status"></p>',
      '  <div class="row"><label><span id="ikLockStartLabel"></span><input id="ikLockStart" type="number" min="0" value="0" readonly disabled/></label><label><span id="ikLockEndLabel"></span><input id="ikLockEnd" type="number" min="0" value="0"/></label></div>',
      '  <div class="row"><label><span id="ikLockBlendInLabel"></span><input id="ikLockBlendIn" type="number" min="0" max="240" value="5"/></label><label><span id="ikLockBlendOutLabel"></span><input id="ikLockBlendOut" type="number" min="0" max="240" value="5"/></label></div>',
      '  <label><span id="ikLockInterpolationLabel"></span><select id="ikLockInterpolation"><option value="cubic"></option><option value="linear"></option><option value="step"></option></select></label>',
      '  <label><span id="ikLockRootLabel"></span><select id="ikLockRoot"><option value="pre_pull"></option><option value="pin"></option><option value="foot_priority"></option><option value="free"></option></select></label>',
      '  <div class="row final-vector-row"><label>X<input id="ikLockX" type="number" step="0.001"/></label><label>Y<input id="ikLockY" type="number" step="0.001"/></label><label>Z<input id="ikLockZ" type="number" step="0.001"/></label></div>',
      '  <div class="mode-row"><button id="ikLockMoveBtn" class="mode-btn active" type="button"></button><button id="ikLockRotateBtn" class="mode-btn" type="button"></button></div>',
      '  <button id="ikLockUpsertBtn" class="primary final-wide-action" type="button"></button>',
      '  <div class="mode-row"><button id="ikLockAddKeyBtn" type="button"></button><button id="ikLockDeleteKeyBtn" type="button"></button></div>',
      '  <div id="ikLockKeyList" class="edit-list compact"></div>',
      '  <div class="mode-row"><button id="ikLockApplyBtn" class="primary" type="button"></button><button id="ikLockDeleteBtn" class="editor-danger" type="button"></button></div>',
      '  <button id="ikLockExitBtn" type="button"></button>',
      '  <p id="ikLockHint" class="status"></p>',
      '</div>',
    ].join(""));
    viewport.insertAdjacentHTML("beforeend", [
      '<aside id="finalSceneOutliner" class="final-scene-outliner hidden">',
      '  <div class="final-outliner-head"><strong id="finalOutlinerTitle"></strong><button id="finalOutlinerCollapse" type="button">-</button></div>',
      '  <div id="finalOutlinerBody"></div>',
      '</aside>',
    ].join(""));
    bindUi();
    localize();
  }

  function localize() {
    const pairs = {
      sceneImportModeBtn: ["\u573a\u666f\u5bfc\u5165", "Scene Import"],
      ikLockModeBtn: ["IK \u9501\u5b9a\u70b9", "IK Lock Point"],
      scenePanelTitle: ["\u573a\u666f\u5bfc\u5165", "Scene Import"],
      sceneAssetKindLabel: ["\u8d44\u4ea7\u7c7b\u578b", "Asset Type"],
      sceneAssetLabel: ["\u8d44\u4ea7", "Asset"],
      sceneAddAssetBtn: ["\u6dfb\u52a0\u5230\u573a\u666f", "Add to Scene"],
      sceneObjectNameLabel: ["\u5b9e\u4f8b\u540d\u79f0", "Instance Name"],
      sceneMoveBtn: ["\u5e73\u79fb", "Move"],
      sceneRotateBtn: ["\u65cb\u8f6c", "Rotate"],
      sceneScaleBtn: ["\u6574\u4f53\u7f29\u653e", "Uniform Scale"],
      sceneParentTypeLabel: ["\u9644\u7740\u7c7b\u578b", "Parent Type"],
      sceneParentLabel: ["\u7236\u7ea7", "Parent"],
      sceneDefaultVisibleLabel: ["\u9ed8\u8ba4\u663e\u793a", "Visible by Default"],
      sceneVisibilityStartLabel: ["\u5f00\u59cb\u5e27", "Start Frame"],
      sceneVisibilityEndLabel: ["\u7ed3\u675f\u5e27\uff08\u4e0d\u5305\u542b\uff09", "End Frame (exclusive)"],
      sceneVisibilityActionLabel: ["\u533a\u95f4\u72b6\u6001", "Range State"],
      sceneAddVisibilityBtn: ["\u6dfb\u52a0\u53ef\u89c1\u533a\u95f4", "Add Visibility Range"],
      sceneClearVisibilityBtn: ["\u6e05\u7a7a\u533a\u95f4", "Clear Ranges"],
      sceneSaveObjectBtn: ["\u4fdd\u5b58\u5b9e\u4f8b", "Save Instance"],
      sceneDeleteObjectBtn: ["\u5220\u9664\u5b9e\u4f8b", "Delete Instance"],
      sceneExitBtn: ["\u9000\u51fa\u573a\u666f\u5bfc\u5165", "Exit Scene Import"],
      ikLockPanelTitle: ["IK \u9501\u5b9a\u70b9", "IK Lock Point"],
      ikLockNameLabel: ["\u9501\u5b9a\u540d\u79f0", "Lock Name"],
      ikLockSolverLabel: ["IK \u89e3\u7b97\u5668", "IK Solver"],
      ikLockEffectorLabel: ["IK \u8282\u70b9", "IK Effector"],
      ikLockStartLabel: ["\u5f00\u59cb\u5e27", "Start Frame"],
      ikLockEndLabel: ["\u7ed3\u675f\u5e27", "End Frame"],
      ikLockBlendInLabel: ["\u524d\u5e73\u6ed1\uff08\u5e27\uff09", "Blend In (frames)"],
      ikLockBlendOutLabel: ["\u540e\u5e73\u6ed1\uff08\u5e27\uff09", "Blend Out (frames)"],
      ikLockInterpolationLabel: ["\u63a7\u5236\u70b9\u66f2\u7ebf", "Target Curve"],
      ikLockRootLabel: ["\u6839\u90e8\u884c\u4e3a", "Root Behavior"],
      ikLockMoveBtn: ["\u5e73\u79fb", "Move"],
      ikLockRotateBtn: ["\u65cb\u8f6c", "Rotate"],
      ikLockUpsertBtn: ["\u5728\u5f53\u524d\u4f4d\u7f6e\u65b0\u5efa / \u66f4\u65b0\u9501\u5b9a\u70b9", "Create / Update Lock at Current Position"],
      ikLockAddKeyBtn: ["\u5728\u5f53\u524d\u5e27\u8bbe\u7f6e\u63a7\u5236\u70b9", "Set Key at Current Frame"],
      ikLockDeleteKeyBtn: ["\u5220\u9664\u5f53\u524d\u5e27\u63a7\u5236\u70b9", "Delete Current Key"],
      ikLockApplyBtn: ["\u5e94\u7528 IK \u9501\u5b9a", "Apply IK Locks"],
      ikLockDeleteBtn: ["\u5220\u9664\u9501\u5b9a\u70b9", "Delete Lock"],
      ikLockExitBtn: ["\u9000\u51fa IK \u9501\u5b9a", "Exit IK Lock"],
      ikLockHint: ["\u59cb\u7ec8\u53ef\u76f4\u63a5\u70b9\u9009 IK \u63a7\u5236\u5668\u5e76\u5e73\u79fb / \u65cb\u8f6c\uff1b\u7f16\u53f7\u63a7\u5236\u70b9\u7528\u4e8e\u7f16\u8f91\u9501\u5b9a\u66f2\u7ebf\u3002", "IK controls are always selectable for translate/rotate; numbered points edit the lock curve."],
      finalOutlinerTitle: ["\u573a\u666f\u5927\u7eb2", "Scene Outliner"],
    };
    Object.entries(pairs).forEach(([id, values]) => {
      if ($(id)) $(id).textContent = tr(values[0], values[1]);
    });
    const assetKind = $("sceneAssetKind");
    if (assetKind) {
      assetKind.options[0].textContent = tr("\u9759\u6001\u7f51\u683c\u4f53", "Static Mesh");
      assetKind.options[1].textContent = tr("\u5730\u5f62", "Terrain");
    }
    const parent = $("sceneParentType");
    if (parent) {
      parent.options[0].textContent = tr("\u4e16\u754c", "World");
      parent.options[1].textContent = tr("\u9aa8\u9abc", "Bone");
      parent.options[2].textContent = tr("IK \u9501\u5b9a\u70b9", "IK Lock Point");
    }
    const range = $("sceneVisibilityAction");
    if (range) {
      range.options[0].textContent = tr("\u9690\u85cf", "Hidden");
      range.options[1].textContent = tr("\u663e\u793a", "Visible");
    }
    const interpolation = $("ikLockInterpolation");
    if (interpolation) {
      interpolation.options[0].textContent = tr("\u5e73\u6ed1\u66f2\u7ebf", "Cubic");
      interpolation.options[1].textContent = tr("\u7ebf\u6027", "Linear");
      interpolation.options[2].textContent = tr("\u9636\u8dc3", "Step");
    }
    const root = $("ikLockRoot");
    if (root) {
      root.options[0].textContent = tr("\u9884\u62c9\u52a8\u9aa8\u76c6", "Pre-pull Pelvis");
      root.options[1].textContent = tr("\u9501\u5b9a\u9aa8\u76c6", "Pin Pelvis");
      root.options[2].textContent = tr("\u811a\u90e8\u4f18\u5148", "Foot Priority");
      root.options[3].textContent = tr("\u81ea\u7531\u6839\u90e8", "Free Root");
    }
    refreshEffectorOptions();
    refreshAssetOptions();
    renderOutliner();
    renderScenePanel();
    renderConstraintPanel();
    renderIKControlInfo();
    if (data.mode === "iklock") {
      renderTimeline();
      status("IK \u9501\u5b9a\u6a21\u5f0f\uff1a\u76f4\u63a5\u70b9\u9009\u5e76\u62d6\u52a8 IK \u63a7\u5236\u5668\u6216\u7f16\u53f7\u66f2\u7ebf\u70b9\u3002", "IK Lock mode: select and drag IK controls or numbered curve points directly.");
    }
  }

  function bindUi() {
    $("sceneImportModeBtn").onclick = () => enterMode("scene").catch(showError);
    $("ikLockModeBtn").onclick = () => enterMode("iklock").catch(showError);
    $("sceneExitBtn").onclick = exitMode;
    $("ikLockExitBtn").onclick = exitMode;
    $("sceneAssetKind").onchange = refreshAssetOptions;
    $("sceneAddAssetBtn").onclick = () => addSelectedAsset().catch(showError);
    $("sceneMoveBtn").onclick = () => setGizmoMode("translate");
    $("sceneRotateBtn").onclick = () => setGizmoMode("rotate");
    $("sceneScaleBtn").onclick = () => setGizmoMode("scale");
    $("sceneParentType").onchange = () => { refreshParentOptions(); applySceneInputs(false); };
    $("sceneParentSelect").onchange = () => applySceneInputs(false);
    $("sceneSaveObjectBtn").onclick = () => saveSelectedObject().catch(showError);
    $("sceneDeleteObjectBtn").onclick = () => deleteSelectedObject().catch(showError);
    $("sceneAddVisibilityBtn").onclick = addVisibilityRange;
    $("sceneClearVisibilityBtn").onclick = () => {
      const item = selectedObject();
      if (!item) return;
      item.visibility.ranges = [];
      renderScenePanel();
      renderTimeline();
      updateFrame(options.state.frame || 0);
    };
    $("ikLockMoveBtn").onclick = () => setGizmoMode("translate");
    $("ikLockRotateBtn").onclick = () => setGizmoMode("rotate");
    $("ikLockUpsertBtn").onclick = upsertConstraint;
    $("ikLockAddKeyBtn").onclick = upsertConstraintKey;
    $("ikLockDeleteKeyBtn").onclick = deleteConstraintKey;
    $("ikLockApplyBtn").onclick = () => saveConstraints(true).catch(showError);
    $("ikLockDeleteBtn").onclick = () => deleteSelectedConstraint().catch(showError);
    $("ikLockSolver").onchange = () => changeIKSolver().catch(showError);
    $("ikLockEffector").onchange = () => {
      const jointName = $("ikLockEffector").value;
      const selected = options.selectIKEffector?.($("ikLockSolver").value, jointName);
      if (selected === false) {
        status("\u5f53\u524d\u89e3\u7b97\u5668\u4e0d\u5305\u542b\u8be5 IK \u8282\u70b9\u3002", "The current solver does not contain that IK effector.");
      }
    };
    $("finalOutlinerCollapse").onclick = () => $("finalSceneOutliner").classList.toggle("collapsed");
    ["sceneObjectName", "sceneDefaultVisible"].forEach((id) => $(id).onchange = () => applySceneInputs(false));
    ["ikLockX", "ikLockY", "ikLockZ"].forEach((id) => $(id).oninput = syncTargetFromInputs);
    window.addEventListener("motionbuilder:languagechange", localize);
  }

  function showError(error) {
    options.setStatus?.(error?.message || String(error));
  }

  function attachRuntime() {
    const runtime = options.getRuntime?.();
    const THREE = options.getThree?.();
    const TransformControls = options.getTransformControls?.();
    if (!runtime || !THREE || !TransformControls || data.runtime === runtime) return;
    detachRuntime();
    data.runtime = runtime;
    data.raycaster = new THREE.Raycaster();
    data.pointer = new THREE.Vector2();
    data.gizmo = new TransformControls(runtime.camera, runtime.canvas);
    data.gizmo.setMode("translate");
    data.gizmo.setSpace("world");
    runtime.scene.add(data.gizmo.getHelper());
    data.gizmo.getHelper().visible = false;
    data.gizmo.addEventListener("dragging-changed", (event) => {
      runtime.controls.enabled = !event.value;
      if (event.value) {
        if (data.mode === "scene" && data.gizmo.mode === "scale") beginUniformScaleDrag();
        return;
      }
      if (data.mode === "scene") {
        if (data.gizmo.mode === "scale") applyUniformScaleDrag();
        syncSelectedObjectFromGizmo();
      }
      if (data.mode === "iklock") syncSelectedTargetFromGizmo(true);
      data.uniformScaleDrag = null;
    });
    data.gizmo.addEventListener("objectChange", () => {
      if (data.mode === "scene") {
        if (data.gizmo.mode === "scale") applyUniformScaleDrag();
        syncSelectedObjectFromGizmo(false);
      }
      if (data.mode === "iklock") syncSelectedTargetFromGizmo(true);
    });
    data.pointerHandler = selectFromViewport;
    runtime.canvas.addEventListener("pointerdown", data.pointerHandler);
  }

  function detachRuntime() {
    if (!data.runtime) return;
    if (data.pointerHandler) data.runtime.canvas.removeEventListener("pointerdown", data.pointerHandler);
    data.gizmo?.detach();
    if (data.gizmo) {
      data.runtime.scene.remove(data.gizmo.getHelper());
      data.gizmo.dispose();
    }
    clearVisuals();
    data.runtime = null;
    data.gizmo = null;
  }

  function clearVisuals() {
    for (const visual of data.visuals.values()) disposeObject3D(visual);
    for (const view of data.targets.values()) {
      disposeObject3D(view.marker);
      disposeObject3D(view.path);
      disposeObject3D(view.keyRoot);
    }
    data.visuals.clear();
    data.targets.clear();
  }

  function disposeObject3D(object) {
    if (!object) return;
    object.removeFromParent();
    object.traverse?.((child) => {
      child.geometry?.dispose?.();
      if (Array.isArray(child.material)) child.material.forEach((material) => { material.map?.dispose?.(); material.dispose?.(); });
      else { child.material?.map?.dispose?.(); child.material?.dispose?.(); }
    });
  }

  async function load(sceneData) {
    ensureUi();
    attachRuntime();
    data.gizmo?.detach();
    data.sceneData = sceneData || null;
    data.objects = structuredClone(sceneData?.scene_objects || []);
    data.constraints = structuredClone(sceneData?.ik_constraints || []);
    data.selectedObjectId = data.objects.some((item) => item.id === data.selectedObjectId) ? data.selectedObjectId : "";
    data.selectedConstraintId = data.constraints.some((item) => item.id === data.selectedConstraintId) ? data.selectedConstraintId : "";
    clearVisuals();
    await refreshCatalogs();
    await Promise.all(data.objects.map((item) => createObjectVisual(item).catch(showError)));
    data.constraints.forEach(createConstraintView);
    if (data.mode === "scene" && data.selectedObjectId) data.gizmo?.attach(data.visuals.get(data.selectedObjectId));
    if (data.mode === "iklock" && data.selectedConstraintId) attachSelectedConstraintGizmo();
    if (data.mode !== "timeline") {
      if (data.runtime?.gizmo?.getHelper()) data.runtime.gizmo.getHelper().visible = false;
      if (data.gizmo?.getHelper()) data.gizmo.getHelper().visible = true;
    }
    $("finalSceneOutliner")?.classList.toggle("hidden", !sceneData);
    renderOutliner();
    renderScenePanel();
    renderConstraintPanel();
    updateFrame(options.state.frame || 0);
  }

  async function refreshCatalogs() {
    try { data.terrains = await options.api("/api/terrains"); }
    catch { data.terrains = []; }
    refreshAssetOptions();
  }

  function refreshAssetOptions() {
    const select = $("sceneAssetSelect");
    if (!select) return;
    const kind = $("sceneAssetKind")?.value || "static_mesh";
    const rows = kind === "terrain" ? data.terrains : (options.getStaticAssets?.() || []);
    const current = select.value;
    select.innerHTML = rows.length
      ? rows.map((item) => '<option value="' + options.escapeHtml(item.id) + '">' + options.escapeHtml(item.name || item.id) + '</option>').join("")
      : '<option value="">' + tr("\u6682\u65e0\u53ef\u7528\u8d44\u4ea7", "No assets available") + '</option>';
    if (rows.some((item) => item.id === current)) select.value = current;
  }

  async function ensureAsset(kind, id) {
    const key = kind + ":" + id;
    if (data.assetCache.has(key)) return data.assetCache.get(key);
    const prefix = kind === "terrain" ? "/api/terrains" : "/api/characters";
    const scene = await options.api(prefix + "/" + encodeURIComponent(id) + "/scene");
    data.assetCache.set(key, scene);
    return scene;
  }

  function makeAssetMesh(sceneData, kind, name) {
    const THREE = options.getThree();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute((sceneData.vertices || []).flat(), 3));
    geometry.setIndex((sceneData.faces || []).flat());
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: kind === "terrain" ? 0x68795c : 0xb6bdc7,
      roughness: kind === "terrain" ? 0.95 : 0.58,
      metalness: kind === "terrain" ? 0.0 : 0.08,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name || "SceneAsset";
    mesh.receiveShadow = kind === "terrain";
    return mesh;
  }

  async function createObjectVisual(item) {
    if (!data.runtime) return;
    const THREE = options.getThree();
    const source = await ensureAsset(item.asset_kind, item.source_asset_id);
    const group = new THREE.Group();
    group.name = item.name;
    group.userData.finalSceneObjectId = item.id;
    const mesh = makeAssetMesh(source, item.asset_kind, item.name);
    mesh.userData.finalSceneObjectId = item.id;
    group.add(mesh);
    data.runtime.scene.add(group);
    data.visuals.set(item.id, group);
    updateObjectWorld(item, options.state.frame || 0, true);
  }

  function defaultTransformForAsset(kind, source) {
    if (kind !== "terrain") return { translation: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: [1, 1, 1] };
    const THREE = options.getThree();
    const transform = source?.transform || {};
    const quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, THREE.MathUtils.degToRad(Number(transform.yaw_degrees || 0)), 0));
    const uniform = Number(transform.uniform_scale || 1);
    return {
      translation: (transform.translation || [0, 0, 0]).slice(0, 3),
      quaternion: quaternion.toArray(),
      scale: [uniform, uniform, uniform],
    };
  }

  async function addSelectedAsset() {
    if (!options.state.composition) throw new Error(tr("\u8bf7\u5148\u5411\u65f6\u95f4\u8f74\u6dfb\u52a0\u52a8\u753b", "Add an animation to the timeline first"));
    const kind = $("sceneAssetKind").value;
    const id = $("sceneAssetSelect").value;
    if (!id) return;
    const source = await ensureAsset(kind, id);
    const sourceRow = kind === "terrain" ? data.terrains.find((item) => item.id === id) : (options.getStaticAssets?.() || []).find((item) => item.id === id);
    const item = {
      id: uid("scene"),
      name: sourceRow?.name || tr("\u573a\u666f\u8d44\u4ea7", "Scene Asset"),
      asset_kind: kind,
      source_asset_id: id,
      asset_revision: String(sourceRow?.revision || sourceRow?.content_hash || ""),
      parent_type: "world",
      parent_id: "",
      parent_joint: -1,
      transform: defaultTransformForAsset(kind, source),
      visibility: { default: true, ranges: [] },
      editor_hidden: false,
    };
    data.objects.push(item);
    await createObjectVisual(item);
    data.selectedObjectId = item.id;
    selectObject(item.id);
    await saveObjects();
    status("\u5df2\u6dfb\u52a0\u573a\u666f\u8d44\u4ea7\uff0c\u539f\u8d44\u4ea7\u672a\u88ab\u4fee\u6539\u3002", "Scene asset added; the source asset was not modified.");
  }

  function parentMatrix(item, frame) {
    const THREE = options.getThree();
    const matrix = new THREE.Matrix4();
    if (item.parent_type === "bone") {
      const bone = data.runtime?.bones?.[Number(item.parent_joint)];
      if (bone) {
        bone.updateWorldMatrix(true, false);
        return bone.matrixWorld.clone();
      }
    }
    if (item.parent_type === "ik_lock") {
      const constraint = data.constraints.find((entry) => entry.id === item.parent_id);
      if (constraint) {
        const sampled = sampleConstraint(constraint, frame);
        return matrix.compose(sampled.position, sampled.quaternion, new THREE.Vector3(1, 1, 1));
      }
    }
    return matrix.identity();
  }

  function localMatrix(item) {
    const THREE = options.getThree();
    const transform = item.transform || {};
    return new THREE.Matrix4().compose(
      new THREE.Vector3().fromArray(transform.translation || [0, 0, 0]),
      new THREE.Quaternion().fromArray(transform.quaternion || [0, 0, 0, 1]).normalize(),
      new THREE.Vector3().fromArray(transform.scale || [1, 1, 1]),
    );
  }

  function isObjectVisible(item, frame) {
    let visible = item.visibility?.default !== false;
    for (const range of item.visibility?.ranges || []) {
      if (frame >= Number(range.start_frame || 0) && frame < Number(range.end_frame || 0)) visible = range.visible !== false;
    }
    return visible && !item.editor_hidden;
  }

  function updateObjectWorld(item, frame, force = false) {
    const visual = data.visuals.get(item.id);
    if (!visual) return;
    visual.visible = isObjectVisible(item, frame);
    if (!force && data.gizmo?.dragging && item.id === data.selectedObjectId) return;
    const world = parentMatrix(item, frame).multiply(localMatrix(item));
    world.decompose(visual.position, visual.quaternion, visual.scale);
    visual.updateMatrixWorld(true);
  }

  function syncSelectedObjectFromGizmo(render = true) {
    const item = selectedObject();
    const visual = item && data.visuals.get(item.id);
    if (!item || !visual) return;
    visual.updateMatrixWorld(true);
    const local = parentMatrix(item, options.state.frame || 0).invert().multiply(visual.matrixWorld.clone());
    const THREE = options.getThree();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    local.decompose(position, quaternion, scale);
    item.transform = { translation: position.toArray(), quaternion: quaternion.normalize().toArray(), scale: scale.toArray() };
    if (render) renderScenePanel();
  }

  function applySceneInputs(render = true) {
    const item = selectedObject();
    if (!item) return;
    item.name = ($("sceneObjectName").value || item.name).trim().slice(0, 80);
    item.visibility.default = $("sceneDefaultVisible").checked;
    item.parent_type = $("sceneParentType").value;
    const parentValue = $("sceneParentSelect").value || "";
    item.parent_id = item.parent_type === "ik_lock" ? parentValue : "";
    item.parent_joint = item.parent_type === "bone" ? Number(parentValue) : -1;
    updateObjectWorld(item, options.state.frame || 0, true);
    if (render) {
      renderOutliner();
      renderScenePanel();
    }
  }

  function refreshParentOptions() {
    const select = $("sceneParentSelect");
    if (!select) return;
    const type = $("sceneParentType")?.value || "world";
    const item = selectedObject();
    if (type === "world") {
      select.innerHTML = '<option value="">' + tr("\u573a\u666f\u6839", "Scene Root") + '</option>';
      select.disabled = true;
    } else if (type === "bone") {
      select.disabled = false;
      select.innerHTML = (data.sceneData?.names || []).map((name, index) => '<option value="' + index + '">' + options.escapeHtml(name) + '</option>').join("");
      select.value = String(item?.parent_joint ?? 0);
    } else {
      select.disabled = false;
      select.innerHTML = data.constraints.length
        ? data.constraints.map((constraint) => '<option value="' + options.escapeHtml(constraint.id) + '">' + options.escapeHtml(constraint.name) + '</option>').join("")
        : '<option value="">' + tr("\u6682\u65e0 IK \u9501\u5b9a\u70b9", "No IK lock points") + '</option>';
      select.value = item?.parent_id || "";
    }
  }

  function addVisibilityRange() {
    const item = selectedObject();
    if (!item) return;
    const start = clamp(Math.round(number("sceneVisibilityStart", 0)), 0, frameCount() - 1);
    const end = clamp(Math.round(number("sceneVisibilityEnd", start + 1)), start + 1, frameCount());
    item.visibility.ranges.push({ start_frame: start, end_frame: end, visible: $("sceneVisibilityAction").value === "true" });
    item.visibility.ranges.sort((a, b) => a.start_frame - b.start_frame);
    renderScenePanel();
    renderTimeline();
    updateFrame(options.state.frame || 0);
  }

  async function saveObjects() {
    if (!options.state.composition) return;
    const row = await options.api("/api/compositions/" + encodeURIComponent(options.state.composition.id) + "/scene-objects", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_objects: data.objects }),
    });
    data.objects = structuredClone(row.scene_objects || data.objects);
    options.onCompositionChange?.(row);
    renderOutliner();
    renderTimeline();
  }

  async function saveSelectedObject() {
    const item = selectedObject();
    if (!item) return;
    syncSelectedObjectFromGizmo();
    applySceneInputs();
    await saveObjects();
    status("\u573a\u666f\u5b9e\u4f8b\u5df2\u4fdd\u5b58\u3002", "Scene instance saved.");
  }

  async function deleteSelectedObject() {
    const item = selectedObject();
    if (!item) return;
    data.gizmo?.detach();
    disposeObject3D(data.visuals.get(item.id));
    data.visuals.delete(item.id);
    data.objects = data.objects.filter((entry) => entry.id !== item.id);
    data.selectedObjectId = "";
    await saveObjects();
    renderScenePanel();
    updateFrame(options.state.frame || 0);
  }

  function beginUniformScaleDrag() {
    const item = selectedObject();
    const visual = item && data.visuals.get(item.id);
    data.uniformScaleDrag = item && visual
      ? { objectId: item.id, visualScale: visual.scale.toArray() }
      : null;
  }

  function applyUniformScaleDrag() {
    const item = selectedObject();
    const visual = item && data.visuals.get(item.id);
    const drag = data.uniformScaleDrag;
    if (!item || !visual || !drag || drag.objectId !== item.id) return;
    const factor = uniformScaleMultiplier(drag.visualScale, visual.scale.toArray());
    visual.scale.fromArray(drag.visualScale).multiplyScalar(factor);
    visual.updateMatrixWorld(true);
  }

  function setGizmoMode(mode) {
    if (!data.gizmo) return;
    data.uniformScaleDrag = null;
    data.gizmo.setMode(mode);
    data.gizmo.setSpace(mode === "scale" ? "local" : "world");
    if (data.mode === "iklock") options.setIKTransformMode?.(mode);
    ["sceneMoveBtn", "ikLockMoveBtn"].forEach((id) => $(id)?.classList.toggle("active", mode === "translate"));
    ["sceneRotateBtn", "ikLockRotateBtn"].forEach((id) => $(id)?.classList.toggle("active", mode === "rotate"));
    $("sceneScaleBtn")?.classList.toggle("active", mode === "scale");
  }

  function selectObject(id) {
    const item = data.objects.find((entry) => entry.id === id);
    if (!item) return;
    data.selectedObjectId = id;
    data.selectedConstraintId = "";
    if (data.mode === "scene") data.gizmo?.attach(data.visuals.get(id));
    renderOutliner();
    renderScenePanel();
  }

  function renderScenePanel() {
    const item = selectedObject();
    const panel = $("sceneImportPanel");
    if (!panel) return;
    panel.classList.toggle("has-selection", Boolean(item));
    ["sceneObjectName", "sceneParentType", "sceneParentSelect", "sceneDefaultVisible", "sceneVisibilityStart", "sceneVisibilityEnd", "sceneVisibilityAction", "sceneAddVisibilityBtn", "sceneClearVisibilityBtn", "sceneSaveObjectBtn", "sceneDeleteObjectBtn"].forEach((id) => {
      if ($(id)) $(id).disabled = !item;
    });
    if (!item) {
      $("sceneObjectName").value = "";
      $("sceneVisibilityList").innerHTML = '<small>' + tr("\u5728\u573a\u666f\u5927\u7eb2\u4e2d\u9009\u62e9\u4e00\u4e2a\u5b9e\u4f8b", "Select an instance in the Scene Outliner") + '</small>';
      refreshParentOptions();
      return;
    }
    $("sceneObjectName").value = item.name || "";
    $("sceneParentType").value = item.parent_type || "world";
    $("sceneDefaultVisible").checked = item.visibility?.default !== false;
    $("sceneVisibilityEnd").max = String(frameCount());
    refreshParentOptions();
    $("sceneVisibilityList").innerHTML = (item.visibility?.ranges || []).length
      ? item.visibility.ranges.map((range, index) => '<div class="final-mini-row"><span>' + range.start_frame + "-" + range.end_frame + " : " + (range.visible ? tr("\u663e\u793a", "Visible") : tr("\u9690\u85cf", "Hidden")) + '</span><button data-delete-visibility="' + index + '">x</button></div>').join("")
      : '<small>' + tr("\u65e0\u989d\u5916\u53ef\u89c1\u6027\u533a\u95f4", "No visibility overrides") + '</small>';
    $("sceneVisibilityList").querySelectorAll("[data-delete-visibility]").forEach((button) => {
      button.onclick = () => {
        item.visibility.ranges.splice(Number(button.dataset.deleteVisibility), 1);
        renderScenePanel();
        renderTimeline();
        updateFrame(options.state.frame || 0);
      };
    });
  }

  function effectorId(jointName) {
    return { LeftFoot: "left_foot", RightFoot: "right_foot", LeftHand: "left_hand", RightHand: "right_hand", Head: "head" }[jointName] || "";
  }

  function refreshEffectorOptions() {
    const select = $("ikLockEffector");
    if (!select) return;
    const current = select.value;
    const fullBody = ($("ikLockSolver")?.value || "two_bone") !== "two_bone";
    const labels = {
      LeftFoot: tr("\u5de6\u811a", "Left Foot"), RightFoot: tr("\u53f3\u811a", "Right Foot"),
      LeftHand: tr("\u5de6\u624b", "Left Hand"), RightHand: tr("\u53f3\u624b", "Right Hand"),
      Head: tr("\u5934\u90e8", "Head"),
    };
    const available = ["LeftFoot", "RightFoot", "LeftHand", "RightHand", ...(fullBody ? ["Head"] : [])]
      .filter((name) => !data.sceneData?.names?.length || data.sceneData.names.includes(name));
    select.innerHTML = available.map((name) => '<option value="' + name + '">' + labels[name] + '</option>').join("");
    if (available.includes(current)) select.value = current;
    else if (available.length) select.value = available[0];
  }

  function controlWorldTransform(control) {
    const THREE = options.getThree();
    if (!THREE || !control) return null;
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    control.getWorldPosition(position);
    control.getWorldQuaternion(quaternion);
    return { position, quaternion };
  }

  function isLockableIKControl(info = data.selectedIKControl) {
    return Boolean(info?.control && info?.jointName && info.type !== "pole" && info.lockable !== false);
  }

  function selectedIKControlTransform() {
    return isLockableIKControl() ? controlWorldTransform(data.selectedIKControl.control) : null;
  }

  function writeTargetFields(position) {
    if (!position) return;
    $("ikLockX").value = position.x.toFixed(4);
    $("ikLockY").value = position.y.toFixed(4);
    $("ikLockZ").value = position.z.toFixed(4);
  }

  function renderIKControlInfo() {
    const info = data.selectedIKControl;
    if (!$("ikLockControlInfo")) return;
    if (!info) {
      $("ikLockControlInfo").textContent = tr("\u8bf7\u5728\u89c6\u53e3\u4e2d\u9009\u62e9 IK \u63a7\u5236\u5668", "Select an IK control in the viewport");
      return;
    }
    $("ikLockControlInfo").textContent = isLockableIKControl(info)
      ? tr("\u5df2\u9009 IK \u63a7\u5236\u5668\uff1a", "Selected IK control: ") + (info.name || info.jointName)
      : tr("\u5df2\u9009 Pole \u63a7\u5236\u5668\uff08\u53ef\u9884\u89c8\uff0c\u4e0d\u53ef\u521b\u5efa\u9501\u5b9a\u70b9\uff09\uff1a", "Selected Pole control (preview only; cannot create a lock): ") + (info.name || "Pole");
  }

  function selectIKControl(info) {
    if (!info?.control) return;
    data.selectedIKControl = info;
    if (!data.selectingConstraintIK) {
      data.selectedConstraintKeyFrame = null;
      data.gizmo?.detach();
      options.setIKControllerGizmoVisible?.(true);
    }
    const transform = controlWorldTransform(info.control);
    if (isLockableIKControl(info)) {
      refreshEffectorOptions();
      if ([...$("ikLockEffector").options].some((option) => option.value === info.jointName)) $("ikLockEffector").value = info.jointName;
    }
    renderIKControlInfo();
    renderOutliner();
    renderConstraintPanel();
    writeTargetFields(transform?.position);
  }

  async function changeIKSolver() {
    const constraint = selectedConstraint();
    const jointName = constraint?.joint_name || $("ikLockEffector")?.value;
    const solver = $("ikLockSolver").value;
    if (constraint) constraint.solver = solver;
    refreshEffectorOptions();
    data.selectedIKControl = null;
    data.selectedConstraintKeyFrame = null;
    data.gizmo?.detach();
    data.selectingConstraintIK = Boolean(constraint);
    try {
      await options.configureIKMode?.(solver, true);
      if (constraint) options.selectIKEffector?.(solver, jointName);
    } finally {
      data.selectingConstraintIK = false;
    }
    if (constraint) {
      options.setIKControllerGizmoVisible?.(false);
      attachSelectedConstraintGizmo();
    }
    renderConstraintPanel();
  }

  function currentTargetPosition() {
    return [number("ikLockX"), number("ikLockY"), number("ikLockZ")];
  }

  function alignConstraintToSelectedControl(constraint, transform, updateTarget = false) {
    if (!constraint || !isLockableIKControl() || !transform) return false;
    const jointName = data.selectedIKControl.jointName;
    const changedJoint = constraint.joint_name !== jointName;
    constraint.solver = $("ikLockSolver").value;
    constraint.joint_name = jointName;
    constraint.effector_id = effectorId(jointName);
    if (changedJoint) {
      constraint.keys = [];
      data.selectedConstraintKeyFrame = null;
    }
    if (changedJoint || updateTarget) {
      constraint.target_position = transform.position.toArray();
      constraint.target_rotation = transform.quaternion.toArray();
    }
    return changedJoint;
  }

  function upsertConstraint() {
    if (!options.state.composition) {
      status("\u8bf7\u5148\u5411\u65f6\u95f4\u8f74\u6dfb\u52a0\u52a8\u753b\u3002", "Add an animation to the timeline first.");
      return;
    }
    const selectedControl = data.selectedIKControl;
    if (!isLockableIKControl(selectedControl)) {
      status("\u8bf7\u5148\u5728\u89c6\u53e3\u4e2d\u9009\u62e9\u53ef\u9501\u5b9a\u7684 IK Goal / Effector \u63a7\u5236\u5668\u3002", "Select a lockable IK Goal / Effector control in the viewport first.");
      return;
    }
    const transform = controlWorldTransform(selectedControl.control);
    if (!transform) return;
    const start = clamp(Math.round(options.state.frame || 0), 0, frameCount() - 1);
    const end = clamp(Math.round(number("ikLockEnd", frameCount() - 1)), start, frameCount() - 1);
    let constraint = selectedConstraint();
    if (!constraint) {
      const joint = selectedControl.jointName;
      constraint = {
        id: uid("ik"),
        name: ($("ikLockName").value || joint + " Lock").trim(),
        solver: $("ikLockSolver").value,
        effector_id: effectorId(joint),
        joint_name: joint,
        start_frame: start,
        end_frame: end,
        blend_in_frames: Math.round(number("ikLockBlendIn", 5)),
        blend_out_frames: Math.round(number("ikLockBlendOut", 5)),
        target_space: "world",
        parent_object_id: "",
        target_position: transform.position.toArray(),
        target_rotation: transform.quaternion.toArray(),
        keys: [],
        interpolation: $("ikLockInterpolation").value,
        root_behavior: $("ikLockRoot").value,
        iterations: 24,
        tolerance: 0.002,
        enabled: true,
      };
      data.constraints.push(constraint);
      data.selectedConstraintId = constraint.id;
      createConstraintView(constraint);
    } else {
      alignConstraintToSelectedControl(constraint, transform, true);
      constraint.name = ($("ikLockName").value || constraint.name).trim();
      constraint.start_frame = start;
      constraint.end_frame = end;
      constraint.blend_in_frames = clamp(Math.round(number("ikLockBlendIn", 5)), 0, 240);
      constraint.blend_out_frames = clamp(Math.round(number("ikLockBlendOut", 5)), 0, 240);
      constraint.interpolation = $("ikLockInterpolation").value;
      constraint.root_behavior = $("ikLockRoot").value;
      updateConstraintView(constraint);
    }
    data.selectedConstraintKeyFrame = null;
    selectConstraint(constraint.id);
    renderTimeline();
    status("IK \u9501\u5b9a\u70b9\u5df2\u66f4\u65b0\uff1b\u70b9\u51fb\u201c\u5e94\u7528 IK \u9501\u5b9a\u201d\u8fdb\u884c\u9010\u5e27\u6c42\u89e3\u3002", "IK lock updated; choose Apply IK Locks to solve it frame by frame.");
  }

  function upsertConstraintKey() {
    const constraint = selectedConstraint();
    if (!constraint) return;
    const view = data.targets.get(constraint.id);
    const frame = clamp(Math.round(options.state.frame || 0), 0, frameCount() - 1);
    const controlTransform = selectedIKControlTransform();
    if (controlTransform) alignConstraintToSelectedControl(constraint, controlTransform, false);
    const position = controlTransform?.position.toArray() || (view ? view.marker.position.toArray() : currentTargetPosition());
    const rotation = controlTransform?.quaternion.toArray() || (view ? view.marker.quaternion.toArray() : constraint.target_rotation);
    const key = constraint.keys.find((entry) => Number(entry.frame) === frame);
    if (key) {
      key.position = position;
      key.rotation = rotation;
    } else {
      constraint.keys.push({ frame, position, rotation });
      constraint.keys.sort((a, b) => a.frame - b.frame);
    }
    data.selectedConstraintKeyFrame = frame;
    updateConstraintView(constraint);
    attachSelectedConstraintGizmo();
    renderConstraintPanel();
    renderTimeline();
  }

  function deleteConstraintKey() {
    const constraint = selectedConstraint();
    if (!constraint) return;
    const frame = Math.round(options.state.frame || 0);
    constraint.keys = (constraint.keys || []).filter((entry) => Number(entry.frame) !== frame);
    data.selectedConstraintKeyFrame = null;
    updateConstraintView(constraint);
    attachSelectedConstraintGizmo();
    renderConstraintPanel();
    renderTimeline();
  }

  function syncTargetFromInputs() {
    const constraint = selectedConstraint();
    if (!constraint) return;
    const view = data.targets.get(constraint.id);
    const key = (constraint.keys || []).find((entry) => Number(entry.frame) === Number(data.selectedConstraintKeyFrame));
    if (key) key.position = currentTargetPosition();
    else if (!(constraint.keys || []).length) constraint.target_position = currentTargetPosition();
    updateConstraintView(constraint);
    options.previewIK?.(constraint, view?.marker).catch?.(showError);
  }

  function syncSelectedTargetFromGizmo(preview) {
    const constraint = selectedConstraint();
    const view = constraint && data.targets.get(constraint.id);
    if (!constraint || !view) return;
    const key = (constraint.keys || []).find((entry) => Number(entry.frame) === Number(data.selectedConstraintKeyFrame));
    const edited = key ? view.keyMarkers.get(Number(key.frame)) : view.marker;
    if (!edited) return;
    if (key) {
      key.position = edited.position.toArray();
      key.rotation = edited.quaternion.toArray();
      if (data.gizmo?.dragging) {
        view.marker.position.copy(edited.position);
        view.marker.quaternion.copy(edited.quaternion);
      } else {
        updateConstraintView(constraint);
        attachSelectedConstraintGizmo();
      }
    } else {
      constraint.target_position = edited.position.toArray();
      constraint.target_rotation = edited.quaternion.toArray();
    }
    writeTargetFields(edited.position);
    if (preview) options.previewIK?.(constraint, view.marker).catch?.(showError);
  }

  function makeNumberSprite(numberValue, color) {
    const THREE = options.getThree();
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 96;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, 96, 96);
    context.beginPath();
    context.arc(48, 48, 31, 0, Math.PI * 2);
    context.fillStyle = "rgba(10, 16, 22, .9)";
    context.fill();
    context.lineWidth = 6;
    context.strokeStyle = "#" + color.getHexString();
    context.stroke();
    context.fillStyle = "#ffffff";
    context.font = "700 40px system-ui";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(String(numberValue), 48, 49);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
    sprite.position.set(0, 0.085, 0);
    sprite.scale.set(0.095, 0.095, 1);
    sprite.renderOrder = 1510;
    return sprite;
  }

  function createConstraintView(constraint) {
    if (!data.runtime || data.targets.has(constraint.id)) return;
    const THREE = options.getThree();
    const marker = new THREE.Group();
    marker.userData.finalIKConstraintId = constraint.id;
    marker.userData.finalIKControlKind = "lock";
    marker.name = constraint.name;
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.075, 0.008, 8, 28),
      new THREE.MeshBasicMaterial({ color: 0xe4ff4e, depthTest: false, transparent: true, opacity: 0.95 }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.userData.finalIKConstraintId = constraint.id;
    ring.userData.finalIKControlKind = "lock";
    const core = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.035, 0),
      new THREE.MeshBasicMaterial({ color: 0x49e6ff, depthTest: false }),
    );
    core.userData.finalIKConstraintId = constraint.id;
    core.userData.finalIKControlKind = "lock";
    marker.add(ring, core);
    marker.renderOrder = 1500;
    data.runtime.scene.add(marker);
    const path = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xe4ff4e, transparent: true, opacity: 0.8, depthTest: false }),
    );
    path.renderOrder = 1490;
    data.runtime.scene.add(path);
    const keyRoot = new THREE.Group();
    keyRoot.name = constraint.name + " Curve Controls";
    data.runtime.scene.add(keyRoot);
    data.targets.set(constraint.id, { marker, path, keyRoot, keyMarkers: new Map() });
    updateConstraintView(constraint);
  }

  function updateConstraintView(constraint) {
    const view = data.targets.get(constraint.id);
    const THREE = options.getThree();
    if (!view || !THREE) return;
    const sampled = sampleConstraint(constraint, options.state.frame || 0);
    if (!data.gizmo?.dragging || data.gizmo.object !== view.marker) {
      view.marker.position.copy(sampled.position);
      view.marker.quaternion.copy(sampled.quaternion);
    }
    const keys = constraint.keys || [];
    const isSinglePoint = keys.length === 0;
    markerPickability(view.marker, constraint.id, isSinglePoint);
    const wantedFrames = new Set(keys.map((key) => Number(key.frame)));
    for (const [frame, keyMarker] of view.keyMarkers) {
      if (!wantedFrames.has(frame)) {
        if (data.gizmo?.object === keyMarker) data.gizmo.detach();
        disposeObject3D(keyMarker);
        view.keyMarkers.delete(frame);
      }
    }
    keys.forEach((key, index) => {
      const color = new THREE.Color().setHSL((0.08 + index * 0.19) % 1, 0.88, 0.6);
      const frame = Number(key.frame);
      let keyMarker = view.keyMarkers.get(frame);
      if (keyMarker && keyMarker.userData.finalIKKeyIndex !== index) {
        if (data.gizmo?.object === keyMarker) data.gizmo.detach();
        disposeObject3D(keyMarker);
        view.keyMarkers.delete(frame);
        keyMarker = null;
      }
      if (!keyMarker) {
        keyMarker = new THREE.Group();
        keyMarker.userData.finalIKConstraintId = constraint.id;
        keyMarker.userData.finalIKKeyFrame = frame;
        keyMarker.userData.finalIKKeyIndex = index;
        keyMarker.userData.finalIKControlKind = "key";
        const point = new THREE.Mesh(
          new THREE.SphereGeometry(0.027, 14, 10),
          new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.98 }),
        );
        point.userData = { ...keyMarker.userData };
        keyMarker.add(point, makeNumberSprite(index + 1, color));
        keyMarker.renderOrder = 1505;
        view.keyRoot.add(keyMarker);
        view.keyMarkers.set(frame, keyMarker);
      }
      if (!data.gizmo?.dragging || data.gizmo.object !== keyMarker) {
        keyMarker.position.fromArray(key.position || constraint.target_position || [0, 0, 0]);
        keyMarker.quaternion.fromArray(key.rotation || constraint.target_rotation || [0, 0, 0, 1]).normalize();
      }
      keyMarker.scale.setScalar(Number(data.selectedConstraintKeyFrame) === Number(key.frame) ? 1.28 : 1);
    });
    const positions = [];
    if (keys.length > 1) {
      const start = Number(keys[0].frame);
      const end = Number(keys[keys.length - 1].frame);
      const steps = Math.max(16, Math.min(160, end - start + 1));
      for (let index = 0; index < steps; index += 1) {
        const frame = start + (end - start) * index / Math.max(1, steps - 1);
        positions.push(sampleConstraint(constraint, frame).position);
      }
    }
    view.path.geometry.dispose();
    view.path.geometry = new THREE.BufferGeometry().setFromPoints(positions);
  }

  function markerPickability(marker, constraintId, enabled) {
    marker.traverse((child) => {
      if (enabled) child.userData.finalIKConstraintId = constraintId;
      else delete child.userData.finalIKConstraintId;
      child.userData.finalIKControlKind = "lock";
    });
  }

  function sampleConstraint(constraint, frame) {
    const THREE = options.getThree();
    const fallbackPosition = new THREE.Vector3().fromArray(constraint.target_position || [0, 0, 0]);
    const fallbackQuaternion = new THREE.Quaternion().fromArray(constraint.target_rotation || [0, 0, 0, 1]).normalize();
    const keys = [...(constraint.keys || [])].sort((a, b) => a.frame - b.frame);
    if (!keys.length) return { position: fallbackPosition, quaternion: fallbackQuaternion };
    if (frame <= keys[0].frame) return {
      position: new THREE.Vector3().fromArray(keys[0].position || constraint.target_position),
      quaternion: new THREE.Quaternion().fromArray(keys[0].rotation || constraint.target_rotation || [0, 0, 0, 1]).normalize(),
    };
    if (frame >= keys[keys.length - 1].frame) {
      const key = keys[keys.length - 1];
      return {
        position: new THREE.Vector3().fromArray(key.position || constraint.target_position),
        quaternion: new THREE.Quaternion().fromArray(key.rotation || constraint.target_rotation || [0, 0, 0, 1]).normalize(),
      };
    }
    const rightIndex = keys.findIndex((key) => key.frame >= frame);
    const leftIndex = rightIndex - 1;
    const left = keys[leftIndex];
    const right = keys[rightIndex];
    const amount = (frame - left.frame) / Math.max(1, right.frame - left.frame);
    const a = new THREE.Vector3().fromArray(left.position || constraint.target_position);
    const b = new THREE.Vector3().fromArray(right.position || constraint.target_position);
    let position;
    if (constraint.interpolation === "step") position = a;
    else if (constraint.interpolation === "linear") position = a.clone().lerp(b, amount);
    else {
      const before = new THREE.Vector3().fromArray((keys[Math.max(0, leftIndex - 1)].position || left.position || constraint.target_position));
      const after = new THREE.Vector3().fromArray((keys[Math.min(keys.length - 1, rightIndex + 1)].position || right.position || constraint.target_position));
      const t2 = amount * amount;
      const t3 = t2 * amount;
      position = before.clone().multiplyScalar(-0.5 * t3 + t2 - 0.5 * amount)
        .add(a.clone().multiplyScalar(1.5 * t3 - 2.5 * t2 + 1))
        .add(b.clone().multiplyScalar(-1.5 * t3 + 2 * t2 + 0.5 * amount))
        .add(after.clone().multiplyScalar(0.5 * t3 - 0.5 * t2));
    }
    const quaternion = new THREE.Quaternion().fromArray(left.rotation || constraint.target_rotation || [0, 0, 0, 1]).normalize()
      .slerp(new THREE.Quaternion().fromArray(right.rotation || constraint.target_rotation || [0, 0, 0, 1]).normalize(), amount);
    return { position, quaternion };
  }

  function selectConstraint(id) {
    const constraint = data.constraints.find((entry) => entry.id === id);
    if (!constraint) return;
    data.selectedConstraintId = id;
    data.selectedConstraintKeyFrame = null;
    data.selectedObjectId = "";
    if (!data.targets.has(id)) createConstraintView(constraint);
    if (data.mode === "iklock") {
      options.setIKControllerGizmoVisible?.(false);
      attachSelectedConstraintGizmo();
      $("ikLockSolver").value = constraint.solver || "pbik";
      refreshEffectorOptions();
      data.selectingConstraintIK = true;
      Promise.resolve(options.configureIKMode?.($("ikLockSolver").value, true))
        .then(() => options.selectIKEffector?.($("ikLockSolver").value, constraint.joint_name))
        .catch(showError)
        .finally(() => {
          data.selectingConstraintIK = false;
          options.setIKControllerGizmoVisible?.(false);
          attachSelectedConstraintGizmo();
          const view = data.targets.get(constraint.id);
          if (view) options.previewIK?.(constraint, view.marker).catch?.(showError);
        });
    }
    renderOutliner();
    renderConstraintPanel();
    updateFrame(options.state.frame || 0);
  }

  function attachSelectedConstraintGizmo() {
    if (data.mode !== "iklock" || !data.gizmo) return;
    const constraint = selectedConstraint();
    const view = constraint && data.targets.get(constraint.id);
    if (!constraint || !view) {
      data.gizmo.detach();
      return;
    }
    const keyMarker = view.keyMarkers.get(Number(data.selectedConstraintKeyFrame));
    if (keyMarker) data.gizmo.attach(keyMarker);
    else if (!(constraint.keys || []).length) data.gizmo.attach(view.marker);
    else data.gizmo.detach();
  }

  function selectConstraintKey(constraintId, frame) {
    const constraint = data.constraints.find((entry) => entry.id === constraintId);
    if (!constraint || !(constraint.keys || []).some((key) => Number(key.frame) === Number(frame))) return;
    data.selectedConstraintId = constraintId;
    data.selectedConstraintKeyFrame = Number(frame);
    options.setIKControllerGizmoVisible?.(false);
    options.setFrame?.(Number(frame));
    updateConstraintView(constraint);
    attachSelectedConstraintGizmo();
    renderOutliner();
    renderConstraintPanel();
  }

  async function saveConstraints(reload) {
    if (!options.state.composition) return;
    const row = await options.api("/api/compositions/" + encodeURIComponent(options.state.composition.id) + "/ik-constraints", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ik_constraints: data.constraints }),
    });
    data.constraints = structuredClone(row.ik_constraints || data.constraints);
    options.onCompositionChange?.(row);
    renderOutliner();
    renderTimeline();
    if (reload) {
      status("\u6b63\u5728\u4f7f\u7528\u9009\u5b9a\u89e3\u7b97\u5668\u9010\u5e27\u5e94\u7528 IK \u9501\u5b9a\u2026", "Applying IK locks frame by frame with the selected solvers...");
      await options.reloadComposition?.(true);
      status("IK \u9501\u5b9a\u5df2\u5e94\u7528\uff0c\u524d\u540e\u5e73\u6ed1\u533a\u95f4\u5df2\u751f\u6548\u3002", "IK locks applied with blend-in and blend-out ranges.");
    }
  }

  async function deleteSelectedConstraint() {
    const constraint = selectedConstraint();
    if (!constraint) return;
    data.gizmo?.detach();
    const view = data.targets.get(constraint.id);
    disposeObject3D(view?.marker);
    disposeObject3D(view?.path);
    disposeObject3D(view?.keyRoot);
    data.targets.delete(constraint.id);
    data.constraints = data.constraints.filter((entry) => entry.id !== constraint.id);
    for (const item of data.objects) {
      if (item.parent_type === "ik_lock" && item.parent_id === constraint.id) {
        item.parent_type = "world";
        item.parent_id = "";
      }
    }
    data.selectedConstraintId = "";
    data.selectedConstraintKeyFrame = null;
    await saveConstraints(true);
    await saveObjects();
    renderConstraintPanel();
  }

  function updateIKActionAvailability() {
    const constraint = selectedConstraint();
    const currentFrame = clamp(Math.round(options.state.frame || 0), 0, frameCount() - 1);
    const hasControl = isLockableIKControl();
    $("ikLockUpsertBtn").disabled = !hasControl;
    $("ikLockAddKeyBtn").disabled = !constraint || !hasControl;
    $("ikLockDeleteKeyBtn").disabled = !constraint || !(constraint.keys || []).some((key) => Number(key.frame) === currentFrame);
    $("ikLockApplyBtn").disabled = !constraint;
    $("ikLockDeleteBtn").disabled = !constraint;
  }

  function renderConstraintPanel() {
    const constraint = selectedConstraint();
    const currentFrame = clamp(Math.round(options.state.frame || 0), 0, frameCount() - 1);
    $("ikLockStart").value = String(currentFrame);
    $("ikLockStart").readOnly = true;
    $("ikLockStart").disabled = true;
    updateIKActionAvailability();
    $("ikLockEnd").max = String(frameCount() - 1);
    if (!constraint) {
      $("ikLockName").value = "";
      $("ikLockEnd").value = String(frameCount() - 1);
      $("ikLockKeyList").innerHTML = '<small>' + tr("\u53ef\u65b0\u5efa\u9501\u5b9a\u70b9\uff0c\u6216\u5728\u573a\u666f\u5927\u7eb2\u4e2d\u9009\u62e9", "Create a lock or select one in the Scene Outliner") + '</small>';
      return;
    }
    $("ikLockName").value = constraint.name || "";
    $("ikLockSolver").value = constraint.solver || "pbik";
    refreshEffectorOptions();
    const selectedJointName = isLockableIKControl() ? data.selectedIKControl.jointName : constraint.joint_name;
    if ([...$("ikLockEffector").options].some((option) => option.value === selectedJointName)) {
      $("ikLockEffector").value = selectedJointName;
    }
    $("ikLockEnd").value = String(constraint.end_frame ?? frameCount() - 1);
    $("ikLockBlendIn").value = String(constraint.blend_in_frames || 0);
    $("ikLockBlendOut").value = String(constraint.blend_out_frames || 0);
    $("ikLockInterpolation").value = constraint.interpolation || "cubic";
    $("ikLockRoot").value = constraint.root_behavior || "pre_pull";
    const selectedKey = (constraint.keys || []).find((key) => Number(key.frame) === Number(data.selectedConstraintKeyFrame));
    const sampled = selectedIKControlTransform() || (selectedKey
      ? sampleConstraint(constraint, Number(selectedKey.frame))
      : sampleConstraint(constraint, currentFrame));
    $("ikLockX").value = sampled.position.x.toFixed(4);
    $("ikLockY").value = sampled.position.y.toFixed(4);
    $("ikLockZ").value = sampled.position.z.toFixed(4);
    $("ikLockKeyList").innerHTML = (constraint.keys || []).length
      ? constraint.keys.map((key, index) => '<button class="final-key-row ' + (Number(data.selectedConstraintKeyFrame) === Number(key.frame) ? 'selected' : '') + '" data-ik-key-frame="' + key.frame + '"><i style="--key-color:hsl(' + ((29 + index * 68) % 360) + ' 88% 60%)">' + (index + 1) + '</i>' + tr("\u5e27 ", "Frame ") + key.frame + " \u00b7 " + key.position.map((value) => Number(value).toFixed(2)).join(", ") + '</button>').join("")
      : '<small>' + tr("\u56fa\u5b9a\u76ee\u6807\uff08\u65e0\u53d8\u5316\u66f2\u7ebf\uff09", "Static target (no animated keys)") + '</small>';
    $("ikLockKeyList").querySelectorAll("[data-ik-key-frame]").forEach((button) => {
      button.onclick = () => selectConstraintKey(constraint.id, Number(button.dataset.ikKeyFrame));
    });
  }

  function renderOutliner() {
    const body = $("finalOutlinerBody");
    if (!body) return;
    const section = (label, rows) => '<section><h4>' + label + '</h4>' + (rows || '<small>' + tr("\u65e0", "None") + '</small>') + '</section>';
    const animationTerrain = data.sceneData?.terrain
      ? '<button class="final-outliner-row" data-animation-terrain><span class="eye">' + (data.runtime?.terrainObject?.visible === false ? "\u25cb" : "\u25c9") + '</span><span>' + tr("\u52a8\u753b\u81ea\u5e26\u5730\u5f62", "Animation Terrain") + '</span></button>'
      : "";
    const objectRows = data.objects.map((item) => '<button class="final-outliner-row ' + (item.id === data.selectedObjectId ? "selected" : "") + '" data-scene-object="' + options.escapeHtml(item.id) + '"><span class="eye" data-scene-eye="' + options.escapeHtml(item.id) + '">' + (item.editor_hidden ? "\u25cb" : "\u25c9") + '</span><span>' + options.escapeHtml(item.name) + '</span><small>' + (item.asset_kind === "terrain" ? tr("\u5730\u5f62", "Terrain") : tr("\u9759\u6001\u7f51\u683c", "Static")) + '</small></button>').join("");
    const constraintRows = data.constraints.map((item) => '<button class="final-outliner-row ' + (item.id === data.selectedConstraintId ? "selected" : "") + '" data-ik-constraint="' + options.escapeHtml(item.id) + '"><span class="eye">\u25c8</span><span>' + options.escapeHtml(item.name) + '</span><small>' + options.escapeHtml(item.solver) + '</small></button>').join("");
    body.innerHTML = [
      section(tr("\u52a8\u753b\u4e0a\u4e0b\u6587", "Animation Context"), animationTerrain),
      section(tr("\u573a\u666f\u8d44\u4ea7", "Scene Assets"), objectRows),
      section(tr("IK \u9501\u5b9a\u70b9", "IK Lock Points"), constraintRows),
    ].join("");
    body.querySelectorAll("[data-scene-object]").forEach((button) => button.onclick = (event) => {
      if (event.target.closest("[data-scene-eye]")) {
        event.stopPropagation();
        const item = data.objects.find((entry) => entry.id === button.dataset.sceneObject);
        if (item) {
          item.editor_hidden = !item.editor_hidden;
          updateFrame(options.state.frame || 0);
          renderOutliner();
          saveObjects().catch(showError);
        }
        return;
      }
      selectObject(button.dataset.sceneObject);
    });
    body.querySelectorAll("[data-ik-constraint]").forEach((button) => button.onclick = () => selectConstraint(button.dataset.ikConstraint));
    body.querySelector("[data-animation-terrain]")?.addEventListener("click", () => {
      const object = data.runtime?.terrainObject;
      if (!object) return;
      object.visible = !object.visible;
      renderOutliner();
    });
  }

  function selectFromViewport(event) {
    if (!data.runtime || !["scene", "iklock"].includes(data.mode) || data.gizmo?.dragging) return;
    const rect = data.runtime.canvas.getBoundingClientRect();
    data.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    data.raycaster.setFromCamera(data.pointer, data.runtime.camera);
    const roots = data.mode === "scene"
      ? [...data.visuals.values()]
      : [...data.targets.values()].flatMap((view) => [view.marker, ...view.keyMarkers.values()]);
    const hit = data.raycaster.intersectObjects(roots, true)[0];
    if (!hit) return;
    let object = hit.object;
    while (object && !object.userData.finalSceneObjectId && !object.userData.finalIKConstraintId) object = object.parent;
    if (object?.userData.finalSceneObjectId) selectObject(object.userData.finalSceneObjectId);
    if (object?.userData.finalIKConstraintId && object.userData.finalIKControlKind === "key") {
      event.preventDefault();
      event.stopImmediatePropagation();
      selectConstraintKey(object.userData.finalIKConstraintId, object.userData.finalIKKeyFrame);
    } else if (object?.userData.finalIKConstraintId) {
      event.preventDefault();
      event.stopImmediatePropagation();
      selectConstraint(object.userData.finalIKConstraintId);
    }
  }

  async function enterMode(mode) {
    if (!options.state.composition || !data.sceneData) {
      status("\u8bf7\u5148\u5411\u65f6\u95f4\u8f74\u6dfb\u52a0\u52a8\u753b\u3002", "Add an animation to the timeline first.");
      return;
    }
    data.mode = mode;
    if (mode === "scene") refreshCatalogs().catch(showError);
    options.state.playing = false;
    options.showMode?.(mode);
    options.setTimelineMutationDisabled?.(true);
    data.runtime?.gizmo?.detach();
    data.runtime?.gizmo?.getHelper() && (data.runtime.gizmo.getHelper().visible = false);
    if (mode === "scene" && data.selectedObjectId) data.gizmo?.attach(data.visuals.get(data.selectedObjectId));
    if (mode === "iklock") {
      data.runtime?.markers?.forEach((marker) => { marker.visible = false; });
      const constraint = selectedConstraint();
      if (constraint) $("ikLockSolver").value = constraint.solver || "pbik";
      refreshEffectorOptions();
      data.selectingConstraintIK = Boolean(constraint);
      try {
        await options.configureIKMode?.($("ikLockSolver").value || "two_bone", true);
        if (constraint) options.selectIKEffector?.($("ikLockSolver").value, constraint.joint_name);
      } finally {
        data.selectingConstraintIK = false;
      }
      if (constraint) options.setIKControllerGizmoVisible?.(false);
      attachSelectedConstraintGizmo();
    }
    if (data.gizmo?.getHelper()) data.gizmo.getHelper().visible = true;
    updateFrame(options.state.frame || 0);
    status(mode === "scene" ? "\u573a\u666f\u5bfc\u5165\u6a21\u5f0f\uff1a\u4ece\u5de6\u4fa7\u5927\u7eb2\u9009\u62e9\u3001\u79fb\u52a8\u6216\u5220\u9664\u5b9e\u4f8b\u3002" : "IK \u9501\u5b9a\u6a21\u5f0f\uff1a\u62d6\u52a8\u9501\u5b9a\u70b9\u9884\u89c8\u5168\u8eab\u53cd\u63a8\u3002", mode === "scene" ? "Scene Import mode: select, transform, or delete instances in the Outliner." : "IK Lock mode: drag targets to preview the full-body response.");
  }

  function exitMode() {
    data.gizmo?.detach();
    if (data.gizmo?.getHelper()) data.gizmo.getHelper().visible = false;
    data.mode = "timeline";
    options.setTimelineMutationDisabled?.(false);
    options.showMode?.("timeline");
    if (data.runtime?.gizmo?.getHelper()) data.runtime.gizmo.getHelper().visible = true;
    options.configureIKMode?.($("ikLockSolver")?.value || "two_bone", false).catch?.(showError);
    data.runtime?.markers?.forEach((marker) => { marker.visible = true; });
    for (const view of data.targets.values()) {
      view.marker.visible = false;
      view.path.visible = false;
      view.keyRoot.visible = false;
    }
    options.resetPose?.();
    status("\u5df2\u9000\u51fa\u7f16\u8f91\u6a21\u5f0f\u3002", "Editor mode closed.");
  }

  function updateFrame(frame) {
    for (const item of data.objects) updateObjectWorld(item, frame);
    for (const constraint of data.constraints) {
      updateConstraintView(constraint);
      const view = data.targets.get(constraint.id);
      if (!view) continue;
      const visible = data.mode === "iklock";
      view.marker.visible = visible && !(constraint.keys || []).length;
      view.path.visible = visible && (constraint.keys || []).length > 1;
      view.keyRoot.visible = visible && (constraint.keys || []).length > 0;
      view.marker.traverse((child) => {
        if (child.material) child.material.opacity = frame >= constraint.start_frame && frame <= constraint.end_frame ? 0.95 : 0.34;
      });
    }
    if (data.mode === "iklock") {
      $("ikLockStart").value = String(clamp(Math.round(frame || 0), 0, frameCount() - 1));
      renderConstraintPanel();
      const constraint = selectedConstraint();
      const view = constraint && data.targets.get(constraint.id);
      if (constraint && view && !data.gizmo?.dragging && !data.selectingConstraintIK) options.previewIK?.(constraint, view.marker).catch?.(showError);
    }
    if (data.mode === "scene") renderScenePanel();
  }

  function appendTimelineTracks(track, totalFrames, top) {
    const playhead = $("timelinePlayhead");
    let lane = 0;
    for (const constraint of data.constraints) {
      const marker = document.createElement("span");
      marker.className = "edit-marker final-ik-lock-marker";
      marker.style.top = (top + lane * 42) + "px";
      marker.dataset.ikConstraintId = constraint.id;
      marker.innerHTML = '<span class="final-ik-blend-in"></span><span class="final-ik-lock-core"></span><span class="final-ik-blend-out"></span><b>IK \u00b7 ' + options.escapeHtml(constraint.name) + '</b>';
      for (const side of ["in", "out"]) {
        const handle = document.createElement("i");
        handle.className = "marker-handle " + (side === "in" ? "left" : "right");
        handle.onpointerdown = (event) => dragIKBlend(event, marker, constraint, side, totalFrames);
        marker.appendChild(handle);
      }
      layoutIKTimelineMarker(marker, constraint, totalFrames);
      marker.onclick = async (event) => {
        event.stopPropagation();
        if (marker.dataset.dragging) return;
        await enterMode("iklock");
        selectConstraint(constraint.id);
      };
      track.insertBefore(marker, playhead);
      lane += 1;
    }
    for (const item of data.objects) {
      const ranges = item.visibility?.ranges || [];
      if (!ranges.length) continue;
      for (const range of ranges) {
        const marker = document.createElement("span");
        marker.className = "edit-marker final-visibility-marker";
        marker.style.left = (range.start_frame / Math.max(1, totalFrames) * 100) + "%";
        marker.style.width = (Math.max(1, range.end_frame - range.start_frame) / Math.max(1, totalFrames) * 100) + "%";
        marker.style.top = (top + lane * 42) + "px";
        marker.textContent = (range.visible ? tr("\u663e\u793a", "Show") : tr("\u9690\u85cf", "Hide")) + " \u00b7 " + item.name;
        marker.onclick = async (event) => { event.stopPropagation(); await enterMode("scene"); selectObject(item.id); };
        track.insertBefore(marker, playhead);
      }
      lane += 1;
    }
    return lane;
  }

  function layoutIKTimelineMarker(marker, constraint, totalFrames) {
    const total = Math.max(1, totalFrames);
    const coreStart = clamp(Math.round(Number(constraint.start_frame) || 0), 0, total - 1);
    const coreEnd = clamp(Math.round(Number(constraint.end_frame) || coreStart), coreStart, total - 1) + 1;
    const blendIn = clamp(Math.round(Number(constraint.blend_in_frames) || 0), 0, coreStart);
    const blendOut = clamp(Math.round(Number(constraint.blend_out_frames) || 0), 0, total - coreEnd);
    const outerStart = coreStart - blendIn;
    const outerEnd = coreEnd + blendOut;
    const span = Math.max(1, outerEnd - outerStart);
    marker.style.left = (outerStart / total * 100) + "%";
    marker.style.width = (span / total * 100) + "%";
    const before = marker.querySelector(".final-ik-blend-in");
    const core = marker.querySelector(".final-ik-lock-core");
    const after = marker.querySelector(".final-ik-blend-out");
    if (before) before.style.width = (blendIn / span * 100) + "%";
    if (core) {
      core.style.left = (blendIn / span * 100) + "%";
      core.style.width = ((coreEnd - coreStart) / span * 100) + "%";
    }
    if (after) after.style.width = (blendOut / span * 100) + "%";
    marker.title = "IK \u00b7 " + constraint.name + " \u00b7 " + tr("\u524d\u5e73\u6ed1 ", "Blend in ") + blendIn + tr(" \u5e27 / \u540e\u5e73\u6ed1 ", " frames / blend out ") + blendOut + tr(" \u5e27", " frames");
  }

  function dragIKBlend(event, marker, constraint, side, totalFrames) {
    event.preventDefault();
    event.stopPropagation();
    const rect = $("timelineTrack").getBoundingClientRect();
    const startX = event.clientX;
    const original = side === "in" ? Number(constraint.blend_in_frames || 0) : Number(constraint.blend_out_frames || 0);
    const limit = side === "in"
      ? Math.max(0, Number(constraint.start_frame) || 0)
      : Math.max(0, totalFrames - 1 - (Number(constraint.end_frame) || 0));
    marker.dataset.dragging = "1";
    const move = (moveEvent) => {
      const delta = Math.round((moveEvent.clientX - startX) / Math.max(1, rect.width) * totalFrames);
      const value = clamp(side === "in" ? original - delta : original + delta, 0, limit);
      if (side === "in") constraint.blend_in_frames = value;
      else constraint.blend_out_frames = value;
      layoutIKTimelineMarker(marker, constraint, totalFrames);
      if (constraint.id === data.selectedConstraintId) {
        $(side === "in" ? "ikLockBlendIn" : "ikLockBlendOut").value = String(value);
      }
      status("IK \u5e73\u6ed1\u533a\u95f4\uff1a\u524d " + (constraint.blend_in_frames || 0) + " \u5e27 / \u540e " + (constraint.blend_out_frames || 0) + " \u5e27", "IK smoothing: " + (constraint.blend_in_frames || 0) + " blend-in frames / " + (constraint.blend_out_frames || 0) + " blend-out frames");
    };
    const commit = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", commit);
      setTimeout(() => { delete marker.dataset.dragging; }, 0);
      saveConstraints(false).catch(showError);
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", commit, { once: true });
  }

  function timelineLaneCount() {
    return data.constraints.length + data.objects.filter((item) => (item.visibility?.ranges || []).length).length;
  }

  async function importBoneAttachments(configs) {
    if (!options.state.composition || !configs?.length) return;
    for (const config of configs) {
      const item = {
        id: config.id || uid("component"),
        name: config.name || tr("\u9aa8\u9abc\u7ec4\u4ef6", "Bone Component"),
        asset_kind: "static_mesh",
        source_asset_id: config.asset_id,
        asset_revision: "",
        parent_type: "bone",
        parent_id: "",
        parent_joint: Number(config.joint),
        transform: {
          translation: (config.position || [0, 0, 0]).slice(0, 3),
          quaternion: (config.quaternion || [0, 0, 0, 1]).slice(0, 4),
          scale: [Number(config.scale || 1), Number(config.scale || 1), Number(config.scale || 1)],
        },
        visibility: { default: true, ranges: [] },
        editor_hidden: false,
      };
      data.objects.push(item);
      await createObjectVisual(item);
    }
    await saveObjects();
    renderOutliner();
    updateFrame(options.state.frame || 0);
  }

  function renderTimeline() {
    options.renderTimeline?.();
  }

  function syncSelectedIKControl() {
    const transform = selectedIKControlTransform();
    renderIKControlInfo();
    writeTargetFields(transform?.position);
    updateIKActionAvailability();
  }

  ensureUi();
  return {
    attachRuntime,
    load,
    updateFrame,
    refreshCatalogs,
    appendTimelineTracks,
    timelineLaneCount,
    importBoneAttachments,
    selectIKControl,
    syncSelectedIKControl,
    enterMode,
    exitMode,
    get mode() { return data.mode; },
  };
}
