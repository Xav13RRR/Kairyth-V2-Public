const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const uid = prefix => `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

function rotvecFromQuaternion(quaternion) {
  let { x, y, z, w } = quaternion;
  const norm = Math.hypot(x, y, z, w) || 1;
  x /= norm; y /= norm; z /= norm; w /= norm;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const vectorLength = Math.hypot(x, y, z);
  if (vectorLength < 1e-9) return [0, 0, 0];
  const scale = 2 * Math.atan2(vectorLength, w) / vectorLength;
  return [x * scale, y * scale, z * scale];
}

function quaternionFromRotvec(THREE, value) {
  const vector = new THREE.Vector3().fromArray(value || [0, 0, 0]);
  const angle = vector.length();
  return angle < 1e-9
    ? new THREE.Quaternion()
    : new THREE.Quaternion().setFromAxisAngle(vector.multiplyScalar(1 / angle), angle);
}

function disposeObject(object) {
  if (!object) return;
  object.removeFromParent();
  object.traverse?.(child => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) child.material.forEach(material => material.dispose?.());
    else child.material?.dispose?.();
  });
}

export function createKimodoAuthoring(options) {
  const $ = id => document.getElementById(id);
  const tr = (zh, en) => window.MotionBuilderI18n?.getLanguage?.() === "en" ? en : zh;
  const data = {
    runtime: null,
    sceneData: null,
    activeTool: "trajectory",
    objects: [],
    visuals: new Map(),
    selectedObjectId: "",
    sceneGizmo: null,
    sceneMode: "translate",
    scaleDrag: null,
    terrains: [],
    assetCache: new Map(),
    saveTimer: 0,
    statusPair: null,
    sceneActive: false,
    poseGizmo: null,
    poseMarkers: [],
    poseJoint: 0,
    poseMode: "rotate",
    poseDrag: null,
    waypointPose: null,
  };

  function status(zh, en = zh) {
    data.statusPair = [zh, en];
    const node = $("kimodoAuthoringHint");
    if (node) node.textContent = tr(zh, en);
  }

  function capturePose() {
    const THREE = options.getThree();
    const bones = data.runtime?.bones || [];
    if (!THREE || !bones.length) throw new Error(tr("SOMA 骨架尚未加载", "The SOMA skeleton is not loaded"));
    data.runtime.scene.updateMatrixWorld(true);
    const rootPosition = new THREE.Vector3();
    bones[0].getWorldPosition(rootPosition);
    return {
      root: rootPosition.toArray(),
      rotations: bones.map(bone => rotvecFromQuaternion(bone.quaternion)),
    };
  }

  function applyPose(constraint) {
    const THREE = options.getThree();
    const bones = data.runtime?.bones || [];
    const rotations = constraint?.local_joints_rot?.[0];
    const root = constraint?.root_positions?.[0];
    if (!THREE || !rotations || !bones.length) return;
    rotations.slice(0, bones.length).forEach((rotation, index) => {
      bones[index].quaternion.copy(quaternionFromRotvec(THREE, rotation));
    });
    if (root) bones[0].position.fromArray(root);
    data.runtime.scene.updateMatrixWorld(true);
  }

  function updatePoseMarkers() {
    if (!data.runtime) return;
    data.runtime.scene.updateMatrixWorld(true);
    data.runtime.bones.forEach((bone, index) => {
      const marker = data.poseMarkers[index];
      if (!marker) return;
      bone.getWorldPosition(marker.position);
      marker.visible = data.activeTool === "waypoint-pose";
      marker.material.color.setHex(index === data.poseJoint ? 0xe4ff4e : 0xcc68ff);
      marker.scale.setScalar(index === data.poseJoint ? 1.35 : 1);
    });
  }

  function canonicalWaypointPose() {
    if (!data.waypointPose) return null;
    const THREE = options.getThree();
    const pose = capturePose();
    const yaw = THREE.MathUtils.degToRad(Number(data.waypointPose.point?.yaw_deg) || 0);
    const inverseYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -yaw);
    const rootQuaternion = quaternionFromRotvec(THREE, pose.rotations[0]);
    pose.rotations[0] = rotvecFromQuaternion(inverseYaw.multiply(rootQuaternion).normalize());
    return {
      type: "fullbody",
      frame_indices: [0],
      root_positions: [[0, pose.root[1], 0]],
      local_joints_rot: [pose.rotations],
      smooth_root_2d: [[0, 0]],
    };
  }

  function emitWaypointPose() {
    const pose = canonicalWaypointPose();
    if (pose) options.onWaypointPoseChange?.(pose);
  }

  function applyWaypointPose(constraint) {
    if (!constraint || !data.waypointPose) return;
    const THREE = options.getThree();
    const copy = structuredClone(constraint);
    const point = data.waypointPose.point;
    const rotations = copy.local_joints_rot?.[0];
    if (!rotations) return;
    const yaw = THREE.MathUtils.degToRad(Number(point?.yaw_deg) || 0);
    const yawQuaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    rotations[0] = rotvecFromQuaternion(yawQuaternion.multiply(quaternionFromRotvec(THREE, rotations[0])).normalize());
    const root = copy.root_positions?.[0] || [0, 0, 0];
    copy.root_positions = [[Number(point?.x) || 0, Number(root[1]) || 0, Number(point?.z) || 0]];
    applyPose(copy);
    updatePoseMarkers();
  }

  function selectWaypointPoseJoint(index) {
    const bones = data.runtime?.bones || [];
    if (!bones[index] || !data.poseGizmo) return;
    data.poseJoint = index;
    if ($("kimodoWaypointPoseNode")) $("kimodoWaypointPoseNode").value = String(index);
    data.poseGizmo.attach(bones[index]);
    data.poseGizmo.setMode(data.poseMode);
    data.poseGizmo.setSpace(data.poseMode === "rotate" ? "local" : "world");
    updatePoseMarkers();
    status(`正在编辑航迹点姿势骨骼：${bones[index].name}`, `Editing waypoint pose bone: ${bones[index].name}`);
  }

  function setWaypointPoseMode(mode) {
    data.poseMode = mode === "translate" ? "translate" : "rotate";
    $("kimodoWaypointPoseTranslate")?.classList.toggle("active", data.poseMode === "translate");
    $("kimodoWaypointPoseRotate")?.classList.toggle("active", data.poseMode === "rotate");
    selectWaypointPoseJoint(data.poseJoint);
  }

  function pickWaypointPoseJoint(event) {
    if (data.activeTool !== "waypoint-pose" || data.poseGizmo?.dragging || !data.runtime) return;
    const THREE = options.getThree();
    const rect = data.runtime.canvas.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      (event.clientX - rect.left) / rect.width * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, data.runtime.camera);
    const hit = raycaster.intersectObjects(data.poseMarkers, false)[0];
    if (hit) selectWaypointPoseJoint(Number(hit.object.userData.kimodoPoseJoint));
  }

  function buildWaypointPoseTools() {
    const THREE = options.getThree();
    const TransformControls = options.getTransformControls();
    if (!THREE || !TransformControls || !data.runtime) return;
    data.poseGizmo = new TransformControls(data.runtime.camera, data.runtime.canvas);
    data.poseGizmo.setMode("rotate");
    data.poseGizmo.setSpace("local");
    data.poseGizmo.setSize(0.75);
    data.runtime.scene.add(data.poseGizmo.getHelper());
    data.poseGizmo.getHelper().visible = false;
    data.poseGizmo.addEventListener("dragging-changed", event => {
      data.runtime.controls.enabled = !event.value;
      const THREE = options.getThree();
      const bone = data.runtime?.bones?.[data.poseJoint];
      if (event.value && bone && data.poseMode === "translate") {
        data.runtime.scene.updateMatrixWorld(true);
        data.poseDrag = { local: bone.position.clone() };
      }
      if (!event.value && bone) {
        if (data.poseMode === "translate" && data.poseDrag) {
          const target = bone.getWorldPosition(new THREE.Vector3());
          if (data.poseJoint === 0) {
            data.waypointPose.point.x = target.x;
            data.waypointPose.point.z = target.z;
            options.onWaypointTransformChange?.();
          } else if (bone.parent?.isBone) {
            bone.position.copy(data.poseDrag.local);
            data.runtime.scene.updateMatrixWorld(true);
            const parent = bone.parent;
            const origin = parent.getWorldPosition(new THREE.Vector3());
            const current = bone.getWorldPosition(new THREE.Vector3()).sub(origin).normalize();
            const desired = target.clone().sub(origin).normalize();
            if (current.lengthSq() > 1e-8 && desired.lengthSq() > 1e-8) {
              const delta = new THREE.Quaternion().setFromUnitVectors(current, desired);
              const parentWorld = parent.getWorldQuaternion(new THREE.Quaternion());
              const desiredWorld = delta.multiply(parentWorld).normalize();
              const ancestorWorld = parent.parent?.getWorldQuaternion(new THREE.Quaternion()) || new THREE.Quaternion();
              parent.quaternion.copy(ancestorWorld.invert().multiply(desiredWorld).normalize());
            }
          }
        }
        data.poseDrag = null;
        data.runtime.scene.updateMatrixWorld(true);
        updatePoseMarkers();
        emitWaypointPose();
      }
    });
    data.poseGizmo.addEventListener("objectChange", () => {
      data.runtime.scene.updateMatrixWorld(true);
      updatePoseMarkers();
    });
    const size = Math.max(Number(data.sceneData?.ik_profile?.control_scale) || 0.04, 0.025) * 0.42;
    data.poseMarkers = data.runtime.bones.map((bone, index) => {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(size, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0xcc68ff, depthTest: false, transparent: true, opacity: 0.9 }),
      );
      marker.renderOrder = 1400;
      marker.userData.kimodoPoseJoint = index;
      marker.visible = false;
      data.runtime.scene.add(marker);
      return marker;
    });
    data.runtime.canvas.removeEventListener("pointerdown", pickWaypointPoseJoint);
    data.runtime.canvas.addEventListener("pointerdown", pickWaypointPoseJoint);
  }

  function beginWaypointPose(point, sourceConstraint, overrideConstraint = null) {
    if (!data.runtime || !sourceConstraint?.local_joints_rot?.length) return false;
    data.sceneActive = false;
    data.activeTool = "waypoint-pose";
    data.waypointPose = { point, source: structuredClone(sourceConstraint) };
    data.runtime.gizmo?.detach?.();
    if (data.runtime.gizmo?.getHelper?.()) data.runtime.gizmo.getHelper().visible = false;
    data.sceneGizmo?.detach();
    if (data.sceneGizmo?.getHelper()) data.sceneGizmo.getHelper().visible = false;
    if (data.poseGizmo?.getHelper()) data.poseGizmo.getHelper().visible = true;
    const select = $("kimodoWaypointPoseNode");
    if (select) {
      select.innerHTML = data.runtime.bones
        .map((bone, index) => `<option value="${index}">${options.escapeHtml(bone.name)}</option>`)
        .join("");
    }
    applyWaypointPose(overrideConstraint || sourceConstraint);
    selectWaypointPoseJoint(Math.min(data.poseJoint, data.runtime.bones.length - 1));
    $("kimodoWaypointPoseTools")?.classList.remove("hidden");
    return true;
  }

  function clearWaypointPose() {
    data.waypointPose = null;
    data.poseGizmo?.detach();
    if (data.poseGizmo?.getHelper()) data.poseGizmo.getHelper().visible = false;
    data.poseMarkers.forEach(marker => { marker.visible = false; });
    $("kimodoWaypointPoseTools")?.classList.add("hidden");
    if (data.activeTool === "waypoint-pose") data.activeTool = "trajectory";
  }

  function resetWaypointPose() {
    if (!data.waypointPose) return;
    applyWaypointPose(data.waypointPose.source);
    options.onWaypointPoseChange?.(null);
    status("已恢复姿势仓库中的原始姿势；仓库资产未被修改。", "Restored the original library pose; the library asset was not modified.");
  }

  function normalizeSceneObject(item) {
    const transform = item.transform || {};
    return {
      id: item.id || uid("scene"),
      name: String(item.name || item.asset_kind || "Scene Asset").slice(0, 80),
      asset_kind: item.asset_kind,
      source_asset_id: item.source_asset_id,
      source_revision: Number(item.source_revision) || 0,
      transform: {
        translation: (transform.translation || [0, 0, 0]).slice(0, 3).map(Number),
        quaternion: (transform.quaternion || [0, 0, 0, 1]).slice(0, 4).map(Number),
        uniform_scale: Number(transform.uniform_scale ?? transform.scale?.[0] ?? 1) || 1,
      },
      editor_hidden: Boolean(item.editor_hidden),
    };
  }

  function getPreviewScene() {
    return { scene_objects: data.objects.map(normalizeSceneObject) };
  }

  async function refreshCatalogs() {
    try { data.terrains = await options.api("/api/terrains"); }
    catch { data.terrains = []; }
    renderAssetOptions();
  }

  function renderAssetOptions() {
    const select = $("kimodoSceneAsset");
    if (!select) return;
    const rows = $("kimodoSceneKind")?.value === "terrain" ? data.terrains : (options.getStaticAssets?.() || []);
    select.innerHTML = rows.length
      ? rows.map(item => `<option value="${options.escapeHtml(item.id)}">${options.escapeHtml(item.name || item.id)}</option>`).join("")
      : `<option value="">${tr("暂无可用资产", "No assets available")}</option>`;
  }

  async function ensureAsset(kind, id) {
    const key = `${kind}:${id}`;
    if (data.assetCache.has(key)) return data.assetCache.get(key);
    const prefix = kind === "terrain" ? "/api/terrains" : "/api/characters";
    const scene = await options.api(`${prefix}/${encodeURIComponent(id)}/scene`);
    data.assetCache.set(key, scene);
    return scene;
  }

  function createAssetMesh(sceneData, kind) {
    const THREE = options.getThree();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute((sceneData.vertices || []).flat(), 3));
    geometry.setIndex((sceneData.faces || []).flat());
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: kind === "terrain" ? 0x3f8cff : 0xe4ff4e,
      roughness: kind === "terrain" ? 0.92 : 0.62,
      metalness: 0.02,
      side: THREE.DoubleSide,
      polygonOffset: kind === "terrain",
      polygonOffsetFactor: kind === "terrain" ? -1 : 0,
      polygonOffsetUnits: kind === "terrain" ? -1 : 0,
    });
    return new THREE.Mesh(geometry, material);
  }

  async function createSceneVisual(item) {
    if (!data.runtime) return;
    const THREE = options.getThree();
    const source = await ensureAsset(item.asset_kind, item.source_asset_id);
    const group = new THREE.Group();
    group.name = item.name;
    group.userData.kimodoSceneObjectId = item.id;
    const mesh = createAssetMesh(source, item.asset_kind);
    mesh.userData.kimodoSceneObjectId = item.id;
    group.add(mesh);
    data.runtime.scene.add(group);
    data.visuals.set(item.id, group);
    applySceneTransform(item, group);
  }

  function applySceneTransform(item, visual = data.visuals.get(item.id)) {
    if (!visual) return;
    visual.position.fromArray(item.transform.translation);
    visual.quaternion.fromArray(item.transform.quaternion).normalize();
    visual.scale.setScalar(item.transform.uniform_scale);
    visual.visible = !item.editor_hidden;
    visual.updateMatrixWorld(true);
  }

  function selectedObject() {
    return data.objects.find(item => item.id === data.selectedObjectId) || null;
  }

  function renderSceneList() {
    const box = $("kimodoSceneList");
    if (!box) return;
    box.innerHTML = data.objects.length
      ? data.objects.map((item, index) => `<button type="button" class="kimodo-scene-row ${item.id === data.selectedObjectId ? "selected" : ""}" data-scene-id="${options.escapeHtml(item.id)}"><strong>${index + 1}. ${options.escapeHtml(item.name)}</strong><span>${item.asset_kind === "terrain" ? tr("地形", "Terrain") : tr("静态网格体", "Static Mesh")}</span></button>`).join("")
      : `<p>${tr("预览场景为空", "Preview scene is empty")}</p>`;
    box.querySelectorAll("[data-scene-id]").forEach(button => {
      button.onclick = () => selectSceneObject(button.dataset.sceneId, true);
    });
  }

  function activateSceneMode() {
    if (!data.runtime) return;
    if (data.activeTool === "waypoint-pose") clearWaypointPose();
    data.activeTool = "scene";
    data.sceneActive = true;
    data.runtime.gizmo?.detach?.();
    if (data.runtime.gizmo?.getHelper?.()) data.runtime.gizmo.getHelper().visible = false;
    if (data.sceneGizmo?.getHelper()) data.sceneGizmo.getHelper().visible = true;
    const visual = data.visuals.get(data.selectedObjectId);
    if (visual) data.sceneGizmo?.attach(visual);
  }

  function deactivate() {
    if (data.activeTool === "waypoint-pose") clearWaypointPose();
    data.activeTool = "trajectory";
    data.sceneActive = false;
    data.sceneGizmo?.detach();
    if (data.sceneGizmo?.getHelper?.()) data.sceneGizmo.getHelper().visible = false;
    if (data.runtime?.gizmo?.getHelper?.()) data.runtime.gizmo.getHelper().visible = true;
  }

  function focusSceneObject(visual) {
    if (!visual || !data.runtime) return;
    const THREE = options.getThree();
    const bounds = new THREE.Box3().setFromObject(visual);
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(0.5, size.length() * 0.5);
    const halfFov = THREE.MathUtils.degToRad(data.runtime.camera.fov * 0.5);
    const distance = radius / Math.max(0.1, Math.sin(halfFov)) * 1.2;
    const direction = new THREE.Vector3(1, 0.72, 1).normalize();
    data.runtime.camera.position.copy(center).addScaledVector(direction, distance);
    data.runtime.camera.near = Math.max(0.01, distance / 1000);
    data.runtime.camera.far = Math.max(100, distance + radius * 10);
    data.runtime.camera.updateProjectionMatrix();
    data.runtime.controls.target.copy(center);
    data.runtime.controls.update();
  }

  function selectSceneObject(id, focus = false) {
    data.selectedObjectId = id;
    activateSceneMode();
    const visual = data.visuals.get(id);
    if (visual) data.sceneGizmo?.attach(visual);
    if (focus) focusSceneObject(visual);
    renderSceneList();
  }

  async function addSceneAsset() {
    const kind = $("kimodoSceneKind")?.value || "static_mesh";
    const id = $("kimodoSceneAsset")?.value;
    if (!id) return;
    const source = await ensureAsset(kind, id);
    const rows = kind === "terrain" ? data.terrains : (options.getStaticAssets?.() || []);
    const sourceRow = rows.find(item => item.id === id);
    const transform = { translation: [0, 0, 0], quaternion: [0, 0, 0, 1], uniform_scale: 1 };
    if (kind === "terrain" && source.transform) {
      const THREE = options.getThree();
      transform.translation = (source.transform.translation || [0, 0, 0]).slice(0, 3);
      transform.quaternion = new THREE.Quaternion()
        .setFromEuler(new THREE.Euler(0, THREE.MathUtils.degToRad(Number(source.transform.yaw_degrees) || 0), 0))
        .toArray();
      transform.uniform_scale = Number(source.transform.uniform_scale) || 1;
    }
    const item = normalizeSceneObject({
      id: uid("scene"),
      name: sourceRow?.name || tr("场景资产", "Scene Asset"),
      asset_kind: kind,
      source_asset_id: id,
      source_revision: sourceRow?.revision || 0,
      transform,
    });
    data.objects.push(item);
    await createSceneVisual(item);
    selectSceneObject(item.id, true);
    scheduleSaveScene();
    status("资产已添加到 Kimodo 预览场景。", "Asset added to the Kimodo preview scene.");
  }

  function syncSceneObjectFromVisual() {
    const item = selectedObject();
    const visual = item && data.visuals.get(item.id);
    if (!item || !visual) return;
    item.transform.translation = visual.position.toArray();
    item.transform.quaternion = visual.quaternion.normalize().toArray();
    item.transform.uniform_scale = visual.scale.x;
    renderSceneList();
  }

  function setSceneMode(mode) {
    data.sceneMode = ["translate", "rotate", "scale"].includes(mode) ? mode : "translate";
    ["Translate", "Rotate", "Scale"].forEach(label => {
      $("kimodoScene" + label)?.classList.toggle("active", data.sceneMode === label.toLowerCase());
    });
    data.sceneGizmo?.setMode(data.sceneMode);
    data.sceneGizmo?.setSpace(data.sceneMode === "rotate" ? "local" : "world");
    activateSceneMode();
  }

  function resetSceneObject() {
    const item = selectedObject();
    if (!item) return;
    item.transform = { translation: [0, 0, 0], quaternion: [0, 0, 0, 1], uniform_scale: 1 };
    applySceneTransform(item);
    scheduleSaveScene();
  }

  function deleteSceneObject() {
    const item = selectedObject();
    if (!item) return;
    data.sceneGizmo?.detach();
    disposeObject(data.visuals.get(item.id));
    data.visuals.delete(item.id);
    data.objects = data.objects.filter(entry => entry.id !== item.id);
    data.selectedObjectId = "";
    renderSceneList();
    scheduleSaveScene();
  }

  async function persistScene() {
    const id = options.getLoadedJobId?.();
    if (!id) return;
    try {
      await options.api(`/api/kimodo-motions/${encodeURIComponent(id)}/preview-scene`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getPreviewScene()),
      });
    } catch (error) {
      status(`场景保存失败：${error.message}`, `Scene save failed: ${error.message}`);
    }
  }

  function scheduleSaveScene() {
    clearTimeout(data.saveTimer);
    data.saveTimer = setTimeout(persistScene, 350);
  }

  function pickSceneObject(event) {
    if (!data.sceneActive || !data.runtime || data.sceneGizmo?.dragging) return;
    const THREE = options.getThree();
    const rect = data.runtime.canvas.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      (event.clientX - rect.left) / rect.width * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, data.runtime.camera);
    const hit = raycaster.intersectObjects([...data.visuals.values()], true)[0];
    const id = hit?.object?.userData?.kimodoSceneObjectId || hit?.object?.parent?.userData?.kimodoSceneObjectId;
    if (id) selectSceneObject(id);
  }

  function buildSceneGizmo() {
    const TransformControls = options.getTransformControls();
    if (!TransformControls || !data.runtime) return;
    data.sceneGizmo = new TransformControls(data.runtime.camera, data.runtime.canvas);
    data.sceneGizmo.setMode("translate");
    data.runtime.scene.add(data.sceneGizmo.getHelper());
    data.sceneGizmo.getHelper().visible = false;
    data.sceneGizmo.addEventListener("dragging-changed", event => {
      data.runtime.controls.enabled = !event.value;
      const visual = data.visuals.get(data.selectedObjectId);
      if (event.value && visual && data.sceneMode === "scale") data.scaleDrag = visual.scale.clone();
      if (!event.value) {
        syncSceneObjectFromVisual();
        scheduleSaveScene();
        data.scaleDrag = null;
      }
    });
    data.sceneGizmo.addEventListener("objectChange", () => {
      const visual = data.visuals.get(data.selectedObjectId);
      if (visual && data.sceneMode === "scale" && data.scaleDrag) {
        const ratios = [
          visual.scale.x / data.scaleDrag.x,
          visual.scale.y / data.scaleDrag.y,
          visual.scale.z / data.scaleDrag.z,
        ].filter(Number.isFinite);
        const factor = ratios.reduce(
          (best, value) => Math.abs(value - 1) > Math.abs(best - 1) ? value : best,
          1,
        );
        visual.scale.copy(data.scaleDrag).multiplyScalar(clamp(factor, 0.001, 1000));
      }
      syncSceneObjectFromVisual();
    });
    data.runtime.canvas.removeEventListener("pointerdown", pickSceneObject);
    data.runtime.canvas.addEventListener("pointerdown", pickSceneObject);
  }

  async function loadPreviewScene(scene = {}) {
    data.sceneGizmo?.detach();
    data.visuals.forEach(disposeObject);
    data.visuals.clear();
    data.objects = (scene.scene_objects || []).map(normalizeSceneObject);
    await Promise.all(data.objects.map(item => createSceneVisual(item).catch(error => {
      status(`资产加载失败：${error.message}`, `Asset load failed: ${error.message}`);
    })));
    data.selectedObjectId = "";
    renderSceneList();
  }

  function disposeRuntimeTools() {
    data.sceneGizmo?.dispose?.();
    data.sceneGizmo?.getHelper?.().removeFromParent();
    data.sceneGizmo = null;
    data.poseGizmo?.dispose?.();
    data.poseGizmo?.getHelper?.().removeFromParent();
    data.poseGizmo = null;
    data.poseMarkers.forEach(disposeObject);
    data.poseMarkers = [];
    data.visuals.forEach(disposeObject);
    data.visuals.clear();
  }

  function attachRuntime(runtime, sceneData) {
    if (!runtime || !sceneData?.names?.length) return;
    runtime.canvas = runtime.canvas || runtime.renderer?.domElement || $("kimodoCanvas");
    if (!runtime.canvas?.addEventListener) throw new Error("Kimodo viewport is not ready");
    disposeRuntimeTools();
    data.runtime = runtime;
    data.sceneData = sceneData;
    data.activeTool = "trajectory";
    data.sceneActive = false;
    buildSceneGizmo();
    buildWaypointPoseTools();
    if (data.runtime.gizmo?.getHelper?.()) data.runtime.gizmo.getHelper().visible = true;
    loadPreviewScene(sceneData.preview_scene || { scene_objects: data.objects }).catch(console.error);
  }

  function bindEvents() {
    $("kimodoWaypointPoseNode")?.addEventListener("change", event => selectWaypointPoseJoint(Number(event.target.value)));
    $("kimodoWaypointPoseTranslate")?.addEventListener("click", () => setWaypointPoseMode("translate"));
    $("kimodoWaypointPoseRotate")?.addEventListener("click", () => setWaypointPoseMode("rotate"));
    $("kimodoWaypointPoseReset")?.addEventListener("click", resetWaypointPose);
    $("kimodoSceneKind")?.addEventListener("change", renderAssetOptions);
    $("kimodoSceneAdd")?.addEventListener("click", () => addSceneAsset().catch(error => status(error.message, error.message)));
    $("kimodoSceneTranslate")?.addEventListener("click", () => setSceneMode("translate"));
    $("kimodoSceneRotate")?.addEventListener("click", () => setSceneMode("rotate"));
    $("kimodoSceneScale")?.addEventListener("click", () => setSceneMode("scale"));
    $("kimodoSceneReset")?.addEventListener("click", resetSceneObject);
    $("kimodoSceneDelete")?.addEventListener("click", deleteSceneObject);
    $("kimodoScenePanel")?.addEventListener("toggle", event => {
      if (event.currentTarget.open) activateSceneMode();
      else if (data.sceneActive) deactivate();
    });
    window.addEventListener("motionbuilder:languagechange", () => {
      renderSceneList();
      renderAssetOptions();
      if (data.statusPair) status(data.statusPair[0], data.statusPair[1]);
    });
  }

  bindEvents();
  renderSceneList();
  refreshCatalogs();

  return {
    attachRuntime,
    getPreviewScene,
    loadPreviewScene,
    refreshCatalogs,
    onFrameChanged() {},
    beginWaypointPose,
    clearWaypointPose,
    isPoseEditing: () => data.activeTool === "waypoint-pose",
    isEditing: () => data.sceneActive || data.activeTool === "waypoint-pose",
    deactivate,
  };
}
