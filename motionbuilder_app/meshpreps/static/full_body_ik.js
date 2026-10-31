import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const EPSILON = 1e-8;
const _jointPosition = new THREE.Vector3();
const _effectorPosition = new THREE.Vector3();
const _targetPosition = new THREE.Vector3();
const _currentDirection = new THREE.Vector3();
const _targetDirection = new THREE.Vector3();
const _worldDelta = new THREE.Vector3();
const _averagePull = new THREE.Vector3();
const _parentQuaternion = new THREE.Quaternion();
const _currentQuaternion = new THREE.Quaternion();
const _desiredQuaternion = new THREE.Quaternion();
const _deltaQuaternion = new THREE.Quaternion();
const _limitedDelta = new THREE.Quaternion();
const _identityQuaternion = new THREE.Quaternion();
const _rootQuaternion = new THREE.Quaternion();

function applyWorldRotationDelta(object, delta) {
  object.getWorldQuaternion(_currentQuaternion);
  _desiredQuaternion.copy(delta).multiply(_currentQuaternion);
  if (object.parent) object.parent.getWorldQuaternion(_parentQuaternion);
  else _parentQuaternion.identity();
  object.quaternion.copy(_parentQuaternion).invert().multiply(_desiredQuaternion).normalize();
  object.updateMatrix();
}

function applyWorldTranslation(object, delta) {
  _worldDelta.copy(delta);
  if (object.parent) {
    object.parent.getWorldQuaternion(_parentQuaternion);
    _worldDelta.applyQuaternion(_parentQuaternion.invert());
  }
  object.position.add(_worldDelta);
  object.updateMatrix();
}

function updateLine(line, from, to) {
  const position = line.geometry.attributes.position;
  position.setXYZ(0, from.x, from.y, from.z);
  position.setXYZ(1, to.x, to.y, to.z);
  position.needsUpdate = true;
  line.computeLineDistances();
}

export class FullBodyIKController {
  constructor({ scene, camera, canvas, orbitControls, rigController, onSelect }) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
    this.orbitControls = orbitControls;
    this.rig = rigController;
    this.onSelect = onSelect;
    this.profile = null;
    this.fullBody = null;
    this.enabled = false;
    this.selected = null;
    this.controls = [];
    this.pickables = [];
    this.views = new Map();
    this.inputPose = null;
    this.boneSettings = new Map();

    this.root = new THREE.Group();
    this.root.name = "fullBodyIKControls";
    this.root.visible = false;
    scene.add(this.root);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.transformControls = new TransformControls(camera, canvas);
    this.transformControls.setMode("translate");
    this.transformControls.setSpace("world");
    this.transformControls.setSize(0.72);
    scene.add(this.transformControls.getHelper());
    this.transformControls.getHelper().visible = false;
    this.transformControls.addEventListener("dragging-changed", (event) => {
      this.orbitControls.enabled = !event.value;
    });
    this.transformControls.addEventListener("objectChange", () => {
      this.solve();
      this._updateGuides();
    });

    this._onPointerDown = this._onPointerDown.bind(this);
    canvas.addEventListener("pointerdown", this._onPointerDown);
  }

  setCamera(camera) {
    this.camera = camera;
    this.transformControls.camera = camera;
  }

  setTransformMode(mode) {
    const resolved = mode === "rotate" ? "rotate" : "translate";
    this.transformControls.setMode(resolved);
    return resolved;
  }

  setTransformSpace(space) {
    const resolved = space === "local" ? "local" : "world";
    this.transformControls.setSpace(resolved);
    return resolved;
  }

  setRootBehavior(value) {
    if (!this.fullBody) return;
    const allowed = new Set(["pre_pull", "pin_to_input", "free"]);
    this.fullBody.settings.root_behavior = allowed.has(value) ? value : "pre_pull";
    this.solve();
  }

  setIterations(value) {
    if (!this.fullBody) return;
    this.fullBody.settings.iterations = THREE.MathUtils.clamp(Math.round(Number(value) || 20), 1, 60);
    this.solve();
  }

  clear() {
    this.transformControls.detach();
    this.selected = null;
    this.profile = null;
    this.fullBody = null;
    this.inputPose = null;
    this.controls = [];
    this.pickables = [];
    this.views.clear();
    this.boneSettings.clear();
    while (this.root.children.length) {
      const child = this.root.children.pop();
      child.traverse((object) => {
        object.geometry?.dispose?.();
        object.material?.dispose?.();
      });
    }
    this.onSelect?.(null);
  }

  dispose() {
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.setEnabled(false);
    this.clear();
    this.scene.remove(this.transformControls.getHelper());
    this.transformControls.dispose();
    this.scene.remove(this.root);
  }

  load(profile) {
    this.clear();
    const fullBody = profile?.full_body || profile?.solvers?.full_body;
    if (!fullBody || fullBody.solver !== "FullBodyIK" || !Array.isArray(fullBody.effectors)) {
      throw new Error("Full Body IK profile is invalid");
    }
    this.profile = profile;
    this.fullBody = structuredClone(fullBody);
    for (const setting of this.fullBody.bone_settings || []) {
      this.boneSettings.set(Number(setting.joint), setting);
    }
    const scale = Math.max(Number(profile.control_scale) || 0.035, 0.01);
    for (const effector of this.fullBody.effectors) this._buildEffector(effector, scale);
    this.captureInputPose();
    this._updateGuides();
    if (this.controls.length) this.selectControl(this.controls[0]);
  }

  _buildEffector(effector, scale) {
    const color = new THREE.Color(effector.color || "#e4ff4e");
    const geometry = effector.id === "head"
      ? new THREE.BoxGeometry(scale * 1.35, scale * 1.35, scale * 1.35)
      : new THREE.OctahedronGeometry(scale, 0);
    const control = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.96 }),
    );
    control.name = effector.control;
    control.position.fromArray(effector.position);
    control.quaternion.fromArray(effector.rotation || [0, 0, 0, 1]);
    control.renderOrder = 1320;
    control.userData.ikControlType = "effector";
    control.userData.ikControlName = effector.control;
    control.userData.ikEffectorId = effector.id;

    const guide = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineDashedMaterial({
        color,
        dashSize: scale * 0.8,
        gapSize: scale * 0.45,
        depthTest: false,
        transparent: true,
        opacity: 0.64,
      }),
    );
    guide.renderOrder = 1310;
    this.root.add(control, guide);
    this.controls.push(control);
    this.pickables.push(control);
    this.views.set(effector.id, { effector, control, guide });
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled) && Boolean(this.fullBody);
    this.root.visible = this.enabled;
    this.rig.setInteractionEnabled(!this.enabled);
    this.transformControls.getHelper().visible = this.enabled;
    if (this.enabled && this.selected) this.transformControls.attach(this.selected);
    else this.transformControls.detach();
    this._updateGuides();
  }

  selectControl(control) {
    if (!control || !this.controls.includes(control)) return;
    for (const item of this.controls) item.scale.setScalar(item === control ? 1.22 : 1.0);
    this.selected = control;
    if (this.enabled) this.transformControls.attach(control);
    const view = this.views.get(control.userData.ikEffectorId);
    this.onSelect?.({
      name: control.userData.ikControlName,
      type: "effector",
      chain: view?.effector?.bone || control.userData.ikEffectorId,
      effectorId: control.userData.ikEffectorId,
      jointName: view?.effector?.bone || "",
      control,
      lockable: true,
    });
  }

  captureInputPose() {
    this.inputPose = this.rig.jointObjs.map((object) => ({
      position: object.position.clone(),
      quaternion: object.quaternion.clone(),
      scale: object.scale.clone(),
    }));
  }

  _restoreInputPose() {
    if (!this.inputPose || this.inputPose.length !== this.rig.jointObjs.length) return;
    for (let index = 0; index < this.inputPose.length; index += 1) {
      const object = this.rig.jointObjs[index];
      const pose = this.inputPose[index];
      object.position.copy(pose.position);
      object.quaternion.copy(pose.quaternion);
      object.scale.copy(pose.scale);
      object.updateMatrix();
    }
    this.rig.root.updateWorldMatrix(true, true);
  }

  syncControlsFromPose({ capture = true } = {}) {
    if (!this.fullBody) return;
    this.rig.root.updateWorldMatrix(true, true);
    this.root.updateWorldMatrix(true, false);
    this.root.getWorldQuaternion(_rootQuaternion);
    for (const view of this.views.values()) {
      const object = this.rig.jointObjs[Number(view.effector.joint)];
      if (!object) continue;
      object.getWorldPosition(_targetPosition);
      view.control.position.copy(this.root.worldToLocal(_targetPosition.clone()));
      object.getWorldQuaternion(_desiredQuaternion);
      view.control.quaternion.copy(_rootQuaternion).invert().multiply(_desiredQuaternion).normalize();
    }
    if (capture) this.captureInputPose();
    this._updateGuides();
  }

  reset() {
    if (!this.fullBody) return;
    this.rig.resetPose();
    for (const view of this.views.values()) {
      view.control.position.fromArray(view.effector.position);
      view.control.quaternion.fromArray(view.effector.rotation || [0, 0, 0, 1]);
    }
    this.captureInputPose();
    this._updateGuides();
  }

  _chainToRoot(effectorIndex) {
    const chain = [];
    const rootIndex = Number(this.fullBody.root_index);
    const excluded = new Set(this.fullBody.excluded_joint_indices || []);
    let current = Number(this.rig.parents[effectorIndex]);
    while (current >= 0) {
      if (!excluded.has(current)) chain.push(current);
      if (current === rootIndex) break;
      current = Number(this.rig.parents[current]);
    }
    return chain;
  }

  _prePullRoot() {
    const behavior = this.fullBody.settings?.root_behavior || "pre_pull";
    if (behavior === "pin_to_input") return;
    const rootObject = this.rig.jointObjs[Number(this.fullBody.root_index)];
    if (!rootObject || !this.views.size) return;
    _averagePull.set(0, 0, 0);
    let totalWeight = 0;
    for (const view of this.views.values()) {
      const endObject = this.rig.jointObjs[Number(view.effector.joint)];
      if (!endObject) continue;
      endObject.getWorldPosition(_effectorPosition);
      view.control.getWorldPosition(_targetPosition);
      const strength = Number(view.effector.strength_alpha ?? 1);
      _averagePull.addScaledVector(_targetPosition.sub(_effectorPosition), strength);
      totalWeight += strength;
    }
    if (totalWeight <= EPSILON) return;
    const alpha = behavior === "free" ? 0.85 : 0.35;
    _averagePull.multiplyScalar(alpha / totalWeight);
    applyWorldTranslation(rootObject, _averagePull);
    this.rig.root.updateWorldMatrix(true, true);
  }

  solve() {
    if (!this.fullBody || !this.inputPose) return;
    if (this.fullBody.settings?.start_from_input_pose !== false) this._restoreInputPose();
    this._prePullRoot();
    const rootBehavior = this.fullBody.settings?.root_behavior || "pre_pull";
    const rootIndex = Number(this.fullBody.root_index);
    const maxAngle = THREE.MathUtils.degToRad(Number(this.fullBody.settings?.max_angle ?? 30));
    const iterations = THREE.MathUtils.clamp(Math.round(Number(this.fullBody.settings?.iterations) || 20), 1, 60);
    const views = [...this.views.values()].sort((left, right) => {
      const priority = (view) => view.effector.id.includes("foot") ? 0 : view.effector.id.includes("hand") ? 1 : 2;
      return priority(left) - priority(right);
    });

    for (let iteration = 0; iteration < iterations; iteration += 1) {
      for (const view of views) {
        const effectorIndex = Number(view.effector.joint);
        const endObject = this.rig.jointObjs[effectorIndex];
        if (!endObject) continue;
        view.control.getWorldPosition(_targetPosition);
        const chain = this._chainToRoot(effectorIndex);
        for (const jointIndex of chain) {
          if (jointIndex === rootIndex && rootBehavior === "pin_to_input") continue;
          const jointObject = this.rig.jointObjs[jointIndex];
          jointObject.getWorldPosition(_jointPosition);
          endObject.getWorldPosition(_effectorPosition);
          _currentDirection.subVectors(_effectorPosition, _jointPosition);
          _targetDirection.subVectors(_targetPosition, _jointPosition);
          if (_currentDirection.lengthSq() <= EPSILON || _targetDirection.lengthSq() <= EPSILON) continue;
          _currentDirection.normalize();
          _targetDirection.normalize();
          _deltaQuaternion.setFromUnitVectors(_currentDirection, _targetDirection);
          const setting = this.boneSettings.get(jointIndex);
          const stiffness = THREE.MathUtils.clamp(Number(setting?.rotation_stiffness ?? 0.15), 0, 1);
          const strength = THREE.MathUtils.clamp(Number(view.effector.strength_alpha ?? 1), 0, 1);
          const pull = THREE.MathUtils.clamp(Number(view.effector.pull_chain_alpha ?? 1), 0, 1);
          const angle = _identityQuaternion.angleTo(_deltaQuaternion);
          const angleLimit = Math.min(angle, maxAngle) / Math.max(angle, EPSILON);
          _limitedDelta.identity().slerp(_deltaQuaternion, (1 - stiffness) * strength * pull * angleLimit);
          applyWorldRotationDelta(jointObject, _limitedDelta);
          this.rig.root.updateWorldMatrix(true, true);
        }
      }
    }

    this.rig.root.updateWorldMatrix(true, true);
    for (const view of views) {
      const endObject = this.rig.jointObjs[Number(view.effector.joint)];
      if (!endObject) continue;
      view.control.getWorldQuaternion(_desiredQuaternion);
      if (endObject.parent) endObject.parent.getWorldQuaternion(_parentQuaternion);
      else _parentQuaternion.identity();
      _currentQuaternion.copy(_parentQuaternion).invert().multiply(_desiredQuaternion).normalize();
      const rotationAlpha = THREE.MathUtils.clamp(Number(view.effector.rotation_alpha ?? 1), 0, 1);
      endObject.quaternion.slerp(_currentQuaternion, rotationAlpha).normalize();
      endObject.updateMatrix();
    }
    this.rig.updatePosePreview();
  }

  _updateGuides() {
    if (!this.fullBody) return;
    this.rig.root.updateWorldMatrix(true, true);
    for (const view of this.views.values()) {
      const object = this.rig.jointObjs[Number(view.effector.joint)];
      if (!object) continue;
      object.getWorldPosition(_effectorPosition);
      view.control.getWorldPosition(_targetPosition);
      updateLine(view.guide, _effectorPosition, _targetPosition);
    }
  }

  _onPointerDown(event) {
    if (!this.enabled || this.transformControls.dragging) return;
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    if (!hits.length) return;
    event.preventDefault();
    event.stopPropagation();
    this.selectControl(hits[0].object);
  }
}
