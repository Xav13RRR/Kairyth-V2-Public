import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const EPSILON = 1e-7;
const _start = new THREE.Vector3();
const _middle = new THREE.Vector3();
const _end = new THREE.Vector3();
const _target = new THREE.Vector3();
const _pole = new THREE.Vector3();
const _direction = new THREE.Vector3();
const _bend = new THREE.Vector3();
const _desiredMiddle = new THREE.Vector3();
const _desiredEnd = new THREE.Vector3();
const _currentDirection = new THREE.Vector3();
const _desiredDirection = new THREE.Vector3();
const _parentWorldQuaternion = new THREE.Quaternion();
const _currentWorldQuaternion = new THREE.Quaternion();
const _desiredWorldQuaternion = new THREE.Quaternion();
const _rotationDelta = new THREE.Quaternion();
const _targetWorldQuaternion = new THREE.Quaternion();
const _rootWorldQuaternion = new THREE.Quaternion();
const _localTargetQuaternion = new THREE.Quaternion();

function safeNormalize(vector, fallback) {
  if (vector.lengthSq() > EPSILON * EPSILON) return vector.normalize();
  return vector.copy(fallback).normalize();
}

function applyWorldRotationDelta(object, delta) {
  object.getWorldQuaternion(_currentWorldQuaternion);
  _desiredWorldQuaternion.copy(delta).multiply(_currentWorldQuaternion);
  if (object.parent) object.parent.getWorldQuaternion(_parentWorldQuaternion);
  else _parentWorldQuaternion.identity();
  object.quaternion
    .copy(_parentWorldQuaternion)
    .invert()
    .multiply(_desiredWorldQuaternion)
    .normalize();
  object.updateMatrix();
}

function updateLine(line, from, to) {
  const position = line.geometry.attributes.position;
  position.setXYZ(0, from.x, from.y, from.z);
  position.setXYZ(1, to.x, to.y, to.z);
  position.needsUpdate = true;
  line.computeLineDistances();
}

export class TwoBoneIKController {
  constructor({ scene, camera, canvas, orbitControls, rigController, onSelect }) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
    this.orbitControls = orbitControls;
    this.rig = rigController;
    this.onSelect = onSelect;
    this.profile = null;
    this.enabled = false;
    this.selected = null;
    this.controls = [];
    this.pickables = [];
    this.chainViews = new Map();

    this.root = new THREE.Group();
    this.root.name = "twoBoneIKControls";
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
      const chainId = this.selected?.userData?.ikChainId;
      if (!chainId) return;
      this.solveChain(chainId);
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
    const requested = mode === "rotate" ? "rotate" : "translate";
    const resolved = this.selected?.userData?.ikControlType === "pole" && requested === "rotate"
      ? "translate"
      : requested;
    this.transformControls.setMode(resolved);
    return resolved;
  }

  setTransformSpace(space) {
    const resolved = space === "local" ? "local" : "world";
    this.transformControls.setSpace(resolved);
    return resolved;
  }

  get selectedControlType() {
    return this.selected?.userData?.ikControlType || null;
  }

  clear() {
    this.transformControls.detach();
    this.selected = null;
    this.profile = null;
    this.controls = [];
    this.pickables = [];
    this.chainViews.clear();
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
    if (!profile || profile.solver !== "TwoBoneIK" || !Array.isArray(profile.chains)) {
      throw new Error("Two-Bone IK profile is invalid");
    }
    this.profile = profile;
    const scale = Math.max(Number(profile.control_scale) || 0.035, 0.01);
    for (const chain of profile.chains) this._buildChain(chain, scale);
    this._updateGuides();
    if (this.controls.length) this.selectControl(this.controls[0]);
  }

  _buildChain(chain, scale) {
    const color = new THREE.Color(chain.color || "#e4ff4e");
    const goal = new THREE.Mesh(
      new THREE.OctahedronGeometry(scale, 0),
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.96 }),
    );
    goal.name = chain.goal.name;
    goal.position.fromArray(chain.goal.position);
    goal.renderOrder = 1300;
    if (chain.goal.rotation) goal.quaternion.fromArray(chain.goal.rotation);
    goal.userData.ikChainId = chain.id;
    goal.userData.ikControlType = "goal";
    goal.userData.ikControlName = chain.goal.name;

    const pole = new THREE.Mesh(
      new THREE.SphereGeometry(scale * 0.55, 14, 10),
      new THREE.MeshBasicMaterial({ color, wireframe: true, depthTest: false, transparent: true, opacity: 0.92 }),
    );
    pole.name = chain.pole.name;
    pole.position.fromArray(chain.pole.position);
    pole.renderOrder = 1300;
    pole.userData.ikChainId = chain.id;
    pole.userData.ikControlType = "pole";
    pole.userData.ikControlName = chain.pole.name;

    const guideMaterial = new THREE.LineDashedMaterial({
      color,
      dashSize: scale * 0.8,
      gapSize: scale * 0.45,
      depthTest: false,
      transparent: true,
      opacity: 0.62,
    });
    const goalGuide = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      guideMaterial.clone(),
    );
    const poleGuide = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      guideMaterial.clone(),
    );
    goalGuide.renderOrder = 1290;
    poleGuide.renderOrder = 1290;

    this.root.add(goal, pole, goalGuide, poleGuide);
    this.controls.push(goal, pole);
    this.pickables.push(goal, pole);
    this.chainViews.set(chain.id, { chain, goal, pole, goalGuide, poleGuide });
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled) && Boolean(this.profile);
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
    if (control.userData.ikControlType === "pole" && this.transformControls.mode === "rotate") this.setTransformMode("translate");
    const view = this.chainViews.get(control.userData.ikChainId);
    this.onSelect?.({
      name: control.userData.ikControlName,
      type: control.userData.ikControlType,
      chain: view?.chain?.label || control.userData.ikChainId,
      chainId: control.userData.ikChainId,
      effectorId: view?.chain?.id || control.userData.ikChainId,
      jointName: view?.chain?.joint_names?.[2] || "",
      control,
      lockable: control.userData.ikControlType === "goal",
    });
  }

  reset() {
    if (!this.profile) return;
    this.rig.resetPose();
    for (const view of this.chainViews.values()) {
      view.goal.position.fromArray(view.chain.goal.position);
      view.pole.position.fromArray(view.chain.pole.position);
      view.goal.quaternion.fromArray(view.chain.goal.rotation || [0, 0, 0, 1]);
    }
    this._updateGuides();
  }

  syncControlsFromPose() {
    if (!this.profile) return;
    this.rig.root.updateWorldMatrix(true, true);
    this.root.updateWorldMatrix(true, false);
    this.root.getWorldQuaternion(_rootWorldQuaternion);
    for (const view of this.chainViews.values()) {
      const [startIndex, middleIndex, endIndex] = view.chain.joints;
      const startObject = this.rig.jointObjs[startIndex];
      const middleObject = this.rig.jointObjs[middleIndex];
      const endObject = this.rig.jointObjs[endIndex];
      if (!startObject || !middleObject || !endObject) continue;
      startObject.getWorldPosition(_start);
      middleObject.getWorldPosition(_middle);
      endObject.getWorldPosition(_end);
      view.goal.position.copy(this.root.worldToLocal(_end.clone()));
      endObject.getWorldQuaternion(_targetWorldQuaternion);
      view.goal.quaternion.copy(_rootWorldQuaternion).invert().multiply(_targetWorldQuaternion).normalize();

      _direction.subVectors(_end, _start);
      safeNormalize(_direction, new THREE.Vector3(0, -1, 0));
      const projected = _start.clone().addScaledVector(_direction, _middle.clone().sub(_start).dot(_direction));
      _bend.subVectors(_middle, projected);
      if (_bend.lengthSq() <= EPSILON * EPSILON && view.chain.preferred_bend) _bend.fromArray(view.chain.preferred_bend);
      safeNormalize(_bend, new THREE.Vector3(0, 0, 1));
      const chainLength = Number(view.chain.bone_lengths?.[0] || 0) + Number(view.chain.bone_lengths?.[1] || 0);
      const poleWorld = _middle.clone().addScaledVector(_bend, Math.max(chainLength * 0.6, 0.05));
      view.pole.position.copy(this.root.worldToLocal(poleWorld));
    }
    this._updateGuides();
  }

  solveChain(chainId) {
    const view = this.chainViews.get(chainId);
    if (!view) return;
    const chain = view.chain;
    const [startIndex, middleIndex, endIndex] = chain.joints;
    const startObject = this.rig.jointObjs[startIndex];
    const middleObject = this.rig.jointObjs[middleIndex];
    const endObject = this.rig.jointObjs[endIndex];
    if (!startObject || !middleObject || !endObject) return;

    this.rig.root.updateWorldMatrix(true, true);
    startObject.getWorldPosition(_start);
    middleObject.getWorldPosition(_middle);
    endObject.getWorldPosition(_end);
    view.goal.getWorldPosition(_target);
    view.pole.getWorldPosition(_pole);

    const upperLength = Number(chain.bone_lengths?.[0]) || _start.distanceTo(_middle);
    const lowerLength = Number(chain.bone_lengths?.[1]) || _middle.distanceTo(_end);
    _direction.subVectors(_target, _start);
    const requestedDistance = _direction.length();
    safeNormalize(_direction, _end.clone().sub(_start));
    const minimumReach = Math.abs(upperLength - lowerLength) + EPSILON;
    const maximumReach = Math.max(upperLength + lowerLength - EPSILON, minimumReach);
    const solvedDistance = THREE.MathUtils.clamp(requestedDistance, minimumReach, maximumReach);
    _desiredEnd.copy(_start).addScaledVector(_direction, solvedDistance);

    _bend.subVectors(_pole, _start);
    _bend.addScaledVector(_direction, -_bend.dot(_direction));
    if (_bend.lengthSq() <= EPSILON * EPSILON && chain.preferred_bend) {
      _bend.fromArray(chain.preferred_bend);
      _bend.addScaledVector(_direction, -_bend.dot(_direction));
    }
    if (_bend.lengthSq() <= EPSILON * EPSILON) {
      _bend.set(0, 0, 1).addScaledVector(_direction, -_direction.z);
    }
    safeNormalize(_bend, new THREE.Vector3(0, 1, 0));

    const along = (
      upperLength * upperLength
      - lowerLength * lowerLength
      + solvedDistance * solvedDistance
    ) / (2 * solvedDistance);
    const height = Math.sqrt(Math.max(upperLength * upperLength - along * along, 0));
    _desiredMiddle.copy(_start).addScaledVector(_direction, along).addScaledVector(_bend, height);

    startObject.getWorldPosition(_start);
    middleObject.getWorldPosition(_middle);
    safeNormalize(_currentDirection.subVectors(_middle, _start), _direction);
    safeNormalize(_desiredDirection.subVectors(_desiredMiddle, _start), _direction);
    _rotationDelta.setFromUnitVectors(_currentDirection, _desiredDirection);
    applyWorldRotationDelta(startObject, _rotationDelta);

    this.rig.root.updateWorldMatrix(true, true);
    middleObject.getWorldPosition(_middle);
    endObject.getWorldPosition(_end);
    safeNormalize(_currentDirection.subVectors(_end, _middle), _direction);
    safeNormalize(_desiredDirection.subVectors(_desiredEnd, _middle), _direction);
    _rotationDelta.setFromUnitVectors(_currentDirection, _desiredDirection);
    applyWorldRotationDelta(middleObject, _rotationDelta);

    this.rig.root.updateWorldMatrix(true, true);
    view.goal.getWorldQuaternion(_targetWorldQuaternion);
    if (endObject.parent) endObject.parent.getWorldQuaternion(_parentWorldQuaternion);
    else _parentWorldQuaternion.identity();
    _localTargetQuaternion.copy(_parentWorldQuaternion).invert().multiply(_targetWorldQuaternion).normalize();
    const rotationAlpha = THREE.MathUtils.clamp(Number(chain.goal.rotation_alpha ?? 1), 0, 1);
    endObject.quaternion.slerp(_localTargetQuaternion, rotationAlpha).normalize();
    endObject.updateMatrix();
    this.rig.updatePosePreview();
  }

  _updateGuides() {
    if (!this.profile) return;
    this.rig.root.updateWorldMatrix(true, true);
    for (const view of this.chainViews.values()) {
      const middleIndex = view.chain.joints[1];
      const endIndex = view.chain.joints[2];
      this.rig.jointObjs[middleIndex]?.getWorldPosition(_middle);
      this.rig.jointObjs[endIndex]?.getWorldPosition(_end);
      view.goal.getWorldPosition(_target);
      view.pole.getWorldPosition(_pole);
      updateLine(view.goalGuide, _end, _target);
      updateLine(view.poleGuide, _middle, _pole);
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
