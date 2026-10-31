import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _m0 = new THREE.Matrix4();
const _m1 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _unitZ = new THREE.Vector3(0, 0, 1);
const _brushNormal = new THREE.Vector3();
const _brushPoint = new THREE.Vector3();

function mat4FromRows(flat) {
  const m = new THREE.Matrix4();
  m.set(
    flat[0], flat[1], flat[2], flat[3],
    flat[4], flat[5], flat[6], flat[7],
    flat[8], flat[9], flat[10], flat[11],
    flat[12], flat[13], flat[14], flat[15],
  );
  return m;
}

function matrix4ToFlatRows(m) {
  const e = m.elements;
  return [
    e[0], e[4], e[8], e[12],
    e[1], e[5], e[9], e[13],
    e[2], e[6], e[10], e[14],
    e[3], e[7], e[11], e[15],
  ];
}

function applyAbsoluteBindPose(jointObjs, parents, bindRowsFlat) {
  for (let j = 0; j < jointObjs.length; j += 1) {
    const world = mat4FromRows(bindRowsFlat[j]);
    const p = parents[j];
    let local;
    if (p < 0) {
      local = world;
    } else {
      jointObjs[p].updateWorldMatrix(true, false);
      local = _m1.copy(jointObjs[p].matrixWorld).invert().multiply(world);
    }
    local.decompose(_pos, _quat, _scale);
    jointObjs[j].position.copy(_pos);
    jointObjs[j].quaternion.copy(_quat);
    jointObjs[j].scale.copy(_scale);
    jointObjs[j].updateMatrix();
  }
}

export class RigController {
  constructor({
    scene,
    camera,
    canvas,
    orbitControls,
    meshGroup,
    onSelect,
  }) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
    this.orbitControls = orbitControls;
    this.meshGroup = meshGroup;
    this.onSelect = onSelect;

    this.root = new THREE.Group();
    this.root.name = "rigRoot";
    scene.add(this.root);

    this.boneLines = new THREE.Group();
    this.root.add(this.boneLines);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pickables = [];

    this.mesh = null;
    this.meshMaterial = null;
    this.restPositions = null;
    this.lbsIndices = null;
    this.lbsWeights = null;
    this.bindInv = null;
    this.skinMats = [];
    this.jointObjs = [];
    this.restLocals = [];
    this.parents = [];
    this.names = [];
    this.selectedIndex = -1;
    this._lbsPending = false;
    this.heatmapJointIndex = -1;
    this.weightTool = null;
    this.interactionEnabled = true;
    this.brushRadiusScale = 0.15;
    this.brushStrength = 0.25;
    this._brushStroke = false;
    this._lastBrushHit = null;
    this._baseMeshColor = 0x8ab4f8;
    this.brushPreview = this._createBrushPreview();
    scene.add(this.brushPreview);

    this.transformControls = new TransformControls(camera, canvas);
    this.transformControls.setMode("translate");
    this.transformControls.setSize(0.85);
    this.scene.add(this.transformControls.getHelper());

    this.transformControls.addEventListener("dragging-changed", (ev) => {
      this.orbitControls.enabled = !ev.value;
    });
    this.transformControls.addEventListener("change", () => {
      this._updateBoneLines();
      this._scheduleLbs();
    });

    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    canvas.addEventListener("pointerdown", this._onPointerDown);
    canvas.addEventListener("pointermove", this._onPointerMove);
    canvas.addEventListener("pointerup", this._onPointerUp);
    canvas.addEventListener("pointerleave", this._onPointerUp);
  }

  setCamera(camera) {
    this.camera = camera;
    this.transformControls.camera = camera;
  }

  setInteractionEnabled(enabled) {
    this.interactionEnabled = Boolean(enabled);
    this.transformControls.enabled = this.interactionEnabled;
    if (!this.interactionEnabled) {
      this.setWeightTool(null);
      this.selectJoint(-1);
    }
  }

  updatePosePreview() {
    this.root.updateWorldMatrix(true, true);
    this._updateBoneLines();
    this._applyLbs();
  }

  dispose() {
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.canvas.removeEventListener("pointermove", this._onPointerMove);
    this.canvas.removeEventListener("pointerup", this._onPointerUp);
    this.canvas.removeEventListener("pointerleave", this._onPointerUp);
    this.transformControls.detach();
    this.scene.remove(this.transformControls.getHelper());
    this.transformControls.dispose();
    this.clear();
    this.scene.remove(this.brushPreview);
    this.brushPreview.geometry.dispose();
    this.brushPreview.material.dispose();
    this.scene.remove(this.root);
  }

  clear() {
    this._hideBrushPreview();
    this.transformControls.detach();
    this.pickables = [];
    this.jointObjs = [];
    this.selectedIndex = -1;
    this.heatmapJointIndex = -1;
    this.weightTool = null;
    this._brushStroke = false;
    this.lbsIndices = null;
    this.lbsWeights = null;
    this.bindInv = null;
    this.skinMats = [];
    while (this.root.children.length > 1) {
      const c = this.root.children.pop();
      if (c !== this.boneLines) c.removeFromParent();
    }
    while (this.boneLines.children.length) {
      const c = this.boneLines.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
    this.mesh = null;
    this.meshMaterial = null;
    this.restPositions = null;
  }

  _createBrushPreview() {
    const points = [];
    const segments = 96;
    for (let i = 0; i < segments; i += 1) {
      const angle = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0));
    }
    const ring = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({
        color: 0x35ff77,
        transparent: true,
        opacity: 0.85,
        depthTest: false,
        depthWrite: false,
      }),
    );
    ring.name = "weightBrushPreview";
    ring.visible = false;
    ring.renderOrder = 1200;
    return ring;
  }

  _hideBrushPreview() {
    if (this.brushPreview) this.brushPreview.visible = false;
    this._lastBrushHit = null;
  }

  loadSkeletonOnly(data, { mesh, material, interactive = true }) {
    this.clear();
    this.mesh = mesh;
    this.meshMaterial = material;
    this.parents = data.parents;
    this.names = data.names || [];
    this._buildJointHierarchy(data.joints, {
      jointColor: 0xffcc00,
      lineColor: 0xffaa00,
      interactive,
    });
  }

  _buildJointHierarchy(joints, { jointColor, lineColor, interactive }) {
    const worldPos = joints.map((p) => new THREE.Vector3(...p));
    this.restLocals = worldPos.map((pos, i) => {
      const p = this.parents[i];
      if (p < 0) return pos.clone();
      return pos.clone().sub(worldPos[p]);
    });

    const refMesh = this.mesh?.geometry?.boundingBox
      ? this.mesh
      : null;
    let jointRadius = 0.008;
    if (refMesh) {
      refMesh.geometry.computeBoundingBox();
      const size = refMesh.geometry.boundingBox.getSize(new THREE.Vector3()).length();
      jointRadius = Math.max(size * 0.012, 0.004);
    }
    const pickRadius = jointRadius * 2.2;
    const sphereGeo = new THREE.SphereGeometry(jointRadius, 10, 10);
    const pickGeo = new THREE.SphereGeometry(pickRadius, 8, 8);

    const nJoints = joints.length;
    for (let i = 0; i < nJoints; i++) {
      const obj = new THREE.Object3D();
      obj.userData.jointIndex = i;
      this.jointObjs.push(obj);
    }

    for (let i = 0; i < nJoints; i++) {
      const p = this.parents[i];
      if (p >= 0) this.jointObjs[p].add(this.jointObjs[i]);
      else this.root.add(this.jointObjs[i]);
      this.jointObjs[i].position.copy(this.restLocals[i]);
    }

    for (let i = 0; i < nJoints; i++) {
      const normalMat = new THREE.MeshBasicMaterial({ color: jointColor, depthTest: true });
      const visual = new THREE.Mesh(sphereGeo, normalMat);
      visual.userData.jointIndex = i;
      visual.userData.isVisual = true;
      this.jointObjs[i].add(visual);

      if (interactive) {
        const pickMat = new THREE.MeshBasicMaterial({
          color: jointColor,
          transparent: true,
          opacity: 0.001,
          depthWrite: false,
        });
        const picker = new THREE.Mesh(pickGeo, pickMat);
        picker.userData.jointIndex = i;
        picker.userData.isPicker = true;
        this.jointObjs[i].add(picker);
        this.pickables.push(picker);
      }
    }

    this._buildBoneLines(lineColor);
    this.root.updateWorldMatrix(true, true);
  }

  loadBinding(data, { mesh, material }) {
    this.clear();
    this.mesh = mesh;
    this.meshMaterial = material;
    this.parents = data.parents;
    this.names = data.names || [];

    const flatVerts = data.vertices.flat();
    this.restPositions = new Float32Array(flatVerts);
    mesh.geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.restPositions.slice(), 3),
    );
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();

    const nJoints = data.joints.length;
    this.skinMats = new Array(nJoints);
    for (let j = 0; j < nJoints; j++) {
      this.skinMats[j] = new THREE.Matrix4();
    }

    const idxFlat = data.lbs_indices.flat();
    const wFlat = data.lbs_weights.flat();
    this.lbsIndices = new Int32Array(idxFlat);
    this.lbsWeights = new Float32Array(wFlat);

    this._buildJointHierarchy(data.joints, {
      jointColor: 0xffcc00,
      lineColor: 0xffaa00,
      interactive: true,
    });
    if (data.bind_matrices?.length) {
      this.bindInv = data.bind_matrices.map((row) => mat4FromRows(row).invert());
      applyAbsoluteBindPose(this.jointObjs, this.parents, data.bind_matrices);
    } else {
      this._recomputeBindInvFromHierarchy();
    }
    this.root.updateWorldMatrix(true, true);
    this._syncBrushRadiusFromMesh();
  }

  getJointWorldBindMatrices() {
    this.root.updateWorldMatrix(true, true);
    return this.jointObjs.map((obj) => {
      obj.updateWorldMatrix(true, false);
      return matrix4ToFlatRows(obj.matrixWorld);
    });
  }

  _recomputeBindInvFromHierarchy() {
    this.root.updateWorldMatrix(true, true);
    this.bindInv = this.jointObjs.map((obj) => {
      obj.updateWorldMatrix(true, false);
      return new THREE.Matrix4().copy(obj.matrixWorld).invert();
    });
  }

  _buildBoneLines(lineColor = 0xffaa00) {
    while (this.boneLines.children.length) {
      const c = this.boneLines.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
    const lineMat = new THREE.LineBasicMaterial({ color: lineColor });
    for (let i = 0; i < this.parents.length; i++) {
      const p = this.parents[i];
      if (p < 0) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const line = new THREE.Line(geo, lineMat);
      line.userData.parentIndex = p;
      line.userData.childIndex = i;
      this.boneLines.add(line);
    }
    this._updateBoneLines();
  }

  _updateBoneLines() {
    for (const line of this.boneLines.children) {
      const p = line.userData.parentIndex;
      const c = line.userData.childIndex;
      this.jointObjs[p].getWorldPosition(_v0);
      this.jointObjs[c].getWorldPosition(_v1);
      const pos = line.geometry.attributes.position;
      pos.setXYZ(0, _v0.x, _v0.y, _v0.z);
      pos.setXYZ(1, _v1.x, _v1.y, _v1.z);
      pos.needsUpdate = true;
    }
  }

  _scheduleLbs() {
    if (this._lbsPending) return;
    this._lbsPending = true;
    requestAnimationFrame(() => {
      this._lbsPending = false;
      this._applyLbs();
    });
  }

  _applyLbs() {
    if (!this.mesh || !this.restPositions) return;

    const nJoints = this.jointObjs.length;
    for (let j = 0; j < nJoints; j++) {
      this.jointObjs[j].updateWorldMatrix(true, false);
      _m0.copy(this.jointObjs[j].matrixWorld).multiply(this.bindInv[j]);
      this.skinMats[j].copy(_m0);
    }

    const posAttr = this.mesh.geometry.attributes.position;
    const nVerts = this.restPositions.length / 3;
    const stride = 4;

    for (let i = 0; i < nVerts; i++) {
      const bx = this.restPositions[i * 3];
      const by = this.restPositions[i * 3 + 1];
      const bz = this.restPositions[i * 3 + 2];
      let ox = 0;
      let oy = 0;
      let oz = 0;
      const base = i * stride;
      for (let k = 0; k < stride; k++) {
        const w = this.lbsWeights[base + k];
        if (w < 1e-7) continue;
        const j = this.lbsIndices[base + k];
        const sm = this.skinMats[j];
        const e = sm.elements;
        ox += w * (e[0] * bx + e[4] * by + e[8] * bz + e[12]);
        oy += w * (e[1] * bx + e[5] * by + e[9] * bz + e[13]);
        oz += w * (e[2] * bx + e[6] * by + e[10] * bz + e[14]);
      }
      posAttr.setXYZ(i, ox, oy, oz);
    }
    posAttr.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
  }

  setMeshOpacity(percent) {
    if (!this.meshMaterial) return;
    const alpha = Math.min(1, Math.max(0, percent / 100));
    this.meshMaterial.transparent = alpha < 0.999;
    this.meshMaterial.opacity = alpha;
    this.meshMaterial.depthWrite = alpha > 0.95;
    this.meshMaterial.needsUpdate = true;
  }

  setTransformMode(mode) {
    const allowed = { translate: "translate", rotate: "rotate", scale: "scale" };
    this.transformControls.setMode(allowed[mode] || "translate");
  }

  setTransformSpace(space) {
    this.transformControls.setSpace(space === "world" ? "world" : "local");
  }

  updateLbs(lbsIndices, lbsWeights) {
    if (!this.mesh || !this.restPositions) return;
    const idxFlat = lbsIndices.flat();
    const wFlat = lbsWeights.flat();
    this.lbsIndices = new Int32Array(idxFlat);
    this.lbsWeights = new Float32Array(wFlat);
    this._applyLbs();
    if (this.heatmapJointIndex >= 0) {
      this.showWeightHeatmap(this.heatmapJointIndex);
    }
  }

  resetWeightTools() {
    this.setWeightTool(null);
    this.selectedIndex = -1;
    this.heatmapJointIndex = -1;
    this.transformControls.detach();
    this.clearWeightHeatmap();
    this._refreshJointColors();
  }

  setBrushRadiusScale(scale) {
    this.brushRadiusScale = Math.max(0.01, Math.min(1, scale));
    this._refreshBrushPreviewFromLastHit();
  }

  setBrushStrength(strength) {
    this.brushStrength = Math.max(0.01, Math.min(1, strength));
    this._refreshBrushPreviewStyle();
  }

  setWeightTool(mode) {
    const allowed = { paint: "paint", erase: "erase" };
    this.weightTool = allowed[mode] || null;
    this._brushStroke = false;
    if (this.weightTool) {
      this.transformControls.detach();
      this._refreshBrushPreviewStyle();
    } else if (this.selectedIndex >= 0) {
      this._hideBrushPreview();
      this.transformControls.attach(this.jointObjs[this.selectedIndex]);
    } else {
      this._hideBrushPreview();
    }
  }

  _syncBrushRadiusFromMesh() {
    if (!this.mesh?.geometry) return;
    this.mesh.geometry.computeBoundingBox();
    const size = this.mesh.geometry.boundingBox.getSize(new THREE.Vector3()).length();
    this._meshBrushBase = Math.max(size * 0.08, 0.01);
  }

  _brushWorldRadius() {
    const base = this._meshBrushBase || 0.05;
    return base * this.brushRadiusScale;
  }

  _refreshBrushPreviewStyle() {
    if (!this.brushPreview?.material) return;
    const strength = Math.max(0.01, Math.min(1, this.brushStrength));
    const hue = this.weightTool === "erase" ? 0.0 : 0.34;
    const saturation = 0.2 + strength * 0.8;
    this.brushPreview.material.color.setHSL(hue, saturation, 0.52);
    this.brushPreview.material.opacity = 0.55 + strength * 0.4;
    this.brushPreview.material.needsUpdate = true;
  }

  _showBrushPreview(hit) {
    if (!hit || !this.weightTool || this.selectedIndex < 0 || !this.brushPreview) {
      this._hideBrushPreview();
      return;
    }
    _brushNormal.copy(hit.face?.normal || _unitZ).transformDirection(hit.object.matrixWorld).normalize();
    _brushPoint.copy(hit.point);
    const radius = this._brushWorldRadius();
    this.brushPreview.position.copy(_brushPoint).addScaledVector(_brushNormal, Math.max(radius * 0.006, 0.0005));
    this.brushPreview.quaternion.setFromUnitVectors(_unitZ, _brushNormal);
    this.brushPreview.scale.setScalar(radius);
    this._refreshBrushPreviewStyle();
    this.brushPreview.visible = true;
    this._lastBrushHit = {
      point: _brushPoint.clone(),
      normal: _brushNormal.clone(),
    };
  }

  _refreshBrushPreviewFromLastHit() {
    if (!this._lastBrushHit || !this.brushPreview?.visible) return;
    const radius = this._brushWorldRadius();
    this.brushPreview.position.copy(this._lastBrushHit.point).addScaledVector(this._lastBrushHit.normal, Math.max(radius * 0.006, 0.0005));
    this.brushPreview.quaternion.setFromUnitVectors(_unitZ, this._lastBrushHit.normal);
    this.brushPreview.scale.setScalar(radius);
    this._refreshBrushPreviewStyle();
  }

  _brushHitFromEvent(event) {
    if (!this.weightTool || this.selectedIndex < 0 || !this.mesh) {
      this._hideBrushPreview();
      return null;
    }
    this._updatePointerFromEvent(event);
    const hits = this.raycaster.intersectObject(this.mesh, false);
    if (!hits.length) {
      this._hideBrushPreview();
      return null;
    }
    this._showBrushPreview(hits[0]);
    return hits[0];
  }

  _getVertexWeight(vi, jointIndex) {
    const base = vi * 4;
    for (let k = 0; k < 4; k += 1) {
      if (this.lbsIndices[base + k] === jointIndex) {
        return this.lbsWeights[base + k];
      }
    }
    return 0;
  }

  _setVertexWeight(vi, jointIndex, weight) {
    const base = vi * 4;
    weight = Math.max(0, Math.min(1, weight));
    let slot = -1;
    let minW = Infinity;
    let minSlot = 0;
    for (let k = 0; k < 4; k += 1) {
      const idx = this.lbsIndices[base + k];
      const w = this.lbsWeights[base + k];
      if (idx === jointIndex) slot = k;
      if (w < minW) {
        minW = w;
        minSlot = k;
      }
    }
    if (slot < 0) slot = minSlot;

    this.lbsIndices[base + slot] = jointIndex;
    this.lbsWeights[base + slot] = weight;

    let otherSum = 0;
    for (let k = 0; k < 4; k += 1) {
      if (k === slot) continue;
      otherSum += this.lbsWeights[base + k];
    }
    const remain = Math.max(0, 1 - weight);
    if (otherSum < 1e-8) {
      for (let k = 0; k < 4; k += 1) {
        if (k === slot) continue;
        this.lbsWeights[base + k] = remain / 3;
      }
    } else {
      const scale = remain / otherSum;
      for (let k = 0; k < 4; k += 1) {
        if (k === slot) continue;
        this.lbsWeights[base + k] *= scale;
      }
    }
  }

  showWeightHeatmap(jointIndex) {
    if (!this.mesh || !this.lbsIndices || jointIndex < 0) return;
    this.heatmapJointIndex = jointIndex;
    const nVerts = this.restPositions.length / 3;
    let colors = this.mesh.geometry.getAttribute("color");
    if (!colors || colors.count !== nVerts) {
      colors = new THREE.BufferAttribute(new Float32Array(nVerts * 3), 3);
      this.mesh.geometry.setAttribute("color", colors);
    }
    for (let i = 0; i < nVerts; i += 1) {
      const w = this._getVertexWeight(i, jointIndex);
      colors.setXYZ(i, w, w, w);
    }
    colors.needsUpdate = true;
    this.meshMaterial.vertexColors = true;
    this.meshMaterial.color.set(0xffffff);
    this.meshMaterial.needsUpdate = true;
  }

  clearWeightHeatmap() {
    this.heatmapJointIndex = -1;
    if (!this.mesh?.geometry) return;
    const colors = this.mesh.geometry.getAttribute("color");
    if (colors) {
      this.mesh.geometry.deleteAttribute("color");
    }
    if (this.meshMaterial) {
      this.meshMaterial.vertexColors = false;
      this.meshMaterial.color.set(this._baseMeshColor);
      this.meshMaterial.needsUpdate = true;
    }
  }

  getLbsPayload() {
    if (!this.lbsIndices || !this.lbsWeights) {
      throw new Error("binding weights not loaded");
    }
    const nVerts = this.restPositions.length / 3;
    const lbs_indices = [];
    const lbs_weights = [];
    for (let i = 0; i < nVerts; i += 1) {
      const base = i * 4;
      lbs_indices.push([
        this.lbsIndices[base],
        this.lbsIndices[base + 1],
        this.lbsIndices[base + 2],
        this.lbsIndices[base + 3],
      ]);
      lbs_weights.push([
        this.lbsWeights[base],
        this.lbsWeights[base + 1],
        this.lbsWeights[base + 2],
        this.lbsWeights[base + 3],
      ]);
    }
    return { lbs_indices, lbs_weights };
  }

  _updatePointerFromEvent(event) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  _applyBrushAtWorldPoint(worldPoint) {
    if (!this.mesh || this.selectedIndex < 0 || !this.weightTool) return;
    const jointIndex = this.selectedIndex;
    const radius = this._brushWorldRadius();
    const strength = this.brushStrength;
    const posAttr = this.mesh.geometry.attributes.position;
    const nVerts = posAttr.count;
    const radiusSq = radius * radius;

    this.mesh.updateMatrixWorld(true);

    for (let i = 0; i < nVerts; i += 1) {
      _v0.fromBufferAttribute(posAttr, i);
      _v0.applyMatrix4(this.mesh.matrixWorld);
      const distSq = _v0.distanceToSquared(worldPoint);
      if (distSq > radiusSq) continue;
      const dist = Math.sqrt(distSq);
      const falloff = 1 - dist / radius;
      const oldW = this._getVertexWeight(i, jointIndex);
      let newW = oldW;
      if (this.weightTool === "paint") {
        newW = oldW + strength * falloff * (1 - oldW);
      } else {
        newW = oldW * (1 - strength * falloff);
      }
      this._setVertexWeight(i, jointIndex, newW);
    }

    this.showWeightHeatmap(jointIndex);
    this._applyLbs();
  }

  _tryBrushStroke(event) {
    if (!this.weightTool || this.selectedIndex < 0 || !this.mesh) return false;
    const hit = this._brushHitFromEvent(event);
    if (!hit) return false;
    this._applyBrushAtWorldPoint(hit.point);
    return true;
  }

  selectJoint(index) {
    if (index < 0 || index >= this.jointObjs.length) {
      this.selectedIndex = -1;
      this.heatmapJointIndex = -1;
      this.transformControls.detach();
      this._hideBrushPreview();
      this.clearWeightHeatmap();
      this._refreshJointColors();
      this.onSelect?.(null);
      return;
    }
    this.selectedIndex = index;
    if (this.weightTool) {
      this.transformControls.detach();
    } else {
      this._hideBrushPreview();
      this.transformControls.attach(this.jointObjs[index]);
    }
    this.showWeightHeatmap(index);
    this._refreshJointColors();
    this.onSelect?.({ index, name: this.names[index] || `joint_${index}` });
  }

  _refreshJointColors() {
    for (let i = 0; i < this.jointObjs.length; i++) {
      for (const child of this.jointObjs[i].children) {
        if (!child.userData.isVisual) continue;
        child.material.color.set(i === this.selectedIndex ? 0x4fc3f7 : 0xffcc00);
      }
    }
  }

  resetPose() {
    for (let i = 0; i < this.jointObjs.length; i++) {
      const obj = this.jointObjs[i];
      obj.position.copy(this.restLocals[i]);
      obj.quaternion.identity();
      obj.scale.set(1, 1, 1);
      obj.updateMatrix();
    }
    this.root.updateWorldMatrix(true, true);
    this._applyLbs();
    this._updateBoneLines();
    if (this.selectedIndex >= 0 && !this.weightTool) {
      this.transformControls.attach(this.jointObjs[this.selectedIndex]);
    }
  }

  _onPointerDown(event) {
    if (!this.interactionEnabled) return;
    if (this.transformControls.dragging) return;

    if (this.weightTool && this.selectedIndex >= 0 && this.mesh) {
      this._updatePointerFromEvent(event);
      const boneHits = this.raycaster.intersectObjects(this.pickables, false);
      if (boneHits.length) {
        event.preventDefault();
        event.stopPropagation();
        this.selectJoint(boneHits[0].object.userData.jointIndex);
        return;
      }
      if (this._tryBrushStroke(event)) {
        event.preventDefault();
        event.stopPropagation();
        this._brushStroke = true;
        this.orbitControls.enabled = false;
        return;
      }
    }

    if (!this.pickables.length) return;
    this._updatePointerFromEvent(event);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    if (hits.length) {
      event.preventDefault();
      event.stopPropagation();
      this.selectJoint(hits[0].object.userData.jointIndex);
    }
  }

  _onPointerMove(event) {
    if (!this.interactionEnabled) return;
    if (this.weightTool && this.selectedIndex >= 0 && this.mesh) {
      const hit = this._brushHitFromEvent(event);
      if (this._brushStroke) {
        if (hit) {
          this._applyBrushAtWorldPoint(hit.point);
          event.preventDefault();
        }
        return;
      }
    }
  }

  _onPointerUp(event) {
    if (!this.interactionEnabled) return;
    if (this._brushStroke) {
      this._brushStroke = false;
      this.orbitControls.enabled = true;
    }
    if (event?.type === "pointerleave") {
      this._hideBrushPreview();
    }
  }
}

export class SomaOverlay {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.root.name = "somaOverlay";
    scene.add(this.root);
    this.boneLines = new THREE.Group();
    this.root.add(this.boneLines);
    this.jointObjs = [];
    this.sphereGeo = null;
    this.lineMat = new THREE.LineBasicMaterial({
      color: 0xff3333,
      depthTest: false,
      transparent: true,
      opacity: 0.95,
    });
    this.jointMat = new THREE.MeshBasicMaterial({
      color: 0xff3333,
      depthTest: false,
      transparent: true,
      opacity: 0.9,
    });
  }

  clear() {
    this.jointObjs = [];
    while (this.root.children.length > 1) {
      const c = this.root.children.pop();
      if (c !== this.boneLines) c.removeFromParent();
    }
    while (this.boneLines.children.length) {
      const c = this.boneLines.children.pop();
      c.geometry?.dispose();
    }
    this.sphereGeo?.dispose();
    this.sphereGeo = null;
  }

  dispose() {
    this.clear();
    this.lineMat.dispose();
    this.jointMat.dispose();
    this.scene.remove(this.root);
  }

  load(data, { meshSize = 1 } = {}) {
    this.clear();
    const parents = data.parents;
    const joints = data.joints;
    const jointRadius = Math.max(meshSize * 0.01, 0.0035);
    this.sphereGeo = new THREE.SphereGeometry(jointRadius, 8, 8);

    const worldPos = joints.map((p) => new THREE.Vector3(...p));
    const restLocals = worldPos.map((pos, i) => {
      const p = parents[i];
      if (p < 0) return pos.clone();
      return pos.clone().sub(worldPos[p]);
    });

    const nJoints = joints.length;
    for (let i = 0; i < nJoints; i++) {
      const obj = new THREE.Object3D();
      this.jointObjs.push(obj);
    }

    for (let i = 0; i < nJoints; i++) {
      const p = parents[i];
      if (p >= 0) this.jointObjs[p].add(this.jointObjs[i]);
      else this.root.add(this.jointObjs[i]);
      this.jointObjs[i].position.copy(restLocals[i]);
      const visual = new THREE.Mesh(this.sphereGeo, this.jointMat);
      visual.renderOrder = 999;
      this.jointObjs[i].add(visual);
    }

    for (let i = 0; i < nJoints; i++) {
      const p = parents[i];
      if (p < 0) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const line = new THREE.Line(geo, this.lineMat);
      line.renderOrder = 998;
      line.userData.parentIndex = p;
      line.userData.childIndex = i;
      this.boneLines.add(line);
    }

    this.root.updateWorldMatrix(true, true);
    this._updateBoneLines();
  }

  _updateBoneLines() {
    for (const line of this.boneLines.children) {
      const p = line.userData.parentIndex;
      const c = line.userData.childIndex;
      this.jointObjs[p].getWorldPosition(_v0);
      this.jointObjs[c].getWorldPosition(_v1);
      const pos = line.geometry.attributes.position;
      pos.setXYZ(0, _v0.x, _v0.y, _v0.z);
      pos.setXYZ(1, _v1.x, _v1.y, _v1.z);
      pos.needsUpdate = true;
    }
  }
}

export class SomaEditController {
  constructor({
    scene,
    camera,
    canvas,
    orbitControls,
    onSelect,
  }) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
    this.orbitControls = orbitControls;
    this.onSelect = onSelect;

    this.root = new THREE.Group();
    this.root.name = "somaEditRoot";
    scene.add(this.root);

    this.boneLines = new THREE.Group();
    this.root.add(this.boneLines);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pickables = [];

    this.jointObjs = [];
    this.restLocals = [];
    this.initialWorld = [];
    this.initialBindMatrices = null;
    this.parents = [];
    this.names = [];
    this.selectedIndex = -1;

    this.transformControls = new TransformControls(camera, canvas);
    this.transformControls.setMode("translate");
    this.transformControls.setSize(0.85);
    this.scene.add(this.transformControls.getHelper());

    this.transformControls.addEventListener("dragging-changed", (ev) => {
      this.orbitControls.enabled = !ev.value;
    });
    this.transformControls.addEventListener("change", () => {
      this._updateBoneLines();
    });

    this._onPointerDown = this._onPointerDown.bind(this);
    canvas.addEventListener("pointerdown", this._onPointerDown);
  }

  setCamera(camera) {
    this.camera = camera;
    this.transformControls.camera = camera;
  }

  dispose() {
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.transformControls.detach();
    this.scene.remove(this.transformControls.getHelper());
    this.transformControls.dispose();
    this.clear();
    this.scene.remove(this.root);
  }

  clear() {
    this.transformControls.detach();
    this.pickables = [];
    this.jointObjs = [];
    this.restLocals = [];
    this.initialWorld = [];
    this.initialBindMatrices = null;
    this.selectedIndex = -1;
    while (this.root.children.length > 1) {
      const c = this.root.children.pop();
      if (c !== this.boneLines) c.removeFromParent();
    }
    while (this.boneLines.children.length) {
      const c = this.boneLines.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
  }

  load(data, { meshSize = 1 } = {}) {
    this.clear();
    const soma = data.soma || data;
    const parents = soma.parents;
    const joints = soma.joints;
    this.parents = parents;
    this.names = soma.names || [];

    const jointRadius = Math.max(meshSize * 0.012, 0.004);
    const pickRadius = jointRadius * 2.2;
    const sphereGeo = new THREE.SphereGeometry(jointRadius, 10, 10);
    const pickGeo = new THREE.SphereGeometry(pickRadius, 8, 8);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xff3333,
      depthTest: false,
      transparent: true,
      opacity: 0.95,
    });

    const worldPos = joints.map((p) => new THREE.Vector3(...p));
    this.restLocals = worldPos.map((pos, i) => {
      const p = parents[i];
      if (p < 0) return pos.clone();
      return pos.clone().sub(worldPos[p]);
    });
    this.initialWorld = worldPos.map((p) => p.clone());

    const nJoints = joints.length;
    for (let i = 0; i < nJoints; i++) {
      const obj = new THREE.Object3D();
      obj.userData.jointIndex = i;
      this.jointObjs.push(obj);
    }

    for (let i = 0; i < nJoints; i++) {
      const p = parents[i];
      if (p >= 0) this.jointObjs[p].add(this.jointObjs[i]);
      else this.root.add(this.jointObjs[i]);
      this.jointObjs[i].position.copy(this.restLocals[i]);
    }

    for (let i = 0; i < nJoints; i++) {
      const normalMat = new THREE.MeshBasicMaterial({
        color: 0xff3333,
        depthTest: false,
        transparent: true,
        opacity: 0.9,
      });
      const visual = new THREE.Mesh(sphereGeo, normalMat);
      visual.renderOrder = 999;
      visual.userData.jointIndex = i;
      visual.userData.isVisual = true;
      this.jointObjs[i].add(visual);

      const pickMat = new THREE.MeshBasicMaterial({
        color: 0xff3333,
        transparent: true,
        opacity: 0.001,
        depthWrite: false,
      });
      const picker = new THREE.Mesh(pickGeo, pickMat);
      picker.userData.jointIndex = i;
      picker.userData.isPicker = true;
      this.jointObjs[i].add(picker);
      this.pickables.push(picker);
    }

    for (let i = 0; i < nJoints; i++) {
      const p = parents[i];
      if (p < 0) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const line = new THREE.Line(geo, lineMat);
      line.renderOrder = 998;
      line.userData.parentIndex = p;
      line.userData.childIndex = i;
      this.boneLines.add(line);
    }

    this.root.updateWorldMatrix(true, true);
    if (soma.bind_matrices?.length) {
      applyAbsoluteBindPose(this.jointObjs, parents, soma.bind_matrices);
      this.initialBindMatrices = soma.bind_matrices.map((row) => row.slice());
    } else {
      this.initialBindMatrices = this.getJointWorldBindMatrices();
    }
    this.root.updateWorldMatrix(true, true);
    this._updateBoneLines();
  }

  _updateBoneLines() {
    for (const line of this.boneLines.children) {
      const p = line.userData.parentIndex;
      const c = line.userData.childIndex;
      this.jointObjs[p].getWorldPosition(_v0);
      this.jointObjs[c].getWorldPosition(_v1);
      const pos = line.geometry.attributes.position;
      pos.setXYZ(0, _v0.x, _v0.y, _v0.z);
      pos.setXYZ(1, _v1.x, _v1.y, _v1.z);
      pos.needsUpdate = true;
    }
  }

  setTransformMode(mode) {
    const allowed = { translate: "translate", rotate: "rotate", scale: "scale" };
    this.transformControls.setMode(allowed[mode] || "translate");
  }

  setTransformSpace(space) {
    this.transformControls.setSpace(space === "world" ? "world" : "local");
  }

  selectJoint(index) {
    if (index < 0 || index >= this.jointObjs.length) {
      this.selectedIndex = -1;
      this.transformControls.detach();
      this._refreshJointColors();
      this.onSelect?.(null);
      return;
    }
    this.selectedIndex = index;
    this.transformControls.attach(this.jointObjs[index]);
    this._refreshJointColors();
    this.onSelect?.({ index, name: this.names[index] || `joint_${index}` });
  }

  _refreshJointColors() {
    for (let i = 0; i < this.jointObjs.length; i++) {
      for (const child of this.jointObjs[i].children) {
        if (!child.userData.isVisual) continue;
        child.material.color.set(i === this.selectedIndex ? 0x4fc3f7 : 0xff3333);
      }
    }
  }

  resetPose() {
    if (this.initialBindMatrices?.length) {
      applyAbsoluteBindPose(this.jointObjs, this.parents, this.initialBindMatrices);
    } else {
      for (let i = 0; i < this.jointObjs.length; i++) {
        const obj = this.jointObjs[i];
        obj.position.copy(this.restLocals[i]);
        obj.quaternion.identity();
        obj.scale.set(1, 1, 1);
        obj.updateMatrix();
      }
    }
    this.root.updateWorldMatrix(true, true);
    this._updateBoneLines();
    if (this.selectedIndex >= 0) {
      this.transformControls.attach(this.jointObjs[this.selectedIndex]);
    }
  }

  getJointWorldBindMatrices() {
    this.root.updateWorldMatrix(true, true);
    return this.jointObjs.map((obj) => {
      obj.updateWorldMatrix(true, false);
      return matrix4ToFlatRows(obj.matrixWorld);
    });
  }

  getJointWorldPositions() {
    const out = [];
    this.root.updateWorldMatrix(true, true);
    for (let i = 0; i < this.jointObjs.length; i += 1) {
      this.jointObjs[i].getWorldPosition(_v0);
      out.push([_v0.x, _v0.y, _v0.z]);
    }
    return out;
  }

  getJointWorldPositions() {
    const out = [];
    for (let i = 0; i < this.jointObjs.length; i++) {
      this.jointObjs[i].getWorldPosition(_v0);
      out.push([_v0.x, _v0.y, _v0.z]);
    }
    return out;
  }

  _onPointerDown(event) {
    if (!this.pickables.length || this.transformControls.dragging) return;
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    if (hits.length) {
      event.preventDefault();
      event.stopPropagation();
      this.selectJoint(hits[0].object.userData.jointIndex);
    }
  }
}

function rowsToMatrix4(rows) {
  const m = new THREE.Matrix4();
  m.set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    rows[3][0], rows[3][1], rows[3][2], rows[3][3],
  );
  return m;
}

export function bakeGeometryTransform(geometry, rows) {
  const m = rowsToMatrix4(rows);
  const pos = geometry.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    v.fromBufferAttribute(pos, i);
    v.applyMatrix4(m);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  pos.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
}

function matrix4ToRows(m) {
  const e = m.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];
}

export class MeshTransformController {
  constructor({ scene, camera, canvas, orbitControls }) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
    this.orbitControls = orbitControls;
    this.target = null;
    this.onChange = null;
    this._pos = new THREE.Vector3();
    this._quat = new THREE.Quaternion();
    this._scale = new THREE.Vector3(1, 1, 1);
    this._euler = new THREE.Euler(0, 0, 0, "XYZ");

    this.transformControls = new TransformControls(camera, canvas);
    this.transformControls.setMode("translate");
    this.transformControls.setSize(0.85);
    this.scene.add(this.transformControls.getHelper());

    this.transformControls.addEventListener("dragging-changed", (ev) => {
      this.orbitControls.enabled = !ev.value;
    });
    this.transformControls.addEventListener("change", () => {
      this._enforceUniformScale();
      if (this.onChange) this.onChange();
    });
  }

  setCamera(camera) {
    this.camera = camera;
    this.transformControls.camera = camera;
  }

  _syncTarget() {
    if (!this.target) return;
    this.target.updateMatrix();
    this.target.updateMatrixWorld(true);
  }

  _enforceUniformScale() {
    if (!this.target || this.transformControls.mode !== "scale") return;
    const s = this.target.scale;
    const u = Math.max(s.x, s.y, s.z);
    if (Math.abs(s.x - u) > 1e-6 || Math.abs(s.y - u) > 1e-6 || Math.abs(s.z - u) > 1e-6) {
      s.set(u, u, u);
      this._syncTarget();
    }
  }

  dispose() {
    this.detach();
    this.scene.remove(this.transformControls.getHelper());
    this.transformControls.dispose();
  }

  attach(mesh) {
    this.target = mesh;
    mesh.position.set(0, 0, 0);
    mesh.rotation.set(0, 0, 0);
    mesh.scale.set(1, 1, 1);
    mesh.updateMatrix();
    this.transformControls.attach(mesh);
    if (this.onChange) this.onChange();
  }

  detach() {
    this.transformControls.detach();
    this.target = null;
  }

  setMode(mode) {
    this.transformControls.setMode(mode);
    if (mode === "scale" && this.target) {
      const u = this.target.scale.x;
      this.target.scale.set(u, u, u);
    }
  }

  setMatrixRows(rows) {
    if (!this.target) return;
    const m = rowsToMatrix4(rows);
    m.decompose(this._pos, this._quat, this._scale);
    const u = Math.max(this._scale.x, this._scale.y, this._scale.z);
    this.target.position.copy(this._pos);
    this.target.quaternion.copy(this._quat);
    this.target.scale.set(u, u, u);
    this._syncTarget();
    if (this.onChange) this.onChange();
  }

  getLocalMatrix4() {
    if (!this.target) return new THREE.Matrix4();
    this.target.updateMatrix();
    const m = new THREE.Matrix4();
    const u = Math.max(this.target.scale.x, 1e-6);
    m.compose(
      this.target.position,
      this.target.quaternion,
      new THREE.Vector3(u, u, u),
    );
    return m;
  }

  getMatrixRows() {
    if (!this.target) {
      return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ];
    }
    this.target.updateMatrix();
    return matrix4ToRows(this.target.matrix);
  }

  getAlignPayload() {
    if (!this.target) {
      return {
        position: [0, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: 1,
      };
    }
    this.target.updateMatrix();
    const u = Math.max(this.target.scale.x, 1e-6);
    return {
      position: [this.target.position.x, this.target.position.y, this.target.position.z],
      quaternion: [
        this.target.quaternion.x,
        this.target.quaternion.y,
        this.target.quaternion.z,
        this.target.quaternion.w,
      ],
      scale: u,
    };
  }

  getTRS() {
    if (!this.target) {
      return { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, scale: 1 };
    }
    this._euler.setFromQuaternion(this.target.quaternion, "XYZ");
    return {
      tx: this.target.position.x,
      ty: this.target.position.y,
      tz: this.target.position.z,
      rx: THREE.MathUtils.radToDeg(this._euler.x),
      ry: THREE.MathUtils.radToDeg(this._euler.y),
      rz: THREE.MathUtils.radToDeg(this._euler.z),
      scale: this.target.scale.x,
    };
  }

  setTRS({ tx, ty, tz, rx, ry, rz, scale }) {
    if (!this.target) return;
    this.target.position.set(tx, ty, tz);
    this._euler.set(
      THREE.MathUtils.degToRad(rx),
      THREE.MathUtils.degToRad(ry),
      THREE.MathUtils.degToRad(rz),
    );
    this.target.quaternion.setFromEuler(this._euler);
    const u = Math.max(scale, 1e-6);
    this.target.scale.set(u, u, u);
    this._syncTarget();
    if (this.onChange) this.onChange();
  }

  resetTargetIdentity() {
    if (!this.target) return;
    this.target.position.set(0, 0, 0);
    this.target.rotation.set(0, 0, 0);
    this.target.scale.set(1, 1, 1);
    this._syncTarget();
    if (this.onChange) this.onChange();
  }

  resetTransform() {
    this.setMatrixRows([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);
  }
}

export class ViewController {
  constructor({ perspectiveCamera, controls, canvas, getFocusBounds }) {
    this.perspectiveCamera = perspectiveCamera;
    this.controls = controls;
    this.canvas = canvas;
    this.getFocusBounds = getFocusBounds;
    this.orthoCamera = null;
    this.activeCamera = perspectiveCamera;
    this.currentView = "perspective";
    this._perspectiveDir = new THREE.Vector3(0.6, 0.4, 0.8).normalize();
  }

  getCamera() {
    return this.activeCamera;
  }

  setCameraHook(setCameraFn) {
    this._setCameraHook = setCameraFn;
  }

  _notifyCamera() {
    if (this._setCameraHook) this._setCameraHook(this.activeCamera);
  }

  _ensureOrtho() {
    if (this.orthoCamera) return this.orthoCamera;
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    this.orthoCamera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0.01, 500);
    return this.orthoCamera;
  }

  _fitOrtho(cam, viewDir, up) {
    const { center, size } = this.getFocusBounds();
    const dist = Math.max(size * 2, 2);
    cam.position.copy(center).add(viewDir.clone().multiplyScalar(dist));
    cam.up.copy(up);
    cam.lookAt(center);
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    const half = Math.max(size * 0.55, 0.5);
    cam.left = -half * aspect;
    cam.right = half * aspect;
    cam.top = half;
    cam.bottom = -half;
    cam.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  setView(name) {
    this.currentView = name;
    if (name === "perspective") {
      this.activeCamera = this.perspectiveCamera;
      const { center, size } = this.getFocusBounds();
      const dist = Math.max(size * 1.2, 1.5);
      this.perspectiveCamera.position.copy(center).add(
        this._perspectiveDir.clone().multiplyScalar(dist),
      );
      this.perspectiveCamera.up.set(0, 1, 0);
      this.perspectiveCamera.lookAt(center);
      this.perspectiveCamera.updateProjectionMatrix();
      this.controls.target.copy(center);
      this.controls.update();
    } else {
      const ortho = this._ensureOrtho();
      this.activeCamera = ortho;
      const views = {
        front: { dir: new THREE.Vector3(0, 0, 1), up: new THREE.Vector3(0, 1, 0) },
        side: { dir: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 1, 0) },
        top: { dir: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 0, -1) },
      };
      const cfg = views[name];
      if (cfg) this._fitOrtho(ortho, cfg.dir, cfg.up);
    }
    this.controls.object = this.activeCamera;
    this._notifyCamera();
  }

  onResize() {
    if (this.currentView !== "perspective" && this.orthoCamera) {
      this.setView(this.currentView);
    }
  }
}
