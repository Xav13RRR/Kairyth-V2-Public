import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

function buildMeshFromData(data, color) {
  const verts = new Float32Array(data.vertices.flat());
  const indices = new Uint32Array(data.faces.flat());
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts.slice(), 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeVertexNormals();
  geo.computeBoundingBox();
  const mat = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.1,
    roughness: 0.65,
    side: THREE.DoubleSide,
  });
  return { mesh: new THREE.Mesh(geo, mat), geo, verts };
}

export class StudioView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0f1115);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
    this.camera.position.set(2.5, 1.8, 3.5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(3, 5, 2);
    this.scene.add(dir);
    this.scene.add(new THREE.GridHelper(4, 20, 0x333333, 0x222222));

    this.defaultRoot = new THREE.Group();
    this.customRoot = new THREE.Group();
    this.scene.add(this.defaultRoot);
    this.scene.add(this.customRoot);

    this.default = { mesh: null, geo: null };
    this.custom = { mesh: null, geo: null };

    this.offsetTransform = new TransformControls(this.camera, canvas);
    this.offsetTransform.setMode("translate");
    this.offsetTransform.setSize(0.65);
    this.scene.add(this.offsetTransform.getHelper());

    this.onOffsetChange = null;

    this.offsetTransform.addEventListener("dragging-changed", (ev) => {
      this.controls.enabled = !ev.value;
    });
    this.offsetTransform.addEventListener("change", () => {
      this.onOffsetChange?.(this.getCustomOffset());
    });

    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _clearCharacter(slot, root) {
    const ch = this[slot];
    if (ch.mesh) {
      ch.geo?.dispose();
      ch.mesh.material?.dispose();
    }
    while (root.children.length) root.remove(root.children[0]);
    ch.mesh = null;
    ch.geo = null;
  }

  _mountCharacter(slot, root, data, { meshColor }) {
    this._clearCharacter(slot, root);
    const { mesh, geo } = buildMeshFromData(data, meshColor);
    root.add(mesh);
    this[slot] = { mesh, geo };
  }

  loadScene(sceneData) {
    this._mountCharacter("default", this.defaultRoot, sceneData.default, { meshColor: 0x9aa0a6 });
    this._mountCharacter("custom", this.customRoot, sceneData.custom, { meshColor: 0x8ab4f8 });
    this.setCustomOffset(sceneData.custom_offset || [1.5, 0, 0]);
    this._frameCamera();
    this.offsetTransform.attach(this.customRoot);
  }

  _frameCamera() {
    const box = new THREE.Box3();
    this.scene.updateMatrixWorld(true);
    if (this.default.mesh) box.expandByObject(this.defaultRoot);
    if (this.custom.mesh) box.expandByObject(this.customRoot);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size * 0.55, size * 0.35, size * 0.75));
  }

  setCustomOffset(offset) {
    this.customRoot.position.set(offset[0] || 0, offset[1] || 0, offset[2] || 0);
  }

  getCustomOffset() {
    const p = this.customRoot.position;
    return [p.x, p.y, p.z];
  }

  setDefaultMeshVertices(flatVerts) {
    this._setMeshVertices(this.default, flatVerts);
  }

  setCustomMeshVertices(flatVerts) {
    this._setMeshVertices(this.custom, flatVerts);
  }

  _setMeshVertices(ch, flatVerts) {
    if (!ch.geo) return;
    const attr = ch.geo.getAttribute("position");
    for (let i = 0; i < attr.count; i++) {
      attr.setXYZ(i, flatVerts[i * 3], flatVerts[i * 3 + 1], flatVerts[i * 3 + 2]);
    }
    attr.needsUpdate = true;
    ch.geo.computeVertexNormals();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
