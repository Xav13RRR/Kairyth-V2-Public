import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(root, "motionbuilder_app", "static", "studio.js"), "utf8");
const context = vm.createContext({ Math, Number, Array, state: { kimodoTrajectory: { points: [] } } });

function loadFunction(name) {
  const line = source.split(/\r?\n/).find((item) => item.startsWith(`function ${name}(`));
  assert.ok(line, `missing ${name}`);
  vm.runInContext(`${line};this.${name}=${name}`, context);
}

for (const name of [
  "normalizeKimodoYaw",
  "kimodoPointDistance",
  "setKimodoPointDistance",
  "kimodoQuatMultiply",
  "kimodoRotvecToQuat",
  "kimodoQuatToRotvec",
  "rotateKimodoFullbodyRoot",
]) loadFunction(name);

context.state.kimodoTrajectory.points = [{ x: 0, z: 0 }, { x: 1, z: 1 }];
assert.equal(context.setKimodoPointDistance(1, 2), true);
assert.ok(Math.abs(context.state.kimodoTrajectory.points[1].x - Math.SQRT2) < 1e-6);
assert.ok(Math.abs(context.state.kimodoTrajectory.points[1].z - Math.SQRT2) < 1e-6);

context.state.kimodoTrajectory.points = [{ x: 0, z: 0 }];
assert.equal(context.setKimodoPointDistance(0, 2), true);
assert.deepEqual(context.state.kimodoTrajectory.points[0], { x: 2, z: 0 });

const pose = { type: "fullbody", local_joints_rot: [[[0, 0, 0], [0, 0, 0]]] };
context.rotateKimodoFullbodyRoot(pose, 90);
assert.ok(Math.abs(pose.local_joints_rot[0][0][0]) < 1e-6);
assert.ok(Math.abs(pose.local_joints_rot[0][0][1] - Math.PI / 2) < 1e-6);
assert.ok(Math.abs(pose.local_joints_rot[0][0][2]) < 1e-6);

assert.match(source, /global_root_heading:points\.map/);
assert.match(source, /data-kimodo-traj-distance="\$\{i\}" type="number" min="0" step="any"/);
assert.match(source, /data-kimodo-traj-yaw="\$\{i\}" type="number" step="any"/);
console.log("kimodo_math_ok");
