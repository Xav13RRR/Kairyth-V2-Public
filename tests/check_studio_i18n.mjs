import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const htmlPath = path.join(root, "motionbuilder_app", "static", "index.html");
const rigHtmlPath = path.join(root, "motionbuilder_app", "meshpreps", "static", "index.html");
const scriptPath = path.join(root, "motionbuilder_app", "static", "studio_i18n.js");
const studioPath = path.join(root, "motionbuilder_app", "static", "studio.js");
const html = fs.readFileSync(htmlPath, "utf8");
const rigHtml = fs.readFileSync(rigHtmlPath, "utf8");
const source = fs.readFileSync(scriptPath, "utf8");
const studioSource = fs.readFileSync(studioPath, "utf8");

const storage = new Map();
const sandbox = {
  console,
  localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value))
  },
  document: { readyState: "loading", addEventListener() {} },
  window: {},
  MutationObserver: class {},
  CustomEvent: class {},
  NodeFilter: { SHOW_ELEMENT: 1, SHOW_TEXT: 4 }
};
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: scriptPath });

const i18n = sandbox.window.MotionBuilderI18n;
assert.ok(i18n, "MotionBuilderI18n should be exposed before studio.js runs");
assert.equal(i18n.getLanguage(), "zh", "Chinese must be the first-use default");
assert.equal(i18n.translateToEnglish("模型仓库"), "Model Library");
assert.equal(i18n.translateToEnglish("角色绑定与蒙皮"), "Rigging & Skinning");
assert.equal(i18n.translateToEnglish("视频动作推理"), "Video-to-Motion");
assert.equal(i18n.translateToEnglish("动画已就绪：207帧"), "Animation ready: 207 frames");
assert.equal(i18n.translateToEnglish("加法动画叠加"), "Additive Animation");
assert.equal(i18n.translateToEnglish("基础动画速率"), "Base Animation Rate");
assert.equal(i18n.translateToEnglish("已存为新动画：Walk + Breath"), "Saved as new animation: Walk + Breath");
assert.equal(i18n.translateToEnglish("动作预览 · User动作"), "Motion Preview · User动作");
assert.equal(i18n.translateToEnglish("已删除 3 段失效的过渡帧"), "Removed 3 invalid transition ranges");
assert.equal(i18n.translateToEnglish("已删除 [43, 71)，剩余 303 帧 · 已删除 1 段失效的过渡帧"), "Deleted [43, 71); 303 frames remain · removed 1 invalid transition ranges");
assert.equal(i18n.translateToEnglish("User Asset"), "User Asset", "unknown user content must remain unchanged");
assert.equal(i18n.translateToEnglish("HybrIK 独立研究进程推理中"), "HybrIK independent research process running");
assert.equal(i18n.translateToEnglish("图片 Pose 任务已取消"), "Image Pose task cancelled");
const poseStageFunction = studioSource.match(/^function normalizePoseWorkerStage\(message\) \{[\s\S]*?^\}/mu)?.[0];
assert.ok(poseStageFunction, "pose worker normalization must be present");
vm.runInContext(poseStageFunction, sandbox);
for (const [english, chinese] of [
  ["Loading configured HybrIK research backend", "正在加载已配置的 HybrIK 研究后端"],
  ["HybrIK single-image inference", "HybrIK 单图推理"],
  ["Converting original output to SOMA30", "正在将原始结果转换为 SOMA30"]
]) {
  assert.equal(sandbox.normalizePoseWorkerStage(english).text, chinese);
  assert.equal(i18n.translateToEnglish(chinese), english);
}
assert.equal(sandbox.normalizePoseWorkerStage("External diagnostic 原文 42").known, false);
assert.equal(sandbox.normalizePoseWorkerStage("External diagnostic 原文 42").text, "External diagnostic 原文 42");
assert.match(source, /"\.pose-job-diagnostic"/u);

// Evaluate the actual capability predicates, without pretending this is a browser test.
const capabilityBlock = studioSource.slice(studioSource.indexOf("let releaseCapabilities="), studioSource.indexOf("function renderHealthBadge()"));
assert.ok(capabilityBlock.includes('"kairyth-capabilities/v1"'));
vm.runInContext("var healthSnapshot=null;var healthChecked=false;" + capabilityBlock, sandbox);
assert.equal(sandbox.researchFeatureAllowed("hybrik"), false, "pending capabilities fail closed");
const features = (enabled, state) => ({hybrik_image_inference:{enabled,state},gvhmr_video_inference:{enabled,state}});
for (const [profile, valid, enabled, state] of [
  ["public-workbench",true,false,"disabled_experimental"],
  ["unconfigured",false,false,"disabled_configuration"],
  ["local-research",true,true,"enabled"]
]) {
  const capability = {schema:"kairyth-capabilities/v1",release_profile:profile,profile_valid:valid,features:features(enabled,state)};
  assert.equal(sandbox.validReleaseCapabilities(capability),true);
  vm.runInContext(`releaseCapabilitiesChecked=true;releaseCapabilities=${JSON.stringify(capability)};healthSnapshot={service_ready:true};healthChecked=true;`,sandbox);
  assert.equal(sandbox.researchFeatureAllowed("hybrik"),enabled);
  assert.equal(sandbox.researchFeatureAllowed("gvhmr"),enabled);
  assert.equal(sandbox.researchFeaturePresentation("hybrik").state,state);
  if (enabled) assert.doesNotMatch(sandbox.researchFeaturePresentation("hybrik").message,/公开版暂不提供/u);
}
vm.runInContext("healthSnapshot=null",sandbox);
assert.equal(sandbox.researchFeatureAllowed("hybrik"),false,"health failure must not reopen inference");
assert.equal(sandbox.validReleaseCapabilities({schema:"invalid"}),false);
assert.equal(i18n.translateToEnglish("实验性，后续会更新与HybrIK以及GVHMR的兼容包"),"Experimental. Compatibility packages for HybrIK and GVHMR are planned for a future update.");

assert.match(html, /id="studioLanguageZh"[^>]*>中<\/button>/u);
assert.match(html, /id="studioLanguageEn"[^>]*>ENG<\/button>/u);
assert.match(html, /studio_i18n\.js\?v=20-release-capabilities/u);
assert.match(html, /<title>KAIRYTH: Generative Motion Systems<\/title>/u);
assert.match(html, /rel="icon"[^>]*grs_cube_vectorized_white\.svg/u);
assert.match(html, /src="\/app-static\/kairyth_wordmark_clean\.svg\?v=1"[^>]*alt="KAIRYTH"/u);
assert.match(html, />Motion, timed by intent</u);
assert.match(html, />BY G\.R\.S: GAME-READY SYSTEMS</u);
assert.match(html, />STATUS:CHECK</u);
assert.ok(html.indexOf("studio_i18n.js") < html.indexOf("studio.js?v="), "i18n must load before studio.js");
assert.doesNotMatch(html, /class="[^"]*\btab\b[^"]*studio-language-tab/u, "language controls must not enter the page router");

const bodyWithoutScripts = html.replace(/<script\b[\s\S]*?<\/script>/giu, "");
const staticTexts = [...bodyWithoutScripts.matchAll(/>([^<>]+)</gu)]
  .map((match) => match[1].trim())
  .filter((text) => text && /\p{Script=Han}/u.test(text) && text !== "中");
const untranslated = [...new Set(staticTexts.filter((text) => i18n.translateToEnglish(text) === text))];
assert.deepEqual(untranslated, [], `Untranslated static text: ${untranslated.join(" | ")}`);
const chineseAttributes = [...html.matchAll(/(?:title|alt|placeholder|aria-label)="([^"]*\p{Script=Han}[^"]*)"/gu)]
  .map((match) => match[1])
  .filter((text) => !text.includes("切换至中文"));
const untranslatedAttributes = [...new Set(chineseAttributes.filter((text) => i18n.translateToEnglish(text) === text))];
assert.deepEqual(untranslatedAttributes, [], `Untranslated static attribute: ${untranslatedAttributes.join(" | ")}`);

const rigBodyWithoutScripts = rigHtml.replace(/<script\b[\s\S]*?<\/script>/giu, "");
const rigStaticTexts = [...rigBodyWithoutScripts.matchAll(/>([^<>]+)</gu)]
  .map((match) => match[1].trim())
  .filter((text) => text && /\p{Script=Han}/u.test(text));
const untranslatedRig = [...new Set(rigStaticTexts.filter((text) => i18n.translateToEnglish(text) === text))];
assert.deepEqual(untranslatedRig, [], `Untranslated rig text: ${untranslatedRig.join(" | ")}`);

const sourceLiterals = [...studioSource.matchAll(/"((?:\\.|[^"\\])*)"/gu)]
  .map((match) => {
    try { return JSON.parse(`"${match[1]}"`); } catch (_) { return ""; }
  })
  .filter((text) => /\p{Script=Han}/u.test(text) && !/[<>]/u.test(text));
const untranslatedLiterals = [...new Set(sourceLiterals.filter((text) => i18n.translateToEnglish(text) === text))];
const intentionalSourceOnly = new Set([
  "失效的过渡帧", "VirtualRoot 动作", "生成进度 ${progress}%", " · 无根运动", "后"
]);
const uncoveredSourceLiterals = untranslatedLiterals.filter((text) => !intentionalSourceOnly.has(text));
assert.deepEqual(uncoveredSourceLiterals, [], `Untranslated studio.js literal: ${uncoveredSourceLiterals.join(" | ")}`);

console.log(`studio i18n contract ok: ${new Set(staticTexts).size} studio + ${new Set(rigStaticTexts).size} rig strings covered; ${new Set(sourceLiterals).size} runtime literals audited`);
