const JSON_HEADERS = { "Content-Type": "application/json" };

const API_PREFIX = window.MKA_API_PREFIX || "";
const PIPELINE = Boolean(API_PREFIX);
function apiUrl(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_PREFIX}${p}`;
}

function t(key, vars) {
  return window.MKA_I18N ? window.MKA_I18N.t(key, vars) : key;
}

function ta(text) {
  return window.MKA_I18N ? window.MKA_I18N.translateApi(text) : text;
}

const studioSetupSection = document.getElementById("studioSetupSection");
const studioPlaybackSection = document.getElementById("studioPlaybackSection");
const studioOffsetSection = document.getElementById("studioOffsetSection");

function setStudioPipelineStep(step) {
  if (!PIPELINE) return;
  const steps = {
    setup: studioSetupSection,
    playback: studioPlaybackSection,
    offset: studioOffsetSection,
  };
  for (const [name, el] of Object.entries(steps)) {
    if (!el) continue;
    el.classList.toggle("hidden", name !== step);
  }
}

if (PIPELINE) {
  setStudioPipelineStep("setup");
}

const healthEl = document.getElementById("health");
const canvas = document.getElementById("viewport");
const timelineStatus = document.getElementById("timelineStatus");
const generateStatus = document.getElementById("generateStatus");
const offsetStatus = document.getElementById("offsetStatus");
const animeditsStatus = document.getElementById("animeditsStatus");
const launchAnimEditsBtn = document.getElementById("launchAnimEditsBtn");

/** @type {import("./studio_view.js").StudioView | null} */
let view = null;
/** @type {import("./timeline.js").TimelineEditor | null} */
let timeline = null;

let modelFps = 30;
let motionMeta = null;
let playing = false;
let playFrame = 0;
let playSpeed = 1;
let lastTick = 0;
let frameCache = new Map();
let pollTimer = null;
let modelLoadPollTimer = null;
let offsetSaveTimer = null;
let ws = null;
let showFrameChain = Promise.resolve();
let lastRenderedFrame = -1;

function queueShowFrame(fi) {
  showFrameChain = showFrameChain
    .then(() => showFrame(fi))
    .catch((e) => {
      generateStatus.textContent = t("studio.generate.frameFail", { msg: e.message });
    });
  return showFrameChain;
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}${API_PREFIX}/ws`);
  ws.onclose = () => setTimeout(connectWs, 2000);
  ws.onerror = () => ws?.close();
}
connectWs();

function estimateFrames() {
  if (!timeline) return 0;
  return timeline.durations.reduce((sum, d) => sum + Math.max(1, Math.floor(d * modelFps) - 1), 0);
}

async function api(path, opts = {}) {
  const res = await fetch(apiUrl(path), opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function checkHealth() {
  try {
    const data = await api("/api/health");
    healthEl.textContent = t("studio.connected", { name: data.binding?.split(/[/\\]/).pop() || "binding" });
    healthEl.className = "health ok";
    return true;
  } catch (e) {
    healthEl.textContent = t("studio.healthFail");
    healthEl.className = "health err";
    return false;
  }
}

async function loadScene() {
  const data = await api("/api/studio/scene");
  view.loadScene(data);
  offsetStatus.textContent = t("studio.offset.value", { vals: data.custom_offset.map((v) => v.toFixed(2)).join(", ") });
}

async function initModules() {
  healthEl.textContent = t("studio.load3d");
  try {
    const [{ StudioView }, { TimelineEditor }] = await Promise.all([
      import("./studio_view.js"),
      import("./timeline.js"),
    ]);

    view = new StudioView(canvas);
    timeline = new TimelineEditor(document.getElementById("timelineSegments"), {
      onChange: () => {
        timelineStatus.textContent = t("studio.timeline.segments", {
          n: timeline.prompts.length,
          frames: estimateFrames(),
        });
      },
    });

    view.onOffsetChange = (offset) => {
      offsetStatus.textContent = t("studio.offset.value", { vals: offset.map((v) => v.toFixed(2)).join(", ") });
      clearTimeout(offsetSaveTimer);
      offsetSaveTimer = setTimeout(async () => {
        try {
          await api("/api/studio/custom/offset", {
            method: "PUT",
            headers: JSON_HEADERS,
            body: JSON.stringify({ offset }),
          });
        } catch (e) {
          offsetStatus.textContent = t("studio.offset.saveFail", { msg: e.message });
        }
      }, 300);
    };

    wireEventListeners();
  } catch (e) {
    healthEl.textContent = t("studio.engineFail", { msg: e.message });
    healthEl.className = "health err";
    generateStatus.textContent = t("studio.engineHint");
    throw e;
  }
}

async function loadTimeline() {
  const data = await api("/api/studio/timeline");
  modelFps = data.model_fps || 30;
  timeline.setFromServer(data);
  timelineStatus.textContent = t("studio.timeline.segments", {
    n: data.prompts.length,
    frames: data.total_frames,
  });
}

function setAnimEditsUiEnabled(enabled) {
  launchAnimEditsBtn.disabled = !enabled;
}

async function launchAnimEdits() {
  if (!motionMeta) return;
  playing = false;
  launchAnimEditsBtn.disabled = true;
  animeditsStatus.textContent = t("studio.playback.exportBundle");
  try {
    await api("/api/studio/animedits/launch", {
      method: "POST",
      headers: JSON_HEADERS,
      body: "{}",
    });
    let bundlePath = "";
    while (true) {
      const data = await api("/api/studio/animedits/status");
      if (data.message) {
        animeditsStatus.textContent = ta(data.message);
      }
      if (data.state === "done") {
        bundlePath = data.bundle_path || "";
        if (data.pipeline_mode && window.MKA && bundlePath) {
          await window.MKA.handoffEdit(bundlePath, (msg) => {
            animeditsStatus.textContent = ta(msg);
          });
          return;
        }
        animeditsStatus.textContent = t("studio.playback.animeditsStarted", { url: data.url });
        break;
      }
      if (data.state === "error") {
        throw new Error(data.error || ta(data.message) || t("studio.playback.animeditsFail"));
      }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  } catch (e) {
    animeditsStatus.textContent = t("studio.generate.startFail", { msg: e.message });
  } finally {
    launchAnimEditsBtn.disabled = !motionMeta;
  }
}

function wireEventListeners() {
  document.getElementById("addSegmentBtn").addEventListener("click", () => timeline.addSegment());

  document.getElementById("saveTimelineBtn").addEventListener("click", async () => {
    try {
      const payload = timeline.getPayload();
      const data = await api("/api/studio/timeline", {
        method: "PUT",
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
      modelFps = data.model_fps || modelFps;
      timelineStatus.textContent = t("studio.timeline.saved", { frames: data.total_frames });
    } catch (e) {
      timelineStatus.textContent = t("studio.timeline.saveFail", { msg: e.message });
    }
  });

  document.getElementById("loadModelBtn").addEventListener("click", async () => {
    generateStatus.textContent = t("studio.generate.loadingModel");
    document.getElementById("loadModelBtn").disabled = true;
    try {
      const data = await api("/api/studio/model/load", {
        method: "POST",
        headers: JSON_HEADERS,
        body: "{}",
      });
      if (data.state === "done" && data.model_name) {
        await onModelLoaded(data);
        return;
      }
      if (modelLoadPollTimer) clearInterval(modelLoadPollTimer);
      modelLoadPollTimer = setInterval(pollModelLoadStatus, 1500);
      await pollModelLoadStatus();
    } catch (e) {
      generateStatus.textContent = t("studio.generate.loadFail", { msg: e.message });
      document.getElementById("loadModelBtn").disabled = false;
    }
  });

  document.getElementById("generateBtn").addEventListener("click", async () => {
    generateStatus.textContent = t("studio.generate.preparing");
    document.getElementById("generateBtn").disabled = true;
    try {
      await api("/api/studio/timeline", {
        method: "PUT",
        headers: JSON_HEADERS,
        body: JSON.stringify(timeline.getPayload()),
      });
      await api("/api/studio/generate", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({ async_mode: true }),
      });
      generateStatus.textContent = t("studio.generate.running");
      pollTimer = setInterval(pollGenerateStatus, 1500);
      pollGenerateStatus();
    } catch (e) {
      generateStatus.textContent = t("studio.generate.startFail", { msg: e.message });
      document.getElementById("generateBtn").disabled = false;
    }
  });

  document.getElementById("playbackFrame").addEventListener("input", (ev) => {
    playFrame = Number(ev.target.value);
    queueShowFrame(playFrame);
  });

  document.getElementById("playbackSpeed").addEventListener("input", (ev) => {
    playSpeed = Number(ev.target.value) / 100;
  });

  document.getElementById("playBtn").addEventListener("click", () => {
    if (!motionMeta) return;
    playing = true;
    lastTick = performance.now();
  });

  document.getElementById("pauseBtn").addEventListener("click", () => {
    playing = false;
  });

  launchAnimEditsBtn.addEventListener("click", launchAnimEdits);

  document.getElementById("exportNpzBtn").addEventListener("click", () => {
    window.location.href = "/api/studio/export/npz";
  });

  document.getElementById("exportFbxBtn").addEventListener("click", () => {
    window.location.href = "/api/studio/export/fbx";
  });
}

async function onModelLoaded(data) {
  modelFps = data.model_fps || modelFps;
  if (data.default_preview) {
    view.loadScene({
      default: data.default_preview,
      custom: (await api("/api/studio/scene")).custom,
      custom_offset: view.getCustomOffset(),
    });
  }
  generateStatus.textContent = t("studio.generate.modelLoaded", { name: data.model_name, fps: modelFps });
  document.getElementById("loadModelBtn").disabled = false;
}

async function pollModelLoadStatus() {
  try {
    const data = await api("/api/studio/model/load/status");
    generateStatus.textContent = ta(data.message) || data.state;
    if (data.state === "running") return;
    if (modelLoadPollTimer) {
      clearInterval(modelLoadPollTimer);
      modelLoadPollTimer = null;
    }
    if (data.state === "done") {
      await onModelLoaded(data);
    } else if (data.state === "error") {
      generateStatus.textContent = t("studio.generate.loadFail", {
        msg: ta(data.error) || ta(data.message),
      });
      document.getElementById("loadModelBtn").disabled = false;
    }
  } catch (e) {
    generateStatus.textContent = t("studio.generate.statusFail", { msg: e.message });
  }
}

async function pollGenerateStatus() {
  try {
    const data = await api("/api/studio/generate/status");
    generateStatus.textContent = ta(data.message) || data.state;
    if (data.state === "running") return;
    clearInterval(pollTimer);
    pollTimer = null;
    document.getElementById("generateBtn").disabled = false;
    if (data.state === "done" && data.motion) {
      await onMotionReady(data.motion);
      generateStatus.textContent = t("studio.generate.done", { frames: data.motion.num_frames });
    } else if (data.state === "error") {
      generateStatus.textContent = t("studio.generate.fail", {
        msg: ta(data.error) || ta(data.message),
      });
    }
  } catch (e) {
    generateStatus.textContent = t("studio.generate.statusFail", { msg: e.message });
  }
}

async function onMotionReady(meta) {
  motionMeta = meta;
  frameCache.clear();
  playFrame = 0;
  playing = false;
  const slider = document.getElementById("playbackFrame");
  slider.max = String(Math.max(0, meta.num_frames - 1));
  slider.value = "0";
  setAnimEditsUiEnabled(true);
  if (PIPELINE) {
    studioSetupSection?.classList.add("hidden");
    studioPlaybackSection?.classList.remove("hidden");
    studioOffsetSection?.classList.remove("hidden");
  }
  await queueShowFrame(0);
}

async function fetchFrameRange(from, to, target) {
  const key = `${target}:${from}-${to}`;
  if (frameCache.has(key)) return frameCache.get(key);
  const data = await api(
    `/api/studio/motion/frames?from_frame=${from}&to_frame=${to}&target=${target}`,
  );
  frameCache.set(key, data.frames);
  return data.frames;
}

async function showFrame(fi) {
  if (!motionMeta) return;
  const batch = 8;
  const start = Math.floor(fi / batch) * batch;
  const end = Math.min(motionMeta.num_frames, start + batch);
  const local = fi - start;
  const [defaultFrames, customFrames] = await Promise.all([
    fetchFrameRange(start, end, "default"),
    fetchFrameRange(start, end, "custom"),
  ]);
  if (defaultFrames[local]) {
    view.setDefaultMeshVertices(defaultFrames[local].flat());
  }
  if (customFrames[local]) {
    view.setCustomMeshVertices(customFrames[local].flat());
  }
  document.getElementById("playbackFrame").value = String(fi);
  lastRenderedFrame = fi;
}

function tick(now) {
  requestAnimationFrame(tick);
  view?.render();
  if (playing && motionMeta) {
    const dt = (now - lastTick) / 1000;
    lastTick = now;
    playFrame += dt * motionMeta.fps * playSpeed;
    if (playFrame >= motionMeta.num_frames) playFrame = 0;
    const fi = Math.floor(playFrame);
    if (fi !== lastRenderedFrame) {
      queueShowFrame(fi);
    }
  }
}

async function bootstrap() {
  const ok = await checkHealth();
  if (!ok) return;

  try {
    await initModules();
    healthEl.textContent = t("studio.loadScene");
    await loadScene();
    await loadTimeline();
    const health = await api("/api/health");
    if (health.has_motion) {
      const meta = await api("/api/studio/motion");
      await onMotionReady(meta);
    } else {
      setAnimEditsUiEnabled(false);
      if (PIPELINE) setStudioPipelineStep("setup");
    }
    await checkHealth();
  } catch (e) {
    generateStatus.textContent = t("studio.generate.initFail", { msg: e.message });
  }
  requestAnimationFrame(tick);
}

bootstrap();

window.MKA_I18N?.onChange(() => {
  window.MKA_I18N.applyDom();
  if (timeline) timeline.render();
});
