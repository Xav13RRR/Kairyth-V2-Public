(function () {
  const prefix = window.MKA_API_PREFIX || "";

  function t(key, vars) {
    return window.MKA_I18N ? window.MKA_I18N.t(key, vars) : key;
  }

  function apiUrl(path) {
    const p = path.startsWith("/") ? path : `/${path}`;
    return `${prefix}${p}`;
  }

  function getJobId() {
    return window.__meshprepsJobId || null;
  }

  function getLbsPayload() {
    const fn = window.__meshprepsGetLbsPayload;
    return typeof fn === "function" ? fn() : null;
  }

  function setStatus(text) {
    const el = document.getElementById("exportRigStatus");
    if (el) el.textContent = text || "";
  }

  function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function formatExportError(bytes, fallbackKey) {
    if (!bytes || bytes.byteLength === 0) {
      return `${t(fallbackKey)}${t("bind.export.emptyFile")}`;
    }
    const preview = new TextDecoder().decode(bytes.slice(0, 160));
    return `${t(fallbackKey)}（${bytes.byteLength} 字节）: ${preview}`;
  }

  async function readResponseBytes(res) {
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    if (contentType.includes("application/json")) {
      const err = await res.text();
      throw new Error(err || t("bind.export.jsonNotFile"));
    }
    return res.arrayBuffer();
  }

  async function saveWeights(jobId, payload) {
    const res = await fetch(apiUrl(`/api/job/${jobId}/binding/weights`), {
      method: "PUT",
      cache: "no-store",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lbs_indices: payload.lbs_indices,
        lbs_weights: payload.lbs_weights,
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `${res.status} ${res.statusText}`);
    }
  }

  async function fetchExportGet(jobId, fileFormat) {
    const res = await fetch(
      apiUrl(`/api/job/${jobId}/download?file_format=${encodeURIComponent(fileFormat)}`),
      { method: "GET", cache: "no-store", credentials: "same-origin" }
    );
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `${res.status} ${res.statusText}`);
    }
    return readResponseBytes(res);
  }

  async function fetchExportPost(jobId, payload, fileFormat, path) {
    const body = {
      lbs_indices: payload.lbs_indices,
      lbs_weights: payload.lbs_weights,
      file_format: fileFormat,
    };
    const res = await fetch(apiUrl(path), {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 405) {
      return null;
    }
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `${res.status} ${res.statusText}`);
    }
    return readResponseBytes(res);
  }

  async function requestExportBytes(jobId, payload, fileFormat) {
    const postPaths = [
      `/api/job/${jobId}/export/download`,
      `/api/job/${jobId}/download`,
    ];
    for (const path of postPaths) {
      const bytes = await fetchExportPost(jobId, payload, fileFormat, path);
      if (bytes) {
        return bytes;
      }
    }
    setStatus(t("bind.export.compat"));
    await saveWeights(jobId, payload);
    return fetchExportGet(jobId, fileFormat);
  }

  function validateExportBytes(bytes, isNpz) {
    if (!bytes || bytes.byteLength === 0) {
      throw new Error(formatExportError(bytes, "bind.export.fail"));
    }
    const head = new Uint8Array(bytes.slice(0, 4));
    if (isNpz) {
      if (head[0] !== 0x50 || head[1] !== 0x4b) {
        throw new Error(formatExportError(bytes, "bind.export.invalidNpz"));
      }
      return;
    }
    const text = new TextDecoder().decode(bytes.slice(0, 24));
    if (!text.startsWith("Kaydara FBX Binary") && !text.startsWith("; FBX")) {
      throw new Error(formatExportError(bytes, "bind.export.invalidFbx"));
    }
  }

  async function exportAsset(kind) {
    const jobId = getJobId();
    const payload = getLbsPayload();
    if (!jobId) {
      setStatus(t("bind.export.noJob"));
      return;
    }
    if (!payload || !payload.lbs_indices || !payload.lbs_weights) {
      setStatus(t("bind.export.noWeights"));
      return;
    }

    const isNpz = kind === "npz";
    const fileFormat = isNpz ? "prepared" : "fbx";
    const filename = isNpz ? "prepared_binding.npz" : "binding_rest.fbx";
    const npzBtn = document.getElementById("exportPreparedNpzBtn");
    const fbxBtn = document.getElementById("exportBindingFbxBtn");

    if (npzBtn) npzBtn.disabled = true;
    if (fbxBtn) fbxBtn.disabled = true;
    setStatus(isNpz ? t("bind.export.exportingNpz") : t("bind.export.exportingFbx"));

    try {
      const bytes = await requestExportBytes(jobId, payload, fileFormat);
      validateExportBytes(bytes, isNpz);
      triggerDownload(new Blob([bytes], { type: "application/octet-stream" }), filename);
      setStatus(isNpz ? t("bind.export.npzDone") : t("bind.export.fbxDone"));
    } catch (err) {
      const msg = String(err.message || err);
      if (msg.includes("Method Not Allowed") || msg.includes('"detail":"Method Not Allowed"')) {
        setStatus(t("bind.export.failOld"));
        return;
      }
      setStatus(t("bind.export.fail", { msg }));
    } finally {
      if (npzBtn) npzBtn.disabled = false;
      if (fbxBtn) fbxBtn.disabled = false;
    }
  }

  function wire() {
    const npzBtn = document.getElementById("exportPreparedNpzBtn");
    const fbxBtn = document.getElementById("exportBindingFbxBtn");
    if (!npzBtn || !fbxBtn || npzBtn.dataset.mkaExportWired === "1") {
      return;
    }
    npzBtn.dataset.mkaExportWired = "1";
    fbxBtn.dataset.mkaExportWired = "1";
    npzBtn.addEventListener("click", () => exportAsset("npz"));
    fbxBtn.addEventListener("click", () => exportAsset("fbx"));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
