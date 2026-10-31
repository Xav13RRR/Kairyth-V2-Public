(function () {
  const prefix = window.MKA_API_PREFIX || "";

  function t(key, vars) {
    return window.MKA_I18N ? window.MKA_I18N.t(key, vars) : key;
  }

  function apiUrl(path) {
    const p = path.startsWith("/") ? path : `/${path}`;
    return `${prefix}${p}`;
  }

  function setVisible(el, visible) {
    if (el) el.classList.toggle("hidden", !visible);
  }

  function setViewportInfo(text) {
    if (typeof window.__meshprepsSetViewportInfo === "function") {
      window.__meshprepsSetViewportInfo(text);
      return;
    }
    const el = document.getElementById("viewportInfo");
    if (!el) return;
    el.textContent = text || "";
    el.classList.toggle("hidden", !text);
  }

  function setUploadIdle() {
    setVisible(document.getElementById("dropzoneIdle"), true);
    setVisible(document.getElementById("dropzoneReady"), false);
    setVisible(document.getElementById("uploadActions"), false);
    const status = document.getElementById("uploadStatus");
    if (status) status.textContent = "";
    const dropzone = document.getElementById("dropzone");
    if (dropzone) dropzone.classList.remove("dropzone-ready");
    setViewportInfo("");
    window.__meshprepsPendingUpload = null;
    window.dispatchEvent(new CustomEvent("meshpreps:upload-reset"));
  }

  function setUploadReady(data) {
    const p = data.preview || {};
    const name = p.source || p.source_path || t("bind.upload.model");
    const stats = t("bind.upload.vertsFaces", {
      verts: p.n_verts || 0,
      faces: p.n_faces || 0,
    });
    setVisible(document.getElementById("dropzoneIdle"), false);
    setVisible(document.getElementById("dropzoneReady"), true);
    setVisible(document.getElementById("uploadActions"), true);
    const dropzone = document.getElementById("dropzone");
    if (dropzone) dropzone.classList.add("dropzone-ready");
    const status = document.getElementById("uploadStatus");
    if (status) status.textContent = "";
    setViewportInfo(`${name}\n${stats}`);
    window.__meshprepsPendingUpload = data;
  }

  async function uploadFile(file) {
    const status = document.getElementById("uploadStatus");
    if (!file) return;
    if (status) status.textContent = "";
    setViewportInfo("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(apiUrl("/api/upload"), { method: "POST", body: fd });
      if (!res.ok) {
        throw new Error((await res.text()) || res.statusText);
      }
      const data = await res.json();
      setUploadReady(data);
      window.dispatchEvent(new CustomEvent("meshpreps:uploaded", { detail: data }));
    } catch (err) {
      const message = t("bind.upload.uploadFail", { msg: err.message });
      if (status) status.textContent = message;
      setViewportInfo(message);
    }
  }

  function wire() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const pickBtn = document.getElementById("pickFileBtn");
    const reuploadBtn = document.getElementById("reuploadBtn");
    const confirmBtn = document.getElementById("confirmUploadBtn");
    if (!dropzone || !fileInput || dropzone.dataset.mkaUploadWired === "1") {
      return;
    }
    dropzone.dataset.mkaUploadWired = "1";

    pickBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    dropzone.addEventListener("click", (e) => {
      if (dropzone.classList.contains("dropzone-ready")) return;
      if (e.target === pickBtn || pickBtn?.contains(e.target)) return;
      fileInput.click();
    });

    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0];
      if (f) uploadFile(f);
      fileInput.value = "";
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (!dropzone.classList.contains("dropzone-ready")) {
        dropzone.classList.add("dragover");
      }
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) uploadFile(f);
    });

    reuploadBtn?.addEventListener("click", () => {
      setUploadIdle();
      fileInput.value = "";
      fileInput.click();
    });

    confirmBtn?.addEventListener("click", () => {
      if (!window.__meshprepsPendingUpload) return;
      window.dispatchEvent(
        new CustomEvent("meshpreps:upload-confirmed", { detail: window.__meshprepsPendingUpload }),
      );
    });
  }

  window.__meshprepsUploadBoot = true;
  window.__meshprepsSetUploadIdle = setUploadIdle;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }

  window.MKA_I18N?.onChange(() => {
    window.MKA_I18N.applyDom();
    if (window.__meshprepsPendingUpload) setUploadReady(window.__meshprepsPendingUpload);
  });
})();
