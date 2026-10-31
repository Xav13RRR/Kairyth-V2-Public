export class TimelineEditor {
  constructor(container, { onChange } = {}) {
    this.container = container;
    this.onChange = onChange;
    this.prompts = ["A person walks forward."];
    this.durations = [6.0];
    this.render();
  }

  _t(key, vars) {
    return window.MKA_I18N ? window.MKA_I18N.t(key, vars) : key;
  }

  render() {
    this.container.innerHTML = "";
    for (let i = 0; i < this.prompts.length; i++) {
      const row = document.createElement("div");
      row.className = "segment-row";
      row.innerHTML = `
        <div class="segment-fields">
          <textarea rows="2" data-idx="${i}" class="seg-prompt">${this.prompts[i]}</textarea>
          <div class="segment-meta">
            <label class="seg-dur-label">${this._t("studio.timeline.sec")}<input type="number" step="0.5" min="2" max="10" data-idx="${i}" class="seg-dur" value="${this.durations[i]}" /></label>
            <button type="button" class="secondary seg-delete" data-idx="${i}">${this._t("studio.timeline.delete")}</button>
          </div>
        </div>
      `;
      this.container.appendChild(row);
    }
    this.container.querySelectorAll(".seg-prompt").forEach((el) => {
      el.addEventListener("input", () => this._syncFromDom());
    });
    this.container.querySelectorAll(".seg-dur").forEach((el) => {
      el.addEventListener("input", () => this._syncFromDom());
    });
    this.container.querySelectorAll(".seg-delete").forEach((el) => {
      el.addEventListener("click", () => this.removeSegment(Number(el.dataset.idx)));
    });
  }

  _syncFromDom() {
    const prompts = [];
    const durations = [];
    this.container.querySelectorAll(".seg-prompt").forEach((el) => {
      prompts.push(el.value.trim() || "A person walks forward.");
    });
    this.container.querySelectorAll(".seg-dur").forEach((el) => {
      durations.push(Number(el.value) || 6);
    });
    this.prompts = prompts;
    this.durations = durations;
    this.onChange?.({ prompts: this.prompts, durations_sec: this.durations });
  }

  addSegment() {
    this.prompts.push("A person walks forward.");
    this.durations.push(3.0);
    this.render();
    this.onChange?.({ prompts: this.prompts, durations_sec: this.durations });
  }

  removeSegment(index) {
    if (this.prompts.length <= 1) {
      return false;
    }
    this.prompts.splice(index, 1);
    this.durations.splice(index, 1);
    this.render();
    this.onChange?.({ prompts: this.prompts, durations_sec: this.durations });
    return true;
  }

  setFromServer(data) {
    this.prompts = [...data.prompts];
    this.durations = [...data.durations_sec];
    this.render();
  }

  getPayload() {
    this._syncFromDom();
    return { prompts: this.prompts, durations_sec: this.durations };
  }
}

if (typeof window !== "undefined") {
  window.MKA_I18N?.onChange(() => {
    document.querySelectorAll(".segment-row").forEach(() => {});
  });
}
