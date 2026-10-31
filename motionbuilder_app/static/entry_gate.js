(function () {
  "use strict";

  const gate = document.getElementById("entryGate");
  const enterButton = document.getElementById("entryGateEnter");
  if (!gate || !enterButton) {
    document.body.classList.remove("entry-locked");
    return;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let exiting = false;

  const finish = function () {
    gate.classList.add("is-hidden");
    gate.setAttribute("aria-hidden", "true");
    document.body.classList.remove("entry-locked");
  };

  const enter = function () {
    if (exiting) return;
    exiting = true;
    gate.classList.add("is-exiting");
    enterButton.disabled = true;
    window.setTimeout(finish, reducedMotion ? 20 : 1180);
  };

  enterButton.addEventListener("click", enter, { once: true });
  window.requestAnimationFrame(function () {
    window.requestAnimationFrame(function () {
      gate.classList.add("is-ready");
      enterButton.focus({ preventScroll: true });
    });
  });
})();
