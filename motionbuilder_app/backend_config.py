# SPDX-License-Identifier: Apache-2.0
# Original configuration bridge only. No rights are granted to external programs.
"""Optional research backends: physical configuration checks, never model imports."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from .release_profile import capabilities, unavailable_backend_status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_PROTOCOL = "kairyth-hybrik/v1"


def _path(value: str | Path, project: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


@dataclass(frozen=True)
class HybrikBackend:
    root: Path
    python: Path
    worker: Path
    source: str

    def status(self) -> dict:
        unavailable = unavailable_backend_status("hybrik")
        if unavailable is not None: return unavailable
        roots = [self.root / "src/HybrIK", self.root / "src", self.root]
        repo = next((path for path in roots if (path / "hybrik/models/HRNetWithCam.py").is_file()), roots[0])
        required = [self.python, self.worker, repo / "hybrik/models/HRNetWithCam.py",
                    repo / "hybrik/models/layers/hrnet/w48.yaml", repo / "hybrik/models/layers/smpl/SMPL.py",
                    repo / "hybrik/models/layers/smpl/lbs.py", repo / "configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml",
                    self.root / "pretrained_models/hybrik_hrnet.pth", self.root / "model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                    self.root / "model_files/J_regressor_h36m.npy", self.root / "model_files/h36m_mean_beta.npy"]
        missing = [str(path) for path in required if not path.is_file()]
        ready = not missing
        return {"backend": "hybrik", "protocol": POSE_PROTOCOL, "state": "ready" if ready else "not_ready",
                "ready": ready, "installed": ready, "configuration_source": self.source,
                "runtime_root": str(self.root), "python": str(self.python), "worker": str(self.worker),
                "checkpoint": str(self.root / "pretrained_models/hybrik_hrnet.pth"),
                "checkpoint_installed": required[-4].is_file(), "smpl_model": str(required[-3]),
                "smpl_model_installed": required[-3].is_file(), "code_installed": (repo / "hybrik").is_dir(),
                "missing": missing, "runtime_verified": False, "rights_verified": False,
                "message": "Physical files present; interpreter/model/permission not verified" if ready else "Research backend not ready: missing configuration or resources"}


def hybrik_backend(project: Path | None = None) -> HybrikBackend:
    project = Path(project or PROJECT_ROOT).resolve()
    return HybrikBackend(_path(os.environ.get("KAIRYTH_HYBRIK_ROOT") or project / "hybrik_runtime", project),
        _path(os.environ.get("KAIRYTH_HYBRIK_PYTHON") or sys.executable, project),
        _path(os.environ.get("KAIRYTH_HYBRIK_WORKER") or project / "research_backends/hybrik_worker.py", project),
        "explicit" if any(os.environ.get(key) for key in ("KAIRYTH_HYBRIK_ROOT", "KAIRYTH_HYBRIK_PYTHON", "KAIRYTH_HYBRIK_WORKER")) else "legacy_project_local")


@dataclass(frozen=True)
class GvhmrBackend:
    root: Path
    python: str
    distro: str
    extractor: Path
    detector_root: Path
    detector_python: str
    source: str

    def status(self, *, cached_pose: bool = False, cached_features: bool = False) -> dict:
        unavailable = unavailable_backend_status("gvhmr")
        if unavailable is not None: return unavailable
        required = [self.root / "tools/demo/demo.py", self.root / "hmr4d/utils/smplx_utils.py", self.extractor,
            self.detector_root / "cli.py", self.root / "inputs/checkpoints/yolo/yolov8x.pt",
            self.root / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
            self.root / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
            self.root / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"]
        # Confirmed default inference/render chain, not training/evaluation,
        # kid models or optional DPVO assets. Never starts an interpreter.
        required.extend(self.root / "hmr4d/utils/body_model" / name for name in (
            "__init__.py", "body_model.py", "body_model_smplh.py", "body_model_smplx.py", "smplx_lite.py",
            "coco_aug_dict.pth", "smpl_coco17_J_regressor.pt", "smplx_verts437.pt",
            "smplx2smpl_sparse.pt", "smpl_neutral_J_regressor.pt"))
        if not cached_pose: required.append(self.root / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth")
        if not cached_features:
            required.extend((self.root / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
                             self.root / "hmr4d/network/hmr2/configs/smpl_mean_params.npz"))
        missing = [str(path) for path in required if not path.is_file()]
        entry = self.root / "tools/demo/demo.py"
        compatible = False
        if entry.is_file() and entry.stat().st_size <= 1024*1024:
            text = entry.read_text(encoding="utf-8", errors="replace")
            compatible = all(flag in text for flag in ("--prepare_only", "--tracking_file"))
        if not compatible: missing.append("compatible GVHMR prepare_only/tracking_file interface")
        if not self.python.startswith("/"): missing.append("absolute Linux MOTIONBUILDER_WSL_PYTHON")
        if not self.detector_python.startswith("/"): missing.append("absolute Linux MOTIONBUILDER_DETECTOR_PYTHON")
        if not self.distro.strip(): missing.append("MOTIONBUILDER_WSL_DISTRO")
        return {"backend": "gvhmr", "state": "ready" if not missing else "not_ready", "ready": not missing,
                "installed": not missing, "configuration_source": self.source, "runtime_root": str(self.root),
                "python": self.python, "wsl_distro": self.distro, "extractor": str(self.extractor),
                "detector_root": str(self.detector_root), "detector_python": self.detector_python,
                "missing": missing, "runtime_verified": False, "rights_verified": False,
                "message": "Physical configuration only; no WSL startup, GPU or authorization probe"}


def gvhmr_backend(project: Path | None = None, *, default_root: Path | None = None,
                   default_python: str | None = None, default_distro: str | None = None,
                   default_detector_python: str | None = None, default_detector_root: Path | None = None) -> GvhmrBackend:
    project = Path(project or PROJECT_ROOT).resolve()
    return GvhmrBackend(_path(os.environ.get("KAIRYTH_GVHMR_ROOT") or default_root or project / "GVHMR", project),
        os.environ.get("MOTIONBUILDER_WSL_PYTHON") or default_python or "/home/gvhmr/miniforge3/envs/gvhmr/bin/python",
        os.environ.get("MOTIONBUILDER_WSL_DISTRO") or default_distro or "Kairyth-V2-GVHMR",
        _path(os.environ.get("KAIRYTH_GVHMR_EXTRACTOR") or project / "motionbuilder_app/wsl_extract.py", project),
        _path(os.environ.get("KAIRYTH_DETECTOR_ROOT") or default_detector_root or project / "detector_tool", project),
        os.environ.get("MOTIONBUILDER_DETECTOR_PYTHON") or default_detector_python or "/home/gvhmr/miniforge3/envs/kairyth-detector/bin/python",
        "explicit" if any(os.environ.get(key) for key in ("KAIRYTH_GVHMR_ROOT", "KAIRYTH_GVHMR_EXTRACTOR",
            "KAIRYTH_DETECTOR_ROOT", "MOTIONBUILDER_WSL_PYTHON", "MOTIONBUILDER_WSL_DISTRO", "MOTIONBUILDER_DETECTOR_PYTHON")) else "legacy_project_local")


def require_backend(status: dict) -> None:
    if not status["ready"]:
        raise RuntimeError("Research backend not ready: " + "; ".join(status["missing"]))


def research_backend_status() -> dict:
    caps = capabilities()
    return {"hybrik": unavailable_backend_status("hybrik") or hybrik_backend().status(),
            "gvhmr": unavailable_backend_status("gvhmr") or gvhmr_backend().status(),
            "release_profile": caps["release_profile"], "capabilities": caps,
            "scope": "Physical readiness only; process separation is not rights clearance"}
