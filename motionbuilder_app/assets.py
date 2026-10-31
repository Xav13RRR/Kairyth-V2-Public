import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
# A separately obtained, authorized skeleton directory can stay outside Git.
# This path override is not a grant of rights to any body-model-derived data.
SKELETONS_ROOT = Path(os.environ.get("KAIRYTH_SKELETONS_ROOT") or ASSETS_ROOT / "skeletons")


def skeleton_asset_path(*parts: str) -> Path:
    path = SKELETONS_ROOT.joinpath(*parts)
    if parts and Path(parts[-1]).suffix and not path.is_file():
        raise FileNotFoundError(
            f"Required skeleton asset is missing: {path}. "
            "See docs/RESOURCE_READINESS.md and run scripts/resource_readiness.py. "
            "Supply the exact authorized asset via KAIRYTH_SKELETONS_ROOT; "
            "a replacement skeleton is not generated automatically."
        )
    return path
