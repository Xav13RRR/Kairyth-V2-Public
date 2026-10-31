"""Offline font provenance checks; fontTools is an audit tool, not a runtime dependency."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FONT_ROOT = ROOT / "motionbuilder_app/static/fonts"


def test_geist_is_exact_reviewed_official_font():
    data = (FONT_ROOT / "GeistSans-Variable.woff2").read_bytes()
    assert len(data) == 58512
    assert hashlib.sha256(data).hexdigest() == (
        "46525b05bbf4527d263f8624ca7024c7d0dce3c64430267d30c31c6dd3821e4d"
    )
    manifest = json.loads((ROOT / "docs/resource-license-evidence.json").read_text(encoding="utf-8"))
    row = next(item for item in manifest["resources"] if item["path"].endswith("GeistSans-Variable.woff2"))
    assert row["publication"] == "redistributable-under-ofl-1.1"
    assert row["license_evidence"]["commit"] == "4d655731f83507f0258ce5541c297b6553d5d081"
    evidence = json.loads((ROOT / "docs/font-provenance-evidence.json").read_text(encoding="utf-8"))
    assert evidence["Geist"]["path"] == row["path"]
    assert evidence["Geist"]["sha256"] == row["sha256"]


def test_geist_includes_both_fixed_release_license_texts():
    expected = {
        "Geist-OFL.txt": "6a873c900f584109b13ae0aaf81d6e3cf0a68751a216b03f7b6c68d547057bb4",
        "Geist-LICENSE.txt": "930853ee1daa68554d9e35c8a9175affb74f699fad9a5da6ee5ebe76379d9137",
    }
    for name, digest in expected.items():
        data = (FONT_ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert b"SIL OPEN FONT LICENSE Version 1.1" in data


def test_oswald_is_not_falsely_called_exact_cdn_or_license_match():
    evidence = json.loads((ROOT / "docs/font-provenance-evidence.json").read_text(encoding="utf-8"))
    assert evidence["Oswald"]["restored"] is False
    assert evidence["Oswald"]["path"] == "motionbuilder_app/static/fonts/Oswald-Bold-Latin.woff2"
    assert evidence["Oswald"]["identity"]["license_name_id_13_present"] is False
    assert evidence["Oswald"]["identity"]["official_cdn_byte_match"] is False
