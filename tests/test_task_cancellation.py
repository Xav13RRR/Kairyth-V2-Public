from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from motionbuilder_app import kimodo_store, stores
from motionbuilder_app.kimodo_store import KimodoMotionStore, _clean_constraints
from motionbuilder_app.stores import CharacterStore, MotionStore


class FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _manifest(root: Path, job: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(job), encoding="utf-8")


class TaskCancellationTests(unittest.TestCase):
    def test_kimodo_backend_preserves_root_heading_constraint(self) -> None:
        payload = [{
            "type": "root2d",
            "frame_indices": [0, 30],
            "smooth_root_2d": [[0.0, 0.0], [1.0, 0.0]],
            "global_root_heading": [[1.0, 0.0], [0.0, 1.0]],
        }]
        cleaned = _clean_constraints(payload)
        self.assertEqual(cleaned[0]["global_root_heading"], payload[0]["global_root_heading"])

    def test_kimodo_process_tree_termination_waits_for_exit(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            creationflags=KimodoMotionStore._creation_flags(),
        )
        try:
            KimodoMotionStore._terminate_process_tree(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()

    def test_kimodo_cancel_hides_job_and_stops_worker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            jobs_root = Path(raw) / "kimodo_motions"
            with mock.patch.object(kimodo_store, "KIMODO_MOTIONS_ROOT", jobs_root):
                store = KimodoMotionStore()
                job = {"id": "kimodo-test", "name": "test", "status": "running", "created_at": 1, "bundle": None}
                _manifest(jobs_root / job["id"], job)
                store.jobs[job["id"]] = job
                process = FakeProcess()
                store.processes[job["id"]] = process
                with mock.patch.object(store, "_terminate_process_tree", side_effect=lambda value: value.terminate()):
                    canceled = store.cancel(job["id"])

                self.assertEqual(canceled["status"], "canceled")
                self.assertTrue(process.terminated)
                self.assertEqual(store.list(), [])
                store.delete(job["id"])
                self.assertFalse((jobs_root / job["id"]).exists())

    def test_kimodo_rename_persists(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            jobs_root = Path(raw) / "kimodo_motions"
            with mock.patch.object(kimodo_store, "KIMODO_MOTIONS_ROOT", jobs_root):
                store = KimodoMotionStore()
                job = {"id": "rename-test", "name": "old", "status": "completed", "created_at": 1, "bundle": "native_kimodo.npz"}
                _manifest(jobs_root / job["id"], job)
                store.jobs[job["id"]] = job

                renamed = store.rename(job["id"], "  new name  ")

                self.assertEqual(renamed["name"], "new name")
                saved = json.loads((jobs_root / job["id"] / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["name"], "new name")

    def test_video_cancel_hides_job_and_waits_for_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            motions_root = root / "motions"
            characters_root = root / "characters"
            with mock.patch.object(stores, "MOTIONS_ROOT", motions_root), mock.patch.object(stores, "CHARACTERS_ROOT", characters_root):
                store = MotionStore(CharacterStore())
                job = {"id": "motion-test", "filename": "input.mp4", "status": "running", "created_at": 1, "log": []}
                _manifest(motions_root / job["id"], job)
                store.jobs[job["id"]] = job
                process = FakeProcess()
                store.processes[job["id"]] = process
                with mock.patch.object(store, "_kill_wsl_group", return_value=None):
                    canceled = store.cancel(job["id"])

                self.assertEqual(canceled["status"], "canceled")
                self.assertTrue(process.terminated)
                self.assertEqual(store.list(), [])


if __name__ == "__main__":
    unittest.main()
