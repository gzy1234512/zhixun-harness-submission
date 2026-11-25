"""Predict-time mount validation must stay safe after the once-per-process cache.

The C timeout fix moved the /model descriptor probes out of the per-row path.
These tests pin the boundary that made the move acceptable: every rejection the
per-row validation performed still fires, and the underlying work runs once.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import contest_submission as submission
from contest_engine.model import load

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "vendor/reference/C/checkpoint.npz"
CONTEXT = {"protocol": "contest.predict-context.v1", "task": "regression", "dataset_key": "C",
           "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                        "predict_argv": ["python3", "/app/harness.py", "predict"]},
           "artifact_id": "test-c-predict"}


def sample(context=None, task="regression", dataset_key="C"):
    rows = (ROOT / "vendor/reference/C/public/validation.jsonl").read_text().splitlines()
    return SimpleNamespace(task=task, inputs=json.loads(rows[0])["inputs"], context=context,
                           dataset_key=dataset_key)


class PredictTargetTests(unittest.TestCase):
    def setUp(self):
        submission.predict_target.cache_clear()
        submission.predictor.cache_clear()
        self.addCleanup(submission.predict_target.cache_clear)
        self.addCleanup(submission.predictor.cache_clear)

    def mounted(self, root, extension=False, symlink_checkpoint=False):
        if symlink_checkpoint:
            (root / "checkpoint.npz").symlink_to(CHECKPOINT)
        else:
            shutil.copyfile(CHECKPOINT, root / "checkpoint.npz")
        if extension:
            shutil.copyfile(CHECKPOINT, root / "extension.npz")
        return root

    def test_validation_runs_once_per_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.mounted(Path(folder), extension=True)
            first = submission.predict_target(str(root), None)
            path, mounted_as, extension, has_extension = first
            self.assertEqual(mounted_as, "checkpoint.npz")
            self.assertEqual(extension, root / "extension.npz")
            self.assertTrue(has_extension)
            self.assertEqual(submission.predict_target.cache_info().misses, 1)
            self.assertIs(submission.predict_target(str(root), None), first)
            self.assertEqual(submission.predict_target.cache_info().hits, 1)
            self.assertEqual(submission.predict_target.cache_info().misses, 1)
        with tempfile.TemporaryDirectory() as folder:
            root = self.mounted(Path(folder))
            _, _, _, has_extension = submission.predict_target(str(root), None)
            self.assertFalse(has_extension)

    def test_symlinked_checkpoint_is_rejected_and_never_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.mounted(Path(folder), symlink_checkpoint=True)
            with self.assertRaises(ValueError):
                submission.predict_target(str(root), None)
            with self.assertRaises(ValueError):
                submission.predict(sample(CONTEXT), root)
            self.assertEqual(submission.predict_target.cache_info().currsize, 0)

    def test_artifact_path_outside_the_mount_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.mounted(Path(folder))
            for escape in (str(CHECKPOINT), "../checkpoint.npz", "../escape.npz", "/etc/passwd"):
                with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": escape}):
                    with self.assertRaises(ValueError):
                        submission.predict(sample(), root)

    def test_symlinked_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.mounted(Path(folder))
            (root / "extension.npz").symlink_to(CHECKPOINT)
            with self.assertRaises(ValueError):
                submission.predict(sample(CONTEXT), root)

    def test_descriptor_must_name_the_mounted_checkpoint_and_extension(self):
        for declared, present in ((["extension.npz"], True), (["checkpoint.npz", "extension.npz"], False),
                                  (["checkpoint.npz"], True), (["../checkpoint.npz"], False)):
            with tempfile.TemporaryDirectory() as folder:
                root = self.mounted(Path(folder), extension=present)
                context = {**CONTEXT, "artifact": {**CONTEXT["artifact"], "checkpoint_paths": declared}}
                with self.assertRaises(ValueError):
                    submission.predict(sample(context), root)

    def test_cached_validation_does_not_change_the_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.mounted(Path(folder))
            row = sample()
            submitted = submission.predict(row, root)
            native = load(root / "checkpoint.npz", dataset="C")
            direct = float(native.predict(np.asarray(
                [submission.input_vector(row.inputs, native.native)], dtype=np.float64))[0])
            self.assertIsInstance(submitted, dict)
            self.assertEqual(set(submitted), {"value"})
            self.assertAlmostEqual(submitted["value"], direct, places=12)
            self.assertEqual(submitted, submission.predict(row, root))
            with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": "checkpoint.npz"}):
                relative = submission.predict(row, root)
            self.assertEqual(relative, submitted)


if __name__ == "__main__":
    unittest.main()
