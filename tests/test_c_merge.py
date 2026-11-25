"""Integration boundaries of the C V2 head, without new model fitting or paid calls."""
from pathlib import Path
import itertools
import json
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contest_engine.c_rul import acceptance_guard, remaining_actions
import contest_submission as submission
from contest_engine.model import sha256_file
from scripts.check_contest_104 import Uploads, state_digest
try:
    from tests.test_a_p0 import LocalRound
except ImportError:
    from test_a_p0 import LocalRound

ROOT = Path(__file__).resolve().parents[1]
TEMP = tempfile.TemporaryDirectory


class LocalCRound(LocalRound):
    def __init__(self, root, action="c_bounded"):
        super().__init__(root)
        self.dataset_key = "C"
        self.task = {"task": "regression"}
        public = ROOT / "vendor/reference/C/public"
        def read(name):
            with (public / name).open() as stream:
                yield from (json.loads(line) for line in itertools.islice(stream, 32))
        self.data = SimpleNamespace(train_path=public / "train.jsonl", validation_path=public / "validation.jsonl",
            training_rows=lambda: read("train.jsonl"), validation_rows=lambda: read("validation.jsonl"))
        self.context = {"stage": self.stage_id, "train_sha256": sha256_file(self.data.train_path),
                        "validation_sha256": sha256_file(self.data.validation_path)}
        self.old_memory = {"dataset": "C", "round": 1, "attempts": [],
                           "next_action": action, "policy_context": self.context}

    def submit_candidate(self, folder, **kwargs):
        self.submissions.append((folder, kwargs))
        parent = kwargs["configuration"].get("role") == "paired_parent"
        return {"artifact_id": "test-c-" + str(len(self.submissions)), "summary": {"metrics": {
            "rmse": 50. if parent else 49., "mae": 35. if parent else 34., "nasa_score": 100. if parent else 99.}}}


class CMergeTests(unittest.TestCase):
    def exercise_c_round(self, action, old_stage=False, invalid_reply=False):
        with TEMP() as temporary:
            root = Path(temporary)
            shutil.copyfile(ROOT / "vendor/reference/C/checkpoint.npz", root / "checkpoint.npz")
            rnd = LocalCRound(root, action)
            if old_stage:
                rnd.old_memory.update(next_action="hold", policy_context={**rnd.context, "stage": "old-stage"})
            if invalid_reply:
                def model(messages, **kwargs):
                    rnd.facts = json.loads(messages[-1]["content"])
                    rnd.call_ids.append("local-c-test-call")
                    return SimpleNamespace(text="{invalid", response={"usage": {}})
                rnd.model = model
            with patch.object(submission.tempfile, "TemporaryDirectory", side_effect=lambda prefix, dir: TEMP(prefix=prefix)), \
                 patch("contest_engine.c_rul.train_c_action", side_effect=lambda parent, *args: (parent, {"solver_fits": 0})) as fit:
                submission.train(rnd)
                fit.assert_called_once()
            return rnd

    def test_c_stage_change_resumes_experiment_and_saves_bound_plan(self):
        rnd = self.exercise_c_round("c_bounded", old_stage=True)
        self.assertEqual(rnd.facts["action_resolution"]["reason"], "context_reset")
        self.assertEqual(rnd.facts["action"], "c_bounded")
        self.assertEqual(rnd.saved["policy_context"], rnd.context)
        self.assertEqual(len(rnd.submissions), 2)

    def test_c_bad_reply_retries_measured_candidate_without_ending_in_hold(self):
        rnd = self.exercise_c_round("c_native_small", invalid_reply=True)
        self.assertEqual(rnd.saved["next_action"], "c_native_small")
        self.assertEqual(rnd.saved["next_action_validation"]["reason"], "invalid_proposal_retry_cached_candidate")
        self.assertFalse(rnd.saved["attempts"][-1]["completed"])

    def test_risk_metrics_must_exist_on_both_sides(self):
        old = {"rmse": 50., "mae": 35., "nasa_score": 100.}
        good = {"rmse": 49., "mae": 34., "nasa_score": 99.}
        self.assertTrue(acceptance_guard(old, good)["passed"])
        for metric in ("mae", "nasa_score"):
            for both_missing in (False, True):
                before, after = dict(old), dict(good)
                after.pop(metric)
                if both_missing:
                    before.pop(metric)
                self.assertFalse(acceptance_guard(before, after)["passed"])
            for invalid in (float("nan"), float("inf"), -1):
                self.assertFalse(acceptance_guard(old, {**good, metric: invalid})["passed"])
        self.assertFalse(acceptance_guard(old, {**good, "nasa_score": 106.})["passed"])

    def test_native_epoch_limit_survives_native_hash_change_and_stage_reopens_head(self):
        actions = ["c_bounded", "c_native_small"]
        state = {"c_native_accepted_train": "train", "attempts": [
            {"parent": ["native", None], "train": "train", "action": "c_bounded",
             "context": ["stage1", "val"], "completed": True}]}
        self.assertEqual(remaining_actions(actions, state, "native", "train", ["stage1", "val"]), [])
        self.assertEqual(remaining_actions(actions, state, "native", "train", ["stage2", "val"]), ["c_bounded"])
        self.assertEqual(remaining_actions(actions, state, "updated-native", "train", ["stage1", "val"]), ["c_bounded"])
        self.assertEqual(remaining_actions(actions, state, "native", "new-train", ["stage1", "val"]), actions)

    def test_c_multifile_state_digest_and_ab_state_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "memory.json").write_text("{}")
            before_c, before_a = state_digest(root, "C"), state_digest(root, "A")
            (root / "c_core.npz").write_bytes(b"cache")
            self.assertNotEqual(state_digest(root, "C"), before_c)
            self.assertEqual(state_digest(root, "A"), before_a)
            files = [{"path": name, "size_bytes": 1, "sha256": "sha256:x"} for name in (
                "memory.json", "c_core.npz", "c_native/checkpoint.npz", "c_native/extension.npz", "c_native/recipe.json")]
            uploads = Uploads(root, "C")
            uploads.begin({"kind": "state", "files": files})
            with self.assertRaises(ValueError):
                Uploads(root, "A").begin({"kind": "state", "files": files})


if __name__ == "__main__":
    unittest.main()
