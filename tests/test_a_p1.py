"""A-only P1 tests: plan binding, branch validation, old-state migration."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contest_engine.model import NATIVE, sha256_file
import contest_submission as submission
try:
    from tests.test_a_p0 import LocalRound
except ImportError:
    from test_a_p0 import LocalRound


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "vendor/reference/A/public"
CHECKPOINT = ROOT / "vendor/reference/A" / NATIVE
TemporaryDirectory = tempfile.TemporaryDirectory
CTX = {"stage": "test-stage", "train_sha256": "t", "validation_sha256": "v"}


class PersistentRound(LocalRound):
    """Two-round flow with a carried memory file and a per-round stage."""

    def __init__(self, root, memory_path, round_index, stage_id):
        super().__init__(root)
        self.memory_path = memory_path
        self.round_index = round_index
        self.stage_id = stage_id
        self.context = {"stage": stage_id,
                        "train_sha256": sha256_file(self.data.train_path),
                        "validation_sha256": sha256_file(self.data.validation_path)}

    def state_workspace(self, path):
        path.mkdir()
        if self.memory_path.exists():
            shutil.copyfile(self.memory_path, path / "memory.json")
        return path

    def save_state(self, path):
        self.saved = json.loads((path / "memory.json").read_text(encoding="utf-8"))
        shutil.copyfile(path / "memory.json", self.memory_path)


class PlanBindingTests(unittest.TestCase):
    def test_same_context_hold_persists(self):
        state = {"policy_context": CTX, "next_action": "hold", "attempts": []}
        action, detail = submission.resolve_action(["decay_pair", "adam_small"], state, CTX)
        self.assertEqual(action, "hold")
        self.assertEqual(detail["reason"], "planned_hold")
        self.assertEqual(state["next_action"], "hold")

    def test_context_change_resets_to_first_action(self):
        old = {"stage": "old", "train_sha256": "t1", "validation_sha256": "v1"}
        state = {"policy_context": old, "next_action": "hold", "attempts": []}
        detail = submission.migrate_plan_state(state, CTX, True, 2)
        self.assertEqual(detail["reason"], "context_changed")
        self.assertEqual(detail["invalidated_next_action"], "hold")
        self.assertIsNone(state["next_action"])
        action, resolution = submission.resolve_action(["decay_pair", "adam_small"], state, CTX)
        self.assertEqual(action, "decay_pair")
        self.assertEqual(resolution["reason"], "context_reset")
        self.assertTrue(resolution["reset"])
        self.assertEqual(resolution["reset_policy"], "first_available")
        self.assertEqual(resolution["requested"], "hold")
        self.assertNotIn("next_action_invalidated", state)
        self.assertEqual(state["plan_invalidation"]["reason"], "context_changed")

    def test_context_reset_covers_stage_train_validation(self):
        base = {"stage": "s", "train_sha256": "t", "validation_sha256": "v"}
        for field in ("stage", "train_sha256", "validation_sha256"):
            with self.subTest(field=field):
                changed = {**base, field: base[field] + "-new"}
                state = {"policy_context": base, "next_action": "hold", "attempts": []}
                detail = submission.migrate_plan_state(state, changed, True, 2)
                self.assertEqual(detail["reason"], "context_changed")
                action, resolution = submission.resolve_action(["decay_pair", "adam_small"],
                                                               state, changed)
                self.assertEqual(action, "decay_pair")
                self.assertEqual(resolution["reason"], "context_reset")
                self.assertTrue(resolution["reset"])
                self.assertEqual(resolution["requested"], "hold")

    def test_missing_context_resets_with_archive(self):
        old_attempts = [{"parent": ["a", None], "train": "t", "action": "decay_pair"}]
        old_measured = [{"artifact_id": "x", "files_sha256": ["a", None]}]
        state = {"attempts": old_attempts, "measured_artifacts": old_measured, "next_action": "hold"}
        detail = submission.migrate_plan_state(state, CTX, True, 2)
        self.assertEqual(detail["reason"], "missing_context")
        self.assertEqual(detail["invalidated_next_action"], "hold")
        self.assertEqual(state["attempts"], [])
        self.assertEqual(state["measured_artifacts"], [])
        self.assertEqual(len(state["legacy"]["attempts"]), 1)
        self.assertEqual(state["legacy"]["measured_artifacts"], old_measured)
        actions = submission.remaining_actions(["decay_pair", "ratio_ensemble"], state,
                                               ["parent", None], "t", CTX)
        self.assertEqual(actions, ["decay_pair", "ratio_ensemble"])
        action, resolution = submission.resolve_action(["decay_pair", "ratio_ensemble"], state, CTX)
        self.assertEqual(action, "decay_pair")
        self.assertEqual(resolution["reason"], "missing_context_reset")
        self.assertTrue(resolution["reset"])
        self.assertEqual(resolution["requested"], "hold")


class BranchValidationTests(unittest.TestCase):
    def test_final_branch_validation(self):
        facts = {"next_if_child": ["ratio_ensemble"], "next_if_parent": ["adam_small"]}
        actual, detail = submission.validate_next_action({"next_action": "ratio_ensemble"}, facts, True)
        self.assertEqual(actual, "ratio_ensemble")
        self.assertEqual(detail["reason"], "accepted")
        actual, detail = submission.validate_next_action({"next_action": "adam_small"}, facts, True)
        self.assertEqual(actual, "hold")
        self.assertEqual(detail["reason"], "not_in_child_next_actions")
        self.assertEqual(detail["original"], "adam_small")
        actual, detail = submission.validate_next_action({"next_action": "adam_small"}, facts, False)
        self.assertEqual(actual, "adam_small")
        actual, detail = submission.validate_next_action({"next_action": "ratio_ensemble"}, facts, False)
        self.assertEqual(actual, "hold")
        self.assertEqual(detail["reason"], "not_in_parent_next_actions")
        actual, detail = submission.validate_next_action(None, facts, False)
        self.assertEqual(actual, "hold")
        self.assertEqual(detail["reason"], "invalid_proposal")


class TrainFlowTests(unittest.TestCase):
    def run_local(self, rnd):
        with patch.object(submission.tempfile, "TemporaryDirectory",
                          side_effect=lambda prefix, dir: TemporaryDirectory(prefix=prefix)):
            submission.train(rnd)

    def test_planned_hold_skips_fit_and_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            rnd = LocalRound(root)
            rnd.old_memory = {"dataset": "A", "round": 1, "attempts": [],
                              "next_action": "hold", "policy_context": rnd.context}
            self.run_local(rnd)
        self.assertEqual(rnd.submissions, [])
        self.assertEqual(rnd.facts["action_resolution"]["reason"], "planned_hold")
        self.assertEqual(rnd.finished["selected_artifact_id"], "mounted-parent")

    def test_context_change_resumes_first_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            rnd = LocalRound(root)
            old = {**rnd.context, "stage": "old-stage"}
            rnd.old_memory = {"dataset": "A", "round": 1, "attempts": [],
                              "next_action": "hold", "policy_context": old}
            self.run_local(rnd)
        self.assertEqual(len(rnd.submissions), 2)
        self.assertEqual(rnd.facts["action"], "decay_pair")
        self.assertEqual(rnd.facts["action_resolution"]["reason"], "context_reset")
        self.assertTrue(rnd.facts["action_resolution"]["reset"])
        self.assertEqual(rnd.saved["plan_invalidation"]["reason"], "context_changed")
        self.assertEqual(rnd.saved["plan_invalidation"]["invalidated_next_action"], "hold")

    def test_missing_context_archives_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            rnd = LocalRound(root)
            rnd.old_memory = {"dataset": "A", "round": 1,
                              "attempts": [{"parent": ["old", None], "train": "t",
                                            "action": "decay_pair"}],
                              "next_action": "adam_small"}
            self.run_local(rnd)
        self.assertEqual(len(rnd.submissions), 2)
        self.assertEqual(rnd.facts["state_migration"]["reason"], "missing_context")
        self.assertEqual(rnd.facts["action"], "decay_pair")
        self.assertEqual(rnd.facts["action_resolution"]["reason"], "missing_context_reset")
        self.assertEqual(rnd.saved["legacy"]["attempts"][0]["action"], "decay_pair")
        self.assertEqual([item["action"] for item in rnd.saved["attempts"]], ["decay_pair"])

    def test_train_validates_against_final_child_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            rnd = LocalRound(root)

            def proposal(messages, **kwargs):
                rnd.facts = json.loads(messages[-1]["content"])
                rnd.call_ids.append("local-test-call")
                return SimpleNamespace(
                    text=json.dumps({"selection": "child", "next_action": "toluene_check",
                                     "reason": "branch test"}),
                    response={"usage": {}})

            rnd.model = proposal
            self.run_local(rnd)
        self.assertTrue(rnd.facts["eligible_child"])
        validation = rnd.finished["configuration"]["next_action_validation"]
        self.assertEqual(validation["branch"], "child")
        self.assertEqual(validation["original"], "toluene_check")
        self.assertEqual(validation["actual"], "hold")
        self.assertEqual(validation["reason"], "not_in_child_next_actions")
        self.assertEqual(rnd.saved["next_action"], "hold")

    def test_stage_switch_resumes_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            memory = root / "state.json"
            first = PersistentRound(root, memory, 1, "stage-1")
            self.run_local(first)
            self.assertEqual(len(first.submissions), 2)
            self.assertEqual(first.saved["next_action"], "hold")
            self.assertEqual(first.saved["policy_context"]["stage"], "stage-1")
            second = PersistentRound(root, memory, 2, "stage-2")
            self.run_local(second)
        self.assertEqual(len(second.submissions), 2)
        self.assertEqual(second.facts["action"], "decay_pair")
        self.assertEqual(second.facts["action_resolution"]["reason"], "context_reset")
        self.assertTrue(second.facts["action_resolution"]["reset"])
        self.assertEqual(second.saved["plan_invalidation"]["reason"], "context_changed")
        self.assertEqual(second.saved["plan_invalidation"]["invalidated_next_action"], "hold")
        self.assertEqual(second.saved["policy_context"]["stage"], "stage-2")


if __name__ == "__main__":
    unittest.main()
