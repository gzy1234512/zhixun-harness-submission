"""A-only regression tests; model calls are local stubs and never use a provider."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import contest_submission as submission
from contest_engine.model import CLASSES, EXTENSION, NATIVE, load, save, sha256_file, source_rows, train_action


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "vendor/reference/A/public"
CHECKPOINT = ROOT / "vendor/reference/A" / NATIVE
TemporaryDirectory = tempfile.TemporaryDirectory


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


class LocalRound:
    """Only the SDK surface used by train(), with deliberately separate host metrics."""

    def __init__(self, root, action="adam_small", measured=None):
        self.dataset_key, self.round_index, self.stage_id = "A", 2, "test-stage"
        self.task, self.wall_seconds = {"task": "classification"}, 240
        self.model_name, self.max_output_tokens = "local-test", 224
        self.parent = SimpleNamespace(
            descriptor={"checkpoint_paths": [NATIVE]}, path=lambda name: root / name,
            artifact_id="mounted-parent", checkpoint_sha256=sha256_file(root / NATIVE))
        self.data = SimpleNamespace(
            train_path=PUBLIC / "train.jsonl", validation_path=PUBLIC / "validation.jsonl",
            training_rows=lambda: rows(PUBLIC / "train.jsonl"),
            validation_rows=lambda: rows(PUBLIC / "validation.jsonl"))
        # P1 binds the next plan to this context; P1's old-state migration is covered in test_a_p1.
        self.context = {"stage": self.stage_id,
                        "train_sha256": sha256_file(self.data.train_path),
                        "validation_sha256": sha256_file(self.data.validation_path)}
        self.old_memory = {"dataset": "A", "round": 1, "attempts": [], "next_action": action,
                           "policy_context": self.context}
        if measured is not None:
            self.old_memory["measured_artifacts"] = measured
        self.call_ids, self.submissions = [], []

    def history(self):
        return []

    def state_workspace(self, path):
        path.mkdir()
        (path / "memory.json").write_text(json.dumps(self.old_memory), encoding="utf-8")
        return path

    def submit_candidate(self, folder, **kwargs):
        self.submissions.append((folder, kwargs))
        score = .8 if kwargs["configuration"].get("role") == "paired_parent" else .81
        return {"artifact_id": "test-artifact-" + str(len(self.submissions)),
                "summary": {"metrics": {"macro_f1": score}}}

    def model(self, messages, **kwargs):
        self.facts = json.loads(messages[-1]["content"])
        self.call_ids.append("local-test-call")
        proposal = {"selection": "child" if self.facts["eligible_child"] else "parent",
                    "next_action": "hold", "reason": "local A regression test"}
        return SimpleNamespace(text=json.dumps(proposal), response={"usage": {}})

    def save_state(self, path):
        self.saved = json.loads((path / "memory.json").read_text(encoding="utf-8"))

    def finish(self, **kwargs):
        self.finished = kwargs


class ClassificationFactsTests(unittest.TestCase):
    def test_matches_independent_sklearn_and_exact_mcnemar(self):
        truth = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 4])
        old = np.array([3, 3, 3, 3, 0, 3, 0, 0, 3, 3, 4, 4])
        new = np.array([0, 0, 0, 0, 3, 3, 0, 0, 3, 3, 5, 4])
        native = SimpleNamespace(meta={"classes": list(CLASSES)})
        parent = SimpleNamespace(native=native, predict=lambda x: np.eye(6)[old])
        child = SimpleNamespace(predict=lambda x: np.eye(6)[new])
        with patch.object(submission, "source_rows", return_value=(np.zeros((12, 128)), truth)):
            facts = submission.classification_facts(parent, child, [])
        for key, predicted in (("parent", old), ("child", new)):
            _, recall, f1, support = precision_recall_fscore_support(
                truth, predicted, labels=np.arange(6), zero_division=0)
            for index, label in enumerate(CLASSES):
                actual = facts[key]["per_class"][label]
                self.assertAlmostEqual(actual["f1"], f1[index])
                self.assertAlmostEqual(actual["recall"], recall[index])
                self.assertEqual(actual["support"], support[index])
            self.assertEqual(facts[key]["confusion_matrix"],
                             confusion_matrix(truth, predicted, labels=np.arange(6)).tolist())
        self.assertEqual(facts["changed_predictions"], 6)
        self.assertEqual(facts["mcnemar"]["improved"], 4)
        self.assertEqual(facts["mcnemar"]["worsened"], 1)
        self.assertAlmostEqual(facts["mcnemar"]["p_value"], .375)
        self.assertEqual(facts["toluene_support"], 1)
        submission.json_text(facts)

    def test_identical_predictions_have_no_discordance(self):
        native = SimpleNamespace(meta={"classes": list(CLASSES)})
        model = SimpleNamespace(native=native, predict=lambda x: np.eye(6)[[0, 3]])
        with patch.object(submission, "source_rows", return_value=(np.zeros((2, 128)), np.array([0, 3]))):
            facts = submission.classification_facts(model, model, [])
        self.assertEqual(facts["changed_predictions"], 0)
        self.assertEqual(facts["mcnemar"]["p_value"], 1.)
        self.assertEqual(facts["parent"]["per_class"]["Toluene"]["support"], 0)


class DuplicateTests(unittest.TestCase):
    def test_parent_bytes_skip_without_historical_context(self):
        found = submission.duplicate_candidate(["a", None], ["a", None], "parent", [], None)
        self.assertEqual(found, {"kind": "duplicate_of_parent", "artifact_id": "parent"})

    def test_historical_match_scoped_to_stage_and_validation(self):
        context = {"stage": "one", "validation_sha256": "v1"}
        measured = [{"evaluation_context": context, "files_sha256": ["old", "side"],
                     "artifact_id": "already-measured", "round": 1}]
        args = (["old", "side"], ["current", None], "parent", measured)
        self.assertEqual(submission.duplicate_candidate(*args, context)["kind"], "duplicate_of_artifact")
        for changed in (None, {"stage": "two", "validation_sha256": "v1"},
                        {"stage": "one", "validation_sha256": "v2"}):
            self.assertIsNone(submission.duplicate_candidate(*args, changed))

    def test_action_attempts_do_not_block_changed_evaluation(self):
        context = {"stage": "one", "validation_sha256": "v1"}
        state = {"attempts": [{"parent": ["a", None], "train": "t", "action": "ratio_ensemble",
                               "evaluation_context": context}]}
        args = (["ratio_ensemble"], state, ["a", None], "t")
        self.assertEqual(submission.remaining_actions(*args, context), [])
        self.assertEqual(submission.remaining_actions(*args, {**context, "stage": "two"}), ["ratio_ensemble"])
        # B/C callers omit the new argument and keep their existing behavior.
        self.assertEqual(submission.remaining_actions(*args), [])


class TrainFlowTests(unittest.TestCase):
    def run_local(self, rnd):
        with patch.object(submission.tempfile, "TemporaryDirectory",
                          side_effect=lambda prefix, dir: TemporaryDirectory(prefix=prefix)):
            submission.train(rnd)

    def test_adam_classification_uses_unchanged_mounted_parent_and_legacy_memory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            model = load(root / NATIVE, dataset="A")
            expected = submission.classification_facts(model, model, rows(PUBLIC / "validation.jsonl"))["parent"]
            rnd = LocalRound(root)
            self.run_local(rnd)
        facts = rnd.facts["classification"]
        self.assertEqual(facts["parent"], expected)
        self.assertNotEqual(facts["parent"]["confusion_matrix"], facts["child"]["confusion_matrix"])
        self.assertEqual(rnd.facts["paired_parent"]["macro_f1"], .8)
        self.assertEqual(rnd.facts["child"]["macro_f1"], .81)
        self.assertEqual(rnd.finished["configuration"]["classification"], facts)
        self.assertEqual(len(rnd.submissions), 2)
        self.assertEqual(len(rnd.saved["measured_artifacts"]), 2)

    def test_duplicate_parent_train_still_calls_model_and_saves_state_without_upload(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            rnd = LocalRound(root, action="ratio_ensemble")
            # Identity action isolates the duplicate upload/measurement boundary.
            with patch.object(submission, "train_action", side_effect=lambda parent, *args: (parent, {})):
                self.run_local(rnd)
        self.assertEqual(rnd.submissions, [])
        self.assertEqual(rnd.call_ids, ["local-test-call"])
        self.assertFalse(rnd.facts["eligible_child"])
        self.assertEqual(rnd.finished["selected_artifact_id"], "mounted-parent")
        self.assertEqual(rnd.finished["configuration"]["measurement_ids"], [])
        self.assertEqual(rnd.saved["attempts"][-1]["duplicate"]["kind"], "duplicate_of_parent")

    def test_duplicate_historical_train_keeps_parent_without_upload(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(CHECKPOINT, root / NATIVE)
            model = load(root / NATIVE, dataset="A")
            candidate = root / "expected"
            candidate.mkdir()
            # A native update creates genuinely different checkpoint bytes.
            x, y = source_rows(rows(PUBLIC / "train.jsonl"), model.native)
            child, _ = train_action(model, "adam_small", x, y)
            save(child, candidate)
            known = [{"artifact_id": "historical-candidate", "round": 1,
                      "files_sha256": submission.artifact_files_sha256(candidate),
                      "evaluation_context": {"stage": "test-stage",
                                             "validation_sha256": sha256_file(PUBLIC / "validation.jsonl")}}]
            rnd = LocalRound(root, measured=known)
            self.run_local(rnd)
        self.assertEqual(rnd.submissions, [])
        self.assertEqual(rnd.facts["duplicate"]["kind"], "duplicate_of_artifact")
        self.assertEqual(rnd.finished["selected_artifact_id"], "mounted-parent")
        self.assertEqual(rnd.saved["attempts"][-1]["duplicate"]["artifact_id"], "historical-candidate")


class LegacyExtensionTests(unittest.TestCase):
    def test_decay_pair_and_toluene_without_references_still_roundtrip(self):
        model = load(CHECKPOINT, dataset="A")
        x, y = source_rows(rows(PUBLIC / "train.jsonl"), model.native)
        validation, _ = source_rows(rows(PUBLIC / "validation.jsonl"), model.native)
        for action in ("decay_pair", "toluene_check"):
            with self.subTest(action=action), TemporaryDirectory() as temporary:
                model, _ = train_action(model, action, x, y)
                self.assertNotIn("references", model.extension)
                folder = Path(temporary)
                save(model, folder, frozen_native=CHECKPOINT)
                restored = load(folder / NATIVE, folder / EXTENSION, dataset="A")
                np.testing.assert_allclose(model.predict(validation), restored.predict(validation), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
