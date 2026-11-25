"""Local input-budget and receipt-reuse checks. No network or paid calls."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.contest_live_gateway import LiveGateway, MODEL, PREFILL_LIMIT, bounded_gateway_context


ROOT = Path(__file__).resolve().parents[1]
RUN = "contest-104-a-p1-reset-live-1126"


def large_context():
    receipt = json.loads((ROOT / "runs" / RUN / "requests" / (RUN + "-A-04.json")).read_text())
    state = json.loads((ROOT / "runs" / RUN / "A/04/state/memory.json").read_text())
    facts = receipt["request"]["facts"]
    facts["recent_attempts"] = state["attempts"][-4:]
    return receipt["request"]["instruction"], facts


class GatewayInputTests(unittest.TestCase):
    def test_small_input_is_unchanged(self):
        facts = {"paired_parent": {"macro_f1": .8}, "child": {"macro_f1": .81}}
        actual, projection = bounded_gateway_context("instruction", facts)
        self.assertIs(actual, facts)
        self.assertIsNone(projection)

    def test_real_history_compaction_keeps_classification_and_decisions(self):
        instruction, facts = large_context()
        original = json.loads(json.dumps(facts))
        actual, projection = bounded_gateway_context(instruction, facts)
        self.assertEqual(facts, original)
        self.assertGreater(projection["original_byte_upper_bound"], PREFILL_LIMIT)
        self.assertLessEqual(projection["provider_byte_upper_bound"], PREFILL_LIMIT)
        for key in facts:
            if key != "recent_attempts":
                self.assertEqual(actual[key], facts[key])
        for before, after in zip(facts["recent_attempts"], actual["recent_attempts"]):
            for key in ("action", "gain", "decision", "failure", "duplicate", "action_resolution"):
                if key in before:
                    self.assertEqual(after[key], before[key])

    def test_uncompactable_input_still_fails(self):
        with self.assertRaisesRegex(ValueError, "excess gateway input volume"):
            bounded_gateway_context("", {"classification": {"huge": "x" * PREFILL_LIMIT}})

    def test_completed_receipt_is_reused_before_any_budget_or_transport(self):
        with tempfile.TemporaryDirectory() as folder:
            gateway = LiveGateway.__new__(LiveGateway)
            gateway.directory, gateway.run_id = Path(folder), "reuse-check"
            gateway.budget, gateway.transport = Mock(), Mock()
            frame = {"run_id": gateway.run_id, "dataset_key": "A", "round_index": 1,
                     "stage_id": "s", "prompt_version": "v", "task_contract": {}, "prompt": "host prompt"}
            facts = {"classification": "x" * PREFILL_LIMIT}
            msg = {"max_tokens": 224, "messages": [{"role": "system", "content": "choose"},
                    {"role": "user", "content": json.dumps(facts)}]}
            meta = {key: frame[key] for key in ("run_id", "dataset_key", "round_index", "stage_id", "prompt_version", "task_contract")}
            request = {"model": MODEL, "instruction": frame["prompt"] + "\n" + json.dumps(meta, sort_keys=True) + "\n\nchoose",
                       "facts": facts, "max_tokens": 224}
            signature = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
            saved = {"request_sha256": signature, "status": "completed", "call_id": "already-paid"}
            (gateway.directory / "reuse-check-A-01.json").write_text(json.dumps(saved))
            self.assertEqual(gateway.model(frame, msg), saved)
            gateway.budget.reserve.assert_not_called()
            gateway.transport.assert_not_called()

    def test_projected_request_keeps_original_cache_identity_and_exact_provider_facts(self):
        instruction, facts = large_context()
        with tempfile.TemporaryDirectory() as folder:
            gateway = LiveGateway.__new__(LiveGateway)
            gateway.directory, gateway.run_id, gateway.mode = Path(folder), "projection-check", "openai"
            ledger = Path(folder) / "ledger.json"
            ledger.write_text("{}")
            identity = "projection-check-A-05"
            gateway.budget = SimpleNamespace(path=ledger, reserve=Mock(return_value=.1),
                finish=Mock(return_value={"entries": [{"id": identity, "charged_upper_cny": .01}]}))
            gateway.transport = Mock(return_value={"provider": {"request_id": "local-fake-provider", "model": MODEL},
                "content": '{"selection":"parent","next_action":"hold","reason":"local test"}',
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
            gateway.transport.redact = lambda value: value
            frame = {"run_id": gateway.run_id, "dataset_key": "A", "round_index": 5,
                     "stage_id": "s", "prompt_version": "v", "task_contract": {}, "prompt": "host prefix"}
            msg = {"max_tokens": 224, "messages": [{"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(facts)}]}
            result = gateway.model(frame, msg)
            sent = gateway.transport.call_args.args[0]
            self.assertEqual(result["request"]["facts"], facts)
            self.assertEqual(result["provider_request"]["facts"], sent)
            self.assertEqual(sent["classification"], facts["classification"])
            signature = hashlib.sha256(json.dumps(result["request"], sort_keys=True, allow_nan=False).encode()).hexdigest()
            self.assertEqual(result["request_sha256"], signature)
            self.assertEqual(gateway.budget.reserve.call_args.args[1]["prefill"], PREFILL_LIMIT)


if __name__ == "__main__":
    unittest.main()
