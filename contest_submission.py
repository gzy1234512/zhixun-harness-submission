"""Competition SDK entry: one bounded experiment and one gateway decision per round."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from functools import lru_cache
from time import monotonic, process_time

import numpy as np

from contest_engine.model import (
    ACTIONS, EXTENSION, NATIVE, TASKS as DATASETS, available_actions, input_vector, load, save, sha256_file,
    source_rows, train_action,
)


TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
METRICS = {"A": "macro_f1", "B": "rmse", "C": "rmse"}
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def parent_files(rnd):
    paths = rnd.parent.descriptor["checkpoint_paths"]
    if not paths or Path(paths[0]).suffix != ".npz" or len(paths) > 2:
        raise ValueError("unsupported mounted checkpoint descriptor")
    if len(paths) == 2 and paths[1] != EXTENSION:
        raise ValueError("unsupported auxiliary checkpoint")
    return [rnd.parent.path(path) for path in paths]


def state_for(directory, dataset, round_index):
    memory = directory / "memory.json"
    if memory.exists():
        previous = json.loads(memory.read_text(encoding="utf-8"))
        if previous.get("dataset") != dataset or previous.get("round", round_index) >= round_index:
            raise ValueError("mounted state does not precede this dataset and round")
        return previous, True
    return {"attempts": [], "next_action": None}, False


def remaining_actions(actions, state, parent_key, train_hash, evaluation_context=None):
    seen = {e["action"] for e in state["attempts"]
            if e["parent"] == parent_key and e["train"] == train_hash
            and (evaluation_context is None or e.get("evaluation_context") == evaluation_context)}
    return [action for action in actions if action not in seen]


def classification_facts(parent, child, rows):
    """Diagnostic replay only; candidate eligibility still uses host measurements."""
    from scipy.stats import binomtest

    x, truth = source_rows(rows, parent.native)
    labels = parent.native.meta["classes"]
    old, new = parent.predict(x).argmax(axis=1), child.predict(x).argmax(axis=1)

    def summary(predicted):
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        np.add.at(matrix, (truth, predicted), 1)
        support, predicted_n = matrix.sum(axis=1), matrix.sum(axis=0)
        per_class = {}
        for index, label in enumerate(labels):
            tp = int(matrix[index, index])
            denominator = int(support[index] + predicted_n[index])
            per_class[label] = {"f1": 2 * tp / denominator if denominator else 0.,
                                "recall": tp / int(support[index]) if support[index] else 0.,
                                "support": int(support[index])}
        return {"per_class": per_class, "confusion_matrix": matrix.tolist(),
                "macro_f1": float(np.mean([item["f1"] for item in per_class.values()])),
                "accuracy": float(np.mean(predicted == truth))}

    before, after = summary(old), summary(new)
    pair = [labels.index("Ethanol"), labels.index("Acetaldehyde")]
    subset = np.isin(truth, pair)
    was_correct, now_correct = old == truth, new == truth
    improved = int(np.sum(subset & ~was_correct & now_correct))
    worsened = int(np.sum(subset & was_correct & ~now_correct))
    discordant = improved + worsened
    return {"source": "local public validation replay; platform macro_f1 remains authoritative",
            "labels": list(labels), "rows": len(truth), "parent": before, "child": after,
            "confusion_delta": (np.asarray(after["confusion_matrix"]) - before["confusion_matrix"]).tolist(),
            "changed_predictions": int(np.sum(old != new)),
            "mcnemar": {"classes": [labels[index] for index in pair],
                         "scope": "true labels in ethanol/acetaldehyde pair", "support": int(subset.sum()),
                         "improved": improved, "worsened": worsened,
                         "p_value": float(binomtest(improved, discordant, .5).pvalue) if discordant else 1.,
                         "method": "two-sided exact binomial"},
            "toluene_support": int(np.sum(truth == labels.index("Toluene")))}


def artifact_files_sha256(folder):
    return [sha256_file(folder / NATIVE),
            sha256_file(folder / EXTENSION) if (folder / EXTENSION).is_file() else None]


def duplicate_candidate(files, parent_key, parent_id, measured_artifacts, evaluation_context):
    if files == parent_key:
        return {"kind": "duplicate_of_parent", "artifact_id": parent_id}
    if evaluation_context is not None:
        for artifact in reversed(measured_artifacts):
            if artifact.get("evaluation_context") == evaluation_context and artifact.get("files_sha256") == files:
                return {"kind": "duplicate_of_artifact", "artifact_id": artifact["artifact_id"],
                        "round": artifact.get("round")}
    return None


def plan_context(rnd, dataset, train_hash):
    """Bind A/C plans to their current projection; B keeps its previous behavior."""
    if dataset not in ("A", "C"):
        return None
    validation = Path(rnd.data.validation_path)
    return {"stage": rnd.stage_id, "train_sha256": train_hash,
            "validation_sha256": sha256_file(validation) if validation.is_file() else None}


def migrate_plan_state(state, policy_context, state_existed, round_index):
    """Invalidate an unverifiable plan and archive old A dedup history."""
    if policy_context is None:
        return None
    if not state_existed:
        state["policy_context"] = policy_context
        return None
    previous = state.get("policy_context")
    if previous == policy_context:
        return None
    reason = "missing_context" if previous is None else "context_changed"
    detail = {"reason": reason, "from": previous, "to": policy_context, "round": round_index,
              "invalidated_next_action": state.get("next_action")}
    if previous is None:
        old_attempts = list(state.get("attempts", []) or [])
        old_measured = list(state.get("measured_artifacts", []) or [])
        detail["archived_attempts"] = len(old_attempts)
        detail["archived_measured_artifacts"] = len(old_measured)
        state["legacy"] = {"attempts": old_attempts, "measured_artifacts": old_measured}
        state["attempts"] = []
        state["measured_artifacts"] = []
    state["policy_context"] = policy_context
    state["next_action"] = None
    state["next_action_invalidated"] = reason
    state["plan_invalidation"] = detail
    return detail


def resolve_action(choices, state, policy_context=None):
    """A: bound plan, no silent first-item fallback. B/C: legacy select_action."""
    if policy_context is None:
        action = select_action(choices, state)
        return action, {"reason": "legacy_bc" if action is not None else "no_action",
                        "requested": state.get("next_action")}
    invalidated = state.pop("next_action_invalidated", None)
    if invalidated is not None:
        invalidation = state.get("plan_invalidation") or {}
        requested = invalidation.get("invalidated_next_action")
        if choices:
            chosen = next(iter(choices), None)
            if chosen is not None:
                reason = "context_reset" if invalidated == "context_changed" else "missing_context_reset"
                return chosen, {"reason": reason, "requested": requested, "invalidated": invalidated,
                                "reset": True, "reset_policy": "first_available"}
        return "hold", {"reason": f"{invalidated}_hold", "requested": requested,
                        "invalidated": invalidated, "reset": False}
    preferred = state.get("next_action")
    if preferred is None:
        chosen = next(iter(choices), None)
        return chosen, {"reason": "initial_first_available" if chosen is not None else "no_action",
                        "requested": None}
    if preferred == "hold":
        return "hold", {"reason": "planned_hold", "requested": "hold"}
    if preferred in choices:
        return preferred, {"reason": "planned", "requested": preferred}
    return "hold", {"reason": "plan_not_in_choices_hold", "requested": preferred,
                    "choices": list(choices)}


def validate_next_action(proposal, facts, accept):
    """Validate the model's next action against the program's final parent/child branch."""
    branch = "child" if accept else "parent"
    allowed = list(facts.get("next_if_child" if accept else "next_if_parent") or [])
    if proposal is None:
        return "hold", {"branch": branch, "original": None, "actual": "hold",
                        "reason": "invalid_proposal", "allowed": allowed}
    requested = proposal.get("next_action")
    if requested == "hold":
        actual, reason = "hold", "planned_hold"
    elif requested in allowed:
        actual, reason = requested, "accepted"
    else:
        actual, reason = "hold", f"not_in_{branch}_next_actions"
    return actual, {"branch": branch, "original": requested, "actual": actual,
                    "reason": reason, "allowed": allowed}


def select_action(choices, state):
    """Legacy B/C/audit helper; A uses resolve_action."""
    preferred = state.get("next_action")
    if preferred == "hold":
        return "hold"
    return preferred if preferred in choices else next(iter(choices), None)


def history_facts(records):
    result = []
    for record in records[-4:]:
        config = record.get("configuration")
        if not isinstance(config, dict):
            config = {}
        row = {"id": record["id"]}
        for key in ("round", "action", "selection", "positive_is_better_gain"):
            value = config.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                row[key] = value
            elif isinstance(value, str) and len(value) <= 64:
                row[key] = value
        result.append(row)
    return result


def official_metric(receipt, metric):
    summary = receipt.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    if not isinstance(metrics, dict):
        return {}
    result = {}
    for name in (metric, "accuracy", "mae", "nasa_score"):
        value = metrics.get(name)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            if name not in ("macro_f1", "accuracy") or value <= 1:
                result[name] = value
    return result


def submit(rnd, folder, hypothesis, change, config, seed):
    checkpoints = [NATIVE, EXTENSION] if (folder / EXTENSION).is_file() else [NATIVE]
    if sum((folder / path).stat().st_size for path in checkpoints) > MAX_ARTIFACT_BYTES:
        raise ValueError("checkpoint artifact exceeds the 64 MiB contest limit")
    return rnd.submit_candidate(folder, hypothesis=hypothesis, change_summary=change,
                                configuration=config, seed=seed, checkpoint_paths=checkpoints,
                                code_paths=[], predict_argv=PREDICT_ARGV)


def decision(rnd, evidence, actions):
    messages = [
        {"role": "system", "content": "Choose the industrial action using only the given measured facts. "
         "The engine computes improvement direction and blocks an ineligible child. "
         "Output only a JSON object with selection, next_action, and reason. "
         "Choose child only when eligible_child is true and the measured gain justifies it. "
         "For next_action use an option from next_if_child or next_if_parent matching your selection. "
         "If that list is empty, use hold. Give a brief reason of at most 100 characters."},
        {"role": "user", "content": json_text(evidence)},
    ]
    messages[0]["content"] += (" A actions: decay_pair and ratio_ensemble correct ethanol/acetaldehyde; "
                                "toluene_check adds a toluene/acetone verifier; decay_triad is a three-class variant; "
                                "adam_small and adam_tiny continue the native model at reduced learning rates.")
    if evidence["dataset"] == "A":
        messages[0]["content"] += (" classification is diagnostic public validation replay, not an official score. "
                                    "You may reject an eligible child because of per-class deterioration or weak support. "
                                    "In reason cite a per-class F1, recall, support, McNemar fact, or the failure. "
                                    "A duplicate was not submitted or remeasured; retain its mounted parent.")
    receipt = rnd.model(messages, max_tokens=min(rnd.max_output_tokens, 224), temperature=0, enable_thinking=False)
    usage = receipt.response.get("usage")
    tokens = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
              if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
    try:
        content = receipt.text.strip()
        if content.startswith("```json\n") and content.endswith("\n```"):
            content = content[8:-4].strip()
        parsed = json.loads(content)
        if (not isinstance(parsed, dict) or set(parsed) != {"selection", "next_action", "reason"}
                or parsed["selection"] not in ("parent", "child") or parsed["next_action"] not in (*actions, "hold")
                or not isinstance(parsed["reason"], str) or not 1 <= len(parsed["reason"].strip()) <= 100):
            raise ValueError("invalid decision fields")
    except (ValueError, TypeError):
        return None, tokens
    return parsed, tokens


def train(rnd):
    start, cpu_start = monotonic(), process_time()
    dataset = rnd.dataset_key
    if rnd.task.get("task") != DATASETS[dataset]:
        raise ValueError("frozen task contract conflicts with dataset")
    metric = METRICS[dataset]
    paths = parent_files(rnd)
    parent = load(*paths, dataset=dataset)
    seed = parent.native.meta["seed"]
    actions = ACTIONS[dataset]
    previous = rnd.history()
    train_hash = sha256_file(rnd.data.train_path)
    native_hash = sha256_file(paths[0])
    parent_key = [native_hash, sha256_file(paths[1]) if len(paths) == 2 else None]
    evaluation_context = None
    if dataset == "A" and Path(rnd.data.validation_path).is_file():
        evaluation_context = {"stage": rnd.stage_id, "validation_sha256": sha256_file(rnd.data.validation_path)}

    with tempfile.TemporaryDirectory(prefix="zhixun-", dir="/workspace") as location:
        work = Path(location)
        state_dir = rnd.state_workspace(work / "state")
        state, state_existed = state_for(state_dir, dataset, rnd.round_index)
        policy_context = plan_context(rnd, dataset, train_hash)
        state_migration = migrate_plan_state(state, policy_context, state_existed, rnd.round_index)
        if dataset == "C":
            from contest_engine.c_rul import remaining_actions as remaining_c_actions
            measurement_context = [rnd.stage_id, sha256_file(rnd.data.validation_path)]
            choices = remaining_c_actions(actions, state, native_hash, train_hash, measurement_context)
        else:
            choices = remaining_actions(available_actions(parent), state, parent_key, train_hash, evaluation_context)
        action, action_resolution = resolve_action(choices, state, policy_context)
        phase = {"dataset": dataset, "round": rnd.round_index, "stage": rnd.stage_id,
                 "parent_artifact_id": rnd.parent.artifact_id, "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                 "train_sha256": train_hash, "parent_files_sha256": parent_key, "action": action}
        if dataset in ("A", "C"):
            phase.update(policy_context=policy_context, state_migration=state_migration,
                         action_requested=action_resolution.get("requested"),
                         action_resolution=action_resolution)
        if dataset == "C":
            phase["c_measurement_context"] = measurement_context
        if type(rnd.wall_seconds) in (int, float) and rnd.wall_seconds <= 8:
            action = None
            phase["action"] = None
        anchor_metrics, child_metrics, recipe = {}, {}, {}
        candidate = child = None
        classification = duplicate = candidate_key = None
        measured_artifacts = state.get("measured_artifacts", []) if dataset == "A" else []
        if dataset == "A":
            phase["evaluation_context"] = evaluation_context
        error = "no_untried_action" if action is None else ("hold_by_policy" if action == "hold" else None)
        uploads = []
        hypothesis = "Bounded source-only update may improve the current stage against the inherited model."
        change = "No new model was selected."
        if action not in (None, "hold"):
            anchor = work / "anchor"
            anchor.mkdir()
            for path, name in zip(paths, (NATIVE, EXTENSION)):
                shutil.copyfile(path, anchor / name)
            try:
                x, y = source_rows(rnd.data.training_rows(), parent.native)
                if dataset == "C":
                    from contest_engine.c_rul import train_c_action
                    child, recipe = train_c_action(parent, x, y, action, state_dir,
                                                   native_hash + ":" + train_hash + ":native-small-v1")
                else:
                    child, recipe = train_action(parent, action, x, y)
                candidate_dir = work / "candidate"
                candidate_dir.mkdir()
                save(child, candidate_dir, frozen_native=None if recipe.get("steps") else paths[0])
                check = load(candidate_dir / NATIVE, candidate_dir / EXTENSION if child.extension is not None else None,
                             dataset=dataset)
                np.testing.assert_allclose(check.predict(x[:16]), child.predict(x[:16]), rtol=1e-12, atol=1e-12)
                phase["train_rows"] = len(y)
                phase["changed_native_sha256"] = sha256_file(candidate_dir / NATIVE)
                if dataset == "A":
                    candidate_key = artifact_files_sha256(candidate_dir)
                    duplicate = duplicate_candidate(candidate_key, parent_key, rnd.parent.artifact_id,
                                                    measured_artifacts, evaluation_context)
                change = f"Tested {action} from the mounted parent on this round's authorized training projection."
            except (ValueError, ArithmeticError, AssertionError) as exc:
                error = f"fit_{type(exc).__name__}"
                print(f"{dataset} round {rnd.round_index} {action}: {type(exc).__name__}", file=sys.stderr)
            if error is None and duplicate is not None:
                error = duplicate["kind"]
                change = f"Skipped upload and measurement of {action}: {error}."
            if error is None:
                reference = submit(rnd, anchor, "Measure the identical mounted parent in this phase.",
                                   "Byte-for-byte copy of the mounted checkpoint and its extension.",
                                   {**phase, "role": "paired_parent"}, seed)
                uploads.append(reference["artifact_id"])
                anchor_metrics = official_metric(reference, metric)
                candidate = submit(rnd, candidate_dir, hypothesis, change, {**phase, "recipe": recipe}, seed)
                uploads.append(candidate["artifact_id"])
                child_metrics = official_metric(candidate, metric)
                if dataset == "A" and evaluation_context is not None:
                    for artifact_id, files in ((reference["artifact_id"], parent_key),
                                               (candidate["artifact_id"], candidate_key)):
                        measured_artifacts.append({"artifact_id": artifact_id, "files_sha256": files,
                                                   "evaluation_context": evaluation_context, "round": rnd.round_index})
        if dataset == "A":
            if child is not None and candidate_key is not None:
                try:
                    # Adam mutates parent.native in place; reload the unchanged mounted parent.
                    classification = classification_facts(load(*paths, dataset=dataset), child,
                                                          rnd.data.validation_rows())
                except (OSError, ValueError, ArithmeticError, TypeError, KeyError) as exc:
                    classification = {"source": "local public validation replay; platform macro_f1 remains authoritative",
                                      "status": "unavailable", "error": type(exc).__name__}
            else:
                classification = {"source": "local public validation replay; platform macro_f1 remains authoritative",
                                  "status": "not_computed", "reason": error}
        gain = None
        if metric in anchor_metrics and metric in child_metrics:
            old, new = anchor_metrics[metric], child_metrics[metric]
            gain = new - old if dataset == "A" else old - new
        eligible = error is None and gain is not None and gain > max(1e-10, anchor_metrics[metric] * 1e-8)
        if dataset == "C":
            from contest_engine.c_rul import acceptance_guard
            phase["c_risk_guard"] = acceptance_guard(anchor_metrics, child_metrics)
            eligible = eligible and phase["c_risk_guard"]["passed"]
        facts = {"dataset": dataset, "round": rnd.round_index, "metric": metric,
                 "direction": "maximize" if dataset == "A" else "minimize", "action": action,
                 "paired_parent": anchor_metrics, "child": child_metrics, "positive_is_better_gain": gain,
                 "eligible_child": eligible, "failure": error, "candidate_actions_for_next_round": list(actions),
                 "next_if_parent": [item for item in choices if item != action],
                 "next_if_child": [item for item in available_actions(child) if item != action] if eligible and child is not None else [],
                 "history_checkpoint_count": len(previous), "recent_host_history": history_facts(previous),
                 "recent_attempts": state["attempts"][-4:]}
        if dataset == "A":
            facts.update(classification=classification, duplicate=duplicate,
                         policy_context=policy_context, state_migration=state_migration,
                         action_resolution=action_resolution)
        if dataset == "C":
            facts["c_risk_guard"] = phase["c_risk_guard"]
            facts.update(policy_context=policy_context, state_migration=state_migration,
                         action_resolution=action_resolution)
            facts["next_if_child"] = []
            if eligible:
                future_state = {**state, "attempts": list(state["attempts"])}
                future_state["attempts"].append({"parent": [phase["changed_native_sha256"], None],
                    "train": train_hash, "action": action, "context": measurement_context, "completed": True})
                if action == "c_native_small":
                    future_state["c_native_accepted_train"] = train_hash
                    future_state["attempts"].append({"parent": [phase["changed_native_sha256"], None],
                        "train": train_hash, "action": "c_bounded", "context": measurement_context, "completed": True})
                facts["next_if_child"] = remaining_c_actions(actions, future_state,
                    phase["changed_native_sha256"], train_hash, measurement_context)
        proposal, tokens = decision(rnd, facts, actions)
        accept = eligible and proposal is not None and proposal["selection"] == "child"
        selected = candidate["artifact_id"] if accept else rnd.parent.artifact_id
        decision_status = "held" if action == "hold" and not accept else ("accepted" if accept else "kept_parent")
        if dataset in ("A", "C"):
            if dataset == "C" and proposal is None and eligible:
                # The measured C fit remains cached: an invalid model reply can retry the same fit safely.
                actual_next = action
                next_action_validation = {"branch": "parent", "original": None, "actual": action,
                    "reason": "invalid_proposal_retry_cached_candidate", "allowed": [action, *facts["next_if_parent"]]}
            else:
                actual_next, next_action_validation = validate_next_action(proposal, facts, accept)
            state["policy_context"] = policy_context
            state["next_action_validation"] = next_action_validation
        else:
            actual_next, next_action_validation = (proposal or {}).get("next_action"), None
        state["next_action"] = actual_next
        config = {**phase, "recipe": recipe, "model_name": rnd.model_name,
                  "eligible_child": eligible, "measurement_ids": uploads, "paired_parent": anchor_metrics,
                  "measured_child": child_metrics, "positive_is_better_gain": gain, "error": error,
                  "selection": decision_status, "proposal": proposal, "gateway_token_usage": tokens,
                  "selected_artifact_id": selected}
        if dataset == "A":
            config.update(classification=classification, duplicate=duplicate,
                          candidate_files_sha256=candidate_key,
                          policy_context=policy_context, state_migration=state_migration,
                          action_resolution=action_resolution,
                          next_action_validation=next_action_validation)
        if dataset == "C":
            config["next_action_validation"] = next_action_validation
        if action is not None:
            attempt = {"parent": parent_key, "train": train_hash, "action": action,
                       "gain": gain, "decision": decision_status, "failure": error}
            if dataset == "A":
                attempt.update(duplicate=duplicate, evaluation_context=evaluation_context,
                               candidate_files_sha256=candidate_key,
                               action_resolution=action_resolution)
            if dataset == "C":
                attempt.update(context=measurement_context,
                               completed=error is None and gain is not None
                               and (not eligible or proposal is not None))
                if accept and action == "c_native_small":
                    state["c_native_accepted_train"] = train_hash
                    state["attempts"].append({**attempt, "parent": [phase["changed_native_sha256"], None],
                                               "action": "c_bounded", "completed": True})
            state["attempts"].append(attempt)
        config["cpu_seconds"] = process_time() - cpu_start
        config["participant_wall_seconds"] = monotonic() - start
        record = {"dataset": dataset, "round": rnd.round_index, "next_action": actual_next,
                  "attempts": state["attempts"][-20:], "last": config}
        if dataset == "A":
            record.update(policy_context=policy_context, state_migration=state_migration,
                          action_resolution=action_resolution,
                          next_action_validation=next_action_validation,
                          legacy=state.get("legacy"),
                          plan_invalidation=state.get("plan_invalidation"))
            record["measured_artifacts"] = measured_artifacts[-40:]
        if dataset == "C":
            record["c_native_accepted_train"] = state.get("c_native_accepted_train")
            record.update(policy_context=policy_context, state_migration=state_migration,
                          action_resolution=action_resolution, next_action_validation=next_action_validation,
                          legacy=state.get("legacy"), plan_invalidation=state.get("plan_invalidation"))
        (state_dir / "memory.json").write_text(json_text(record), encoding="utf-8")
        rnd.save_state(state_dir)
        reason = (f"{decision_status}; model call {rnd.call_ids[-1]}; valid decision={proposal is not None}; "
                  f"paired {metric} gain={gain}; positive means improvement; eligible={eligible}; failure={error}.")
        if dataset in ("A", "C") and next_action_validation is not None:
            reason += (f" next_action={next_action_validation['actual']} "
                       f"({next_action_validation['reason']}).")
        rnd.finish(selected_artifact_id=selected, hypothesis=hypothesis, change_summary=change,
                   reason=reason, configuration=config, seed=seed,
                   history_ids=[item["id"] for item in previous], model_call_ids=list(rnd.call_ids))


@lru_cache(maxsize=1)
def predictor(path, extension, dataset):
    return load(path, extension, dataset=dataset)


@lru_cache(maxsize=4)
def predict_target(model_dir, artifact_env):
    """Resolve and validate the mounted artifact once per process.

    The mounted checkpoint directory is read-only and fixed for the life of one
    prediction stream, so the descriptor validation only has to run once. On a
    slow bind mount those probes dominated the row budget: nine metadata
    syscalls at about 0.36 ms each against roughly 0.85 ms of real row work.
    Validation semantics are unchanged; the cache key is the mount and the
    CONTEST_ARTIFACT_PATH value.
    """
    root = Path(model_dir).resolve()
    path = Path(artifact_env) if artifact_env is not None else root / NATIVE
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("selected checkpoint must be within /model")
    extension = root / EXTENSION
    if extension.is_symlink():
        raise ValueError("extension must be a regular mounted file")
    return path, path.relative_to(root).as_posix(), extension, extension.is_file()


def predict(sample, model_dir):
    path, mounted_as, extension, has_extension = predict_target(
        str(model_dir), os.environ.get("CONTEST_ARTIFACT_PATH"))
    if sample.context is not None:
        declared = sample.context["artifact"].get("checkpoint_paths", [])
        if mounted_as not in declared or (EXTENSION in declared) != has_extension:
            raise ValueError("checkpoint files differ from the selected artifact descriptor")
    model = predictor(path, extension if has_extension else None, sample.dataset_key)
    if sample.task != DATASETS[model.dataset]:
        raise ValueError("prediction type differs from the checkpoint")
    x = np.asarray([input_vector(sample.inputs, model.native)], dtype=np.float64)
    result = model.predict(x)[0]
    if sample.task == "regression":
        return {"value": float(result)}
    scores = result.tolist()
    best = int(np.argmax(scores))
    if scores.count(scores[best]) > 1:
        scores[best] = math.nextafter(scores[best], 1.)
    return [{"label": label, "score": score} for label, score in zip(model.native.meta["classes"], scores)]
