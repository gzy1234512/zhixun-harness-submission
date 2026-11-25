"""Portable extensions of the mounted native NumPy/Adam checkpoint."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import numpy as np

try:
    from .reference.data import flatten
    from .reference.model import Model
except ModuleNotFoundError as exc:
    if exc.name != "contest_engine.reference":
        raise
    from vendor.reference.A.baselines.data import flatten
    from vendor.reference.A.baselines.model import Model


TASKS = {"A": "classification", "B": "regression", "C": "regression"}
WIDTHS = {"A": 128, "B": 34, "C": 24}
NATIVE = "checkpoint.npz"
EXTENSION = "extension.npz"
CLASSES = ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene")
ETHANOL = CLASSES.index("Ethanol")
ACETALDEHYDE = CLASSES.index("Acetaldehyde")
ACETONE = CLASSES.index("Acetone")
TOLUENE = CLASSES.index("Toluene")
ACTIONS = {
    "A": ("decay_pair", "ratio_ensemble", "adam_small", "adam_tiny", "toluene_check", "decay_triad"),
    "B": ("residual", "residual_light", "adam_small", "adam_tiny"),
    "C": ("c_bounded", "c_native_small", "c_bounded_narrow", "c_bounded_shrink"),
}


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def input_vector(inputs, native):
    paths, values = flatten(inputs)
    if paths != native.meta["paths"]:
        raise ValueError("input schema/order differs from the mounted checkpoint")
    return values


def source_rows(rows, native):
    x, y = [], []
    classes = native.meta["classes"]
    for row in rows:
        x.append(input_vector(row["inputs"], native))
        target = row["answer"]
        if classes:
            target = classes.index(target)
        elif type(target) not in (int, float) or not np.isfinite(target) or target < 0:
            raise ValueError("regression train label must be finite and nonnegative")
        y.append(target)
    if not x:
        raise ValueError("phase training projection is empty")
    return np.asarray(x, dtype=np.float64), np.asarray(y)


def decay_view(x, floor):
    decay = x.reshape(-1, 16, 8)[:, :, 5:8]
    logs = np.log(np.maximum(np.abs(decay), floor))
    return (logs - logs.mean(axis=2, keepdims=True)).reshape(-1, 48)


def decay_margin(x, side, prefix=""):
    view = (decay_view(x, side[prefix + "floor"]) - side[prefix + "mean"]) / side[prefix + "scale"]
    support = side[prefix + "support"]
    distance = (view * view).sum(1)[:, None] + (support * support).sum(1) - 2 * view @ support.T
    kernel = np.exp(-float(side[prefix + "gamma"]) * np.maximum(distance, 0.))
    return kernel @ side[prefix + "dual"] + float(side[prefix + "bias"])


def ratio_floor(x):
    """Tiny per-feature epsilon added to the ratiometric denominator."""
    return np.full(x.shape[1] // 16, 1e-8)


def ratio_view(x, reference, floor):
    """Divide every sensor by one strong reference sensor, then keep the decay block."""
    sensors = x.reshape(-1, 16, 8)
    denominator = np.abs(sensors[:, reference:reference + 1, :]) + floor
    return (sensors / denominator)[:, :, 5:8].reshape(-1, 48)


def ratio_margin(x, side, prefix, reference):
    view = (ratio_view(x, int(reference), side["ratio_floor"]) - side[prefix + "mean"]) / side[prefix + "scale"]
    support = side[prefix + "support"]
    distance = (view * view).sum(1)[:, None] + (support * support).sum(1) - 2 * view @ support.T
    kernel = np.exp(-float(side[prefix + "gamma"]) * np.maximum(distance, 0.))
    return kernel @ side[prefix + "dual"] + float(side[prefix + "bias"])


def select_references(x, count=4):
    """Choose reference sensors by mean response on the authorized training projection only."""
    means = x.reshape(-1, 16, 8)[:, :, 0].mean(axis=0)
    order = np.argsort(-means)
    count = max(1, min(count, len(order) - 1))
    references = order[:count].astype(int)
    gap = float(means[order[count - 1]] / max(float(means[order[count]]), 1e-12))
    return references, means, gap


def fit_ratio_ensemble(x, y, negative, positive, references):
    from sklearn.svm import SVC

    keep = np.isin(y, [negative, positive])
    if len(np.unique(y[keep])) != 2:
        raise ValueError(f"ratio fit needs both {CLASSES[negative]} and {CLASSES[positive]} in authorized train")
    x_keep, labels = x[keep], y[keep] == positive
    floor = ratio_floor(x)
    side = {"references": np.asarray(references, dtype=np.int64), "ratio_floor": floor,
            "gate": np.asarray([negative, positive])}
    for index, reference in enumerate(references):
        # Scale on the full authorized training projection; fit the classifier only on the pair labels.
        full = ratio_view(x, int(reference), floor)
        mean, scale = full.mean(axis=0), np.maximum(full.std(axis=0), 1e-6)
        view = ratio_view(x_keep, int(reference), floor)
        gamma = 1.0 / view.shape[1]
        fitted = SVC(C=2.0, gamma=gamma, class_weight="balanced", cache_size=64).fit((view - mean) / scale, labels)
        prefix = f"ref{index}_"
        side[prefix + "mean"] = mean
        side[prefix + "scale"] = scale
        side[prefix + "gamma"] = np.array(gamma)
        side[prefix + "bias"] = np.array(fitted.intercept_[0])
        side[prefix + "support"] = fitted.support_vectors_
        side[prefix + "dual"] = fitted.dual_coef_[0]
    return side


def has_decay_pair(extension):
    return extension is not None and np.array_equal(extension.get("gate"), [ETHANOL, ACETALDEHYDE])


def has_ratio_ensemble(extension):
    return extension is not None and "references" in extension


def available_actions(model):
    actions = ACTIONS[model.dataset]
    if model.dataset == "A":
        if has_ratio_ensemble(model.extension):
            actions = tuple(action for action in actions if action != "decay_pair")
        if not has_decay_pair(model.extension):
            actions = tuple(action for action in actions if action != "toluene_check")
    return actions


def tree_output(x, side):
    children, values = side["children"], side["values"]
    trees = np.arange(len(children))[None, :]
    result = []
    for offset in range(0, len(x), 2048):
        batch = x[offset:offset + 2048]
        positions = np.zeros((len(batch), len(children)), dtype=np.intp)
        for _ in range(int(side["depth"]) + 1):
            feature = side["features"][trees, positions]
            threshold = side["thresholds"][trees, positions]
            branch = (batch[np.arange(len(batch))[:, None], feature] > threshold).astype(np.intp)
            positions = children[trees, positions, branch]
        result.append(float(side["intercept"]) + values[trees, positions].sum(axis=1))
    return np.concatenate(result)


@dataclass
class Predictor:
    dataset: str
    native: Model
    extension: dict | None = None

    def predict(self, x):
        if self.extension is None:
            return self.native.predict(x)
        side = self.extension
        if self.dataset == "C" and str(side.get("kind", "")) == "c.rul.regime.v1":
            from .c_rul import predict
            return predict(self.native, x, side)
        if self.dataset == "A":
            scores = self.native.predict(x)
            labels = scores.argmax(axis=1)
            if has_ratio_ensemble(side):
                margin = np.mean([ratio_margin(x, side, f"ref{index}_", reference)
                                  for index, reference in enumerate(side["references"])], axis=0)
            else:
                margin = decay_margin(x, side)
            eligible = np.isin(labels, side["gate"])
            chosen = np.where(eligible & (margin > 0), ACETALDEHYDE, labels)
            negatives = side["gate"][side["gate"] != ACETALDEHYDE]
            rejected = (labels == ACETALDEHYDE) & (margin <= 0)
            chosen[rejected] = negatives[scores[rejected][:, negatives].argmax(axis=1)]
            if "toluene_support" in side:
                toluene_rows = np.flatnonzero(chosen == TOLUENE)
                if toluene_rows.size:
                    verified = decay_margin(x[toluene_rows], side, prefix="toluene_")
                    chosen[toluene_rows[verified < 0]] = ACETONE
            changed = np.flatnonzero(chosen != labels)
            scores[changed, chosen[changed]], scores[changed, labels[changed]] = (
                scores[changed, labels[changed]], scores[changed, chosen[changed]].copy())
            return scores
        hidden, logits = self.native.forward(self.native.normalized(x))
        residual = tree_output(np.column_stack((x, hidden)), side)
        corrected = logits[:, 0] + float(side["strength"]) * residual
        return np.maximum(0., corrected * self.native.meta["target_scale"] + self.native.meta["target_mean"])


def load(native_path, extension_path=None, dataset=None):
    native = Model.load(native_path)
    inferred = next((key for key, size in WIDTHS.items() if size == len(native.meta["paths"])), None)
    if inferred is None or (dataset is not None and dataset != inferred) or native.meta["task"] != TASKS[inferred]:
        raise ValueError("unexpected industrial checkpoint schema")
    if inferred == "A" and tuple(native.meta["classes"]) != CLASSES:
        raise ValueError("unexpected gas order")
    side = None
    if extension_path is not None:
        with np.load(extension_path, allow_pickle=False) as arrays:
            side = {key: arrays[key] for key in arrays.files}
        if str(side.pop("dataset")) != inferred or str(side.pop("native_sha256")) != sha256_file(native_path):
            raise ValueError("extension does not bind to this mounted checkpoint")
    return Predictor(inferred, native, side)


def save(predictor, folder, frozen_native=None):
    native_path = folder / NATIVE
    if frozen_native is None:
        predictor.native.save(native_path)
    else:
        shutil.copyfile(frozen_native, native_path)
    if predictor.extension is not None:
        np.savez_compressed(folder / EXTENSION, dataset=np.array(predictor.dataset),
                            native_sha256=np.array(sha256_file(native_path)), **predictor.extension)


def fit_decay(x, y, negative, positive):
    from sklearn.svm import SVC

    keep = np.isin(y, [negative, positive])
    if len(np.unique(y[keep])) != 2:
        raise ValueError(f"decay fit needs both {CLASSES[negative]} and {CLASSES[positive]} in authorized train")
    x, y = x[keep], y[keep] == positive
    floor = np.maximum(np.median(np.abs(x), axis=0).reshape(16, 8)[:, 5:8], 1e-8) * 1e-6
    v = decay_view(x, floor)
    mean, scale = v.mean(axis=0), np.maximum(v.std(axis=0), 1e-6)
    gamma = 1.0 / v.shape[1]
    fitted = SVC(C=2.0, gamma=gamma, class_weight="balanced", cache_size=64).fit((v - mean) / scale, y)
    return {"floor": floor, "mean": mean, "scale": scale, "gamma": np.array(gamma),
            "bias": np.array(fitted.intercept_[0]), "support": fitted.support_vectors_,
            "dual": fitted.dual_coef_[0]}


def fit_residual(native, x, y, light):
    from sklearn.ensemble import HistGradientBoostingRegressor

    hidden, logits = native.forward(native.normalized(x))
    target = (y - native.meta["target_mean"]) / native.meta["target_scale"] - logits[:, 0]
    view = np.column_stack((x, hidden))
    depth, rounds, strength = (3, 20, .65) if light else (5, 36, .9)
    limit = min(31, 2 ** depth)
    fitted = HistGradientBoostingRegressor(
        max_depth=depth, max_leaf_nodes=limit, max_iter=rounds, learning_rate=.1,
        min_samples_leaf=20, l2_regularization=4., early_stopping=False, random_state=native.meta["seed"],
    ).fit(view, target)
    size = 2 * limit - 1
    features = np.zeros((rounds, size), dtype=np.intp)
    thresholds, values = np.zeros((rounds, size)), np.zeros((rounds, size))
    children = np.zeros((rounds, size, 2), dtype=np.intp)
    for i, predictors in enumerate(fitted._predictors):
        nodes = predictors[0].nodes
        n = len(nodes)
        leaf = nodes["is_leaf"].astype(bool)
        features[i, :n] = nodes["feature_idx"]
        thresholds[i, :n] = nodes["num_threshold"]
        values[i, :n] = nodes["value"]
        children[i, :n, 0] = np.where(leaf, np.arange(n), nodes["left"])
        children[i, :n, 1] = np.where(leaf, np.arange(n), nodes["right"])
    side = {"features": features, "values": values, "thresholds": thresholds, "children": children,
            "depth": np.array(depth), "strength": np.array(strength), "intercept": np.array(fitted._baseline_prediction.item())}
    check = view[:128]
    np.testing.assert_allclose(tree_output(check, side), fitted.predict(check), atol=1e-12, rtol=1e-12)
    return side


def train_action(parent, action, x, y):
    native = parent.native
    before = native.meta["steps"]
    if parent.dataset == "C" and action.startswith("c_bounded"):
        from .c_rul import train_bounded
        side, recipe = train_bounded(parent, x, y, action)
        return Predictor("C", native, side), recipe
    if action in ("adam_small", "adam_tiny"):
        factor = .25 if action == "adam_small" else .08
        original_rate = native.meta["learning_rate"]
        native.meta["learning_rate"] = original_rate * factor
        try:
            loss = native.fit(x, y, 1)[0]
        finally:
            native.meta["learning_rate"] = original_rate
        return parent, {"epochs": 1, "lr_factor": factor, "steps": native.meta["steps"] - before, "train_loss": loss}
    if action in ("decay_pair", "decay_triad") and parent.dataset == "A":
        if has_ratio_ensemble(parent.extension):
            raise ValueError("decay_pair would replace the mounted ratio ensemble")
        gate = [ETHANOL, ACETALDEHYDE]
        if action == "decay_triad":
            gate.append(TOLUENE)
        side = fit_decay(x, y, ETHANOL, ACETALDEHYDE)
        if action == "decay_pair" and parent.extension is not None:
            side.update({key: value for key, value in parent.extension.items() if key.startswith("toluene_")})
        side["gate"] = np.asarray(gate)
        return Predictor(parent.dataset, native, side), {"epochs": 0, "solver_fits": 1, "gate": gate,
                                                        "support": len(side["support"]),
                                                        "preserved_toluene": "toluene_support" in side}
    if action == "ratio_ensemble" and parent.dataset == "A":
        references, means, gap = select_references(x)
        side = fit_ratio_ensemble(x, y, ETHANOL, ACETALDEHYDE, references)
        if parent.extension is not None:
            side.update({key: value for key, value in parent.extension.items() if key.startswith("toluene_")})
        support = int(sum(len(side[f"ref{index}_support"]) for index in range(len(references))))
        recipe = {"epochs": 0, "solver_fits": len(references), "gate": [ETHANOL, ACETALDEHYDE],
                  "references": [int(item) for item in references],
                  "reference_means": [round(float(item), 1) for item in means[references]],
                  "reference_gap": round(gap, 4), "support": support,
                  "preserved_toluene": "toluene_support" in side}
        return Predictor(parent.dataset, native, side), recipe
    if action == "toluene_check" and parent.dataset == "A":
        if not has_decay_pair(parent.extension):
            raise ValueError("toluene_check requires a mounted decay_pair model")
        check = fit_decay(x, y, ACETONE, TOLUENE)
        side = {**parent.extension, **{f"toluene_{name}": value for name, value in check.items()}}
        recipe = {"epochs": 0, "solver_fits": 1, "gate": [TOLUENE], "support": len(check["support"])}
        return Predictor(parent.dataset, native, side), recipe
    if action in ("residual", "residual_light") and parent.dataset in ("B", "C"):
        side = fit_residual(native, x, y, light=action == "residual_light")
        return Predictor(parent.dataset, native, side), {"epochs": 0, "solver_fits": 1, "trees": len(side["children"]),
                                                        "strength": float(side["strength"]), "depth": int(side["depth"])}
    raise ValueError("action is not available for this task")
