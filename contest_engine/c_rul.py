"""C-only portable residual heads fitted on authorized training features.

The native MLP, input normalization, target scaling and Adam state are retained.
Engine/cycle identifiers and validation labels never enter this model's features.
"""
import json
import hashlib
import copy
import math
from pathlib import Path
import numpy as np

KIND = 'c.rul.regime.v1'
RECIPES = {
    'c_raw': {'regime': False, 'iterations': 120},
    'c_regime': {'regime': True, 'iterations': 120},
    'c_regime_long': {'regime': True, 'iterations': 240},
    'c_health': {'regime': True, 'iterations': 120, 'health_features': True},
    'c_bounded': {'regime': True, 'iterations': 120},
    'c_bounded_narrow': {'regime': True, 'iterations': 120},
    'c_bounded_shrink': {'regime': True, 'iterations': 120},
}


def _indices(native, name):
    return np.asarray([i for i, p in enumerate(native.meta['paths'])
                       if json.loads(p)[0] == ['key', name]], dtype=np.intp)


def _regimes(x, side):
    settings = (x[:, side['setting_indices']] - side['setting_mean']) / side['setting_scale']
    distance = ((settings[:, None, :] - side['centers'][None, :, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1)


def features(native, x, side):
    hidden, logits = native.forward(native.normalized(x))
    health = bool(side.get('health_features', False))
    pieces = [x[:, np.r_[side['sensor_indices'], side['setting_indices']]], logits] if health else [x, hidden, logits]
    if bool(side['use_regime']):
        labels = _regimes(x, side)
        centered = (x[:, side['sensor_indices']] - side['sensor_mean'][labels]) / side['sensor_scale'][labels]
        pieces += [np.clip(centered, -12, 12), np.eye(len(side['centers']))[labels]]
    return np.column_stack(pieces), logits[:, 0]


def _export_trees(fitted, depth):
    rounds = len(fitted._predictors)
    limit = fitted.max_leaf_nodes
    size = 2 * limit - 1
    out = {'features': np.zeros((rounds, size), dtype=np.intp),
           'thresholds': np.zeros((rounds, size)), 'values': np.zeros((rounds, size)),
           'children': np.zeros((rounds, size, 2), dtype=np.intp),
           'depth': np.array(depth), 'intercept': np.array(fitted._baseline_prediction.item()),
           'strength': np.array(1.)}
    for i, predictors in enumerate(fitted._predictors):
        nodes = predictors[0].nodes
        n = len(nodes)
        leaves = nodes['is_leaf'].astype(bool)
        out['features'][i, :n] = nodes['feature_idx']
        out['thresholds'][i, :n] = nodes['num_threshold']
        out['values'][i, :n] = nodes['value']
        out['children'][i, :n, 0] = np.where(leaves, np.arange(n), nodes['left'])
        out['children'][i, :n, 1] = np.where(leaves, np.arange(n), nodes['right'])
    return out


def fit(native, x, y, action):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import HistGradientBoostingRegressor
    from .model import tree_output

    recipe = RECIPES[action]
    seed = native.meta['seed']
    side = {'kind': np.array(KIND), 'use_regime': np.array(recipe['regime']),
            'sensor_indices': _indices(native, 'sensors'),
            'setting_indices': _indices(native, 'settings'),
            'health_features': np.array(recipe.get('health_features', False))}
    if len(side['sensor_indices']) != 21 or len(side['setting_indices']) != 3:
        raise ValueError('C residual head requires the frozen 21-sensor 3-setting schema')
    if recipe.get('health_features'):
        side['sensor_indices'] = side['sensor_indices'][[1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]]
    if recipe['regime']:
        settings = x[:, side['setting_indices']]
        side['setting_mean'] = settings.mean(axis=0)
        side['setting_scale'] = np.maximum(settings.std(axis=0), 1e-6)
        normalized = (settings - side['setting_mean']) / side['setting_scale']
        clusters = MiniBatchKMeans(n_clusters=6, random_state=seed, n_init=3,
                                  batch_size=2048, max_iter=60).fit(normalized)
        side['centers'] = clusters.cluster_centers_
        labels = _regimes(x, side)
        sensors = x[:, side['sensor_indices']]
        floor = np.maximum(sensors.std(axis=0) * .001, 1e-6)
        means, scales = [], []
        for k in range(len(side['centers'])):
            part = sensors[labels == k]
            means.append(part.mean(axis=0) if len(part) else sensors.mean(axis=0))
            scales.append(np.maximum(part.std(axis=0), floor) if len(part) else np.maximum(sensors.std(axis=0), floor))
        side['sensor_mean'], side['sensor_scale'] = np.asarray(means), np.asarray(scales)
    view, logits = features(native, x, side)
    target = (y - native.meta['target_mean']) / native.meta['target_scale'] - logits
    fitted = HistGradientBoostingRegressor(
        loss='squared_error', max_depth=6, max_leaf_nodes=31,
        max_iter=recipe['iterations'], learning_rate=.08,
        min_samples_leaf=40, l2_regularization=10., max_bins=128,
        early_stopping=False, random_state=seed,
    ).fit(view, target)
    side.update(_export_trees(fitted, 6))
    np.testing.assert_allclose(tree_output(view[:256], side), fitted.predict(view[:256]), atol=1e-11, rtol=1e-11)
    config = {'epochs': 0, 'solver_fits': 1, 'trees': recipe['iterations'],
              'depth': 6, 'strength': 1., 'min_samples_leaf': 40, 'l2_regularization': 10.,
              'regime_features': recipe['regime'], 'clusters': 6 if recipe['regime'] else 0,
              'health_features': recipe.get('health_features', False),
              'fit_rows': len(y), 'native_state_unchanged': True,
              'uses_only_inputs_features': True}
    return side, config


def predict(native, x, side):
    from .model import tree_output
    view, logits = features(native, x, side)
    correction = float(side.get('strength', 1.)) * tree_output(view, side)
    bound = float(side.get('correction_bound_cycles', np.inf)) / native.meta['target_scale']
    corrected = logits + np.clip(correction, -bound, bound)
    return np.maximum(0., corrected * native.meta['target_scale'] + native.meta['target_mean'])


def remaining_actions(actions, state, native_sha256, train_sha256, measurement_context):
    """Fit reuse and stage-specific decision completion are separate concepts."""
    seen = {e['action'] for e in state.get('attempts', [])
            if e.get('parent', [None])[0] == native_sha256
            and e.get('train') == train_sha256 and e.get('context') == measurement_context
            and e.get('completed', False)}
    # Only one accepted native epoch per authorized training projection. An
    # accepted epoch changes native SHA, which otherwise reopens itself forever.
    if state.get('c_native_accepted_train') == train_sha256:
        seen.add('c_native_small')
    return [a for a in actions if a not in seen]


def train_bounded(parent, x, y, action, cache_dir=None):
    if action not in ('c_bounded', 'c_bounded_narrow', 'c_bounded_shrink'):
        raise ValueError('unknown C bounded action')
    h = hashlib.sha256()
    h.update(json.dumps({k: parent.native.meta[k] for k in
                        ('paths', 'target_mean', 'target_scale', 'seed')}, sort_keys=True).encode())
    # Head reuse also depends on native features. Keep this correct if a future
    # team integration changes the native base while preserving the sidecar.
    native_arrays = tuple(parent.native.params[k] for k in ('w1', 'b1', 'w2', 'b2'))
    for arr in (*native_arrays, parent.native.mean, parent.native.scale, x, y):
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    fingerprint = h.hexdigest()
    previous = parent.extension
    def matches(side):
        return (side is not None and str(side.get('kind', '')) == KIND
                and str(side.get('fit_fingerprint', '')) == fingerprint
                and str(side.get('core_recipe', '')) == 'regime120')
    cache = Path(cache_dir) / 'c_core.npz' if cache_dir is not None else None
    if not matches(previous) and cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        if matches(saved):
            previous = saved
    reuse = matches(previous)
    if reuse:
        side = dict(previous)
        config = {'epochs': 0, 'solver_fits': 0, 'trees': len(side['children']),
                  'depth': int(side['depth']), 'regime_features': True,
                  'fit_rows': len(y), 'native_state_unchanged': True,
                  'uses_only_inputs_features': True, 'reused_matching_head': True}
    else:
        side, config = fit(parent.native, x, y, action)
        side['fit_fingerprint'] = np.array(fingerprint)
        side['core_recipe'] = np.array('regime120')
        config['reused_matching_head'] = False
    bound = 15. if action == 'c_bounded_narrow' else 30.
    strength = .5 if action == 'c_bounded_shrink' else 1.
    side['correction_bound_cycles'] = np.array(bound)
    side['strength'] = np.array(strength)
    config.update(correction_bound_cycles=bound, strength=strength,
                  fit_fingerprint=fingerprint, bounded_residual=True)
    if cache is not None:
        np.savez_compressed(cache, **side)
    return side, config


def train_c_action(parent, x, y, action, state_dir, native_trial_key):
    """Persist rejected fits too; state never participates in prediction inputs."""
    from .model import Predictor, save, load
    if action != 'c_native_small':
        side, recipe = train_bounded(parent, x, y, action, cache_dir=state_dir)
        return Predictor('C', parent.native, side), recipe
    folder = Path(state_dir) / 'c_native'
    recipe_path = folder / 'recipe.json'
    if recipe_path.is_file():
        cached = json.loads(recipe_path.read_text(encoding='utf-8'))
        if cached.get('native_trial_key') == native_trial_key:
            child = load(folder / 'checkpoint.npz', folder / 'extension.npz', dataset='C')
            recipe = cached['recipe']
            return child, {**recipe, 'solver_fits': 0, 'epochs': 0,
                           'reused_native_candidate': True}
    native = copy.deepcopy(parent.native)
    before, rate = native.meta['steps'], native.meta['learning_rate']
    native.meta['learning_rate'] = rate * .25
    try:
        loss = native.fit(x, y, 1)[0]
    finally:
        native.meta['learning_rate'] = rate
    # Keep the parent's core cache: a rejected native update must not evict it.
    side, recipe = train_bounded(Predictor('C', native), x, y, 'c_bounded')
    recipe.update(epochs=1, lr_factor=.25, steps=native.meta['steps'] - before,
                  train_loss=loss, native_state_unchanged=False,
                  reused_native_candidate=False)
    child = Predictor('C', native, side)
    folder.mkdir(exist_ok=True)
    for name in ('checkpoint.npz', 'extension.npz', 'recipe.json'):
        (folder / name).unlink(missing_ok=True)
    save(child, folder)
    recipe_path.write_text(json.dumps({'native_trial_key': native_trial_key, 'recipe': recipe},
                                       allow_nan=False), encoding='utf-8')
    return child, recipe


def acceptance_guard(parent_metrics, child_metrics):
    old, new = parent_metrics.get('rmse'), child_metrics.get('rmse')
    guard = {'minimum_rmse_gain': None, 'nasa_growth_limit': 1.05,
             'mae_growth_limit': 1.01, 'passed': False, 'reason': 'missing_rmse'}
    if (type(old) not in (int, float) or type(new) not in (int, float)
            or not math.isfinite(old) or not math.isfinite(new) or min(old, new) < 0):
        return guard
    guard['minimum_rmse_gain'] = max(.05, old * .001)
    if old - new < guard['minimum_rmse_gain']:
        guard['reason'] = 'insufficient_rmse_gain'
        return guard
    for name, limit in (('nasa_score', 1.05), ('mae', 1.01)):
        if name not in parent_metrics or name not in child_metrics:
            guard['reason'] = name + '_unavailable'
            return guard
        if name in parent_metrics and name in child_metrics:
            if any(type(m[name]) not in (int, float) or not math.isfinite(m[name]) or m[name] < 0
                   for m in (parent_metrics, child_metrics)):
                guard['reason'] = name + '_unavailable'
                return guard
            if child_metrics[name] > max(parent_metrics[name] * limit, 1e-12):
                guard['reason'] = name + '_regression'
                return guard
    guard.update(passed=True, reason='primary_gain_and_available_risk_metrics_passed')
    return guard
