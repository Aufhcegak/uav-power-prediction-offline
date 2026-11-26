"""Fold-frozen platform power specialists with CPU prediction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import PLATFORMS
from physical_anchor import anchor_value, fit_anchor
from physics_c import DISK_COLUMNS, HISTORY_COLUMNS, ROUTE_COLUMNS, flight_physics
from tree_models import POWERED_W, fit_booster, load_tree, save_tree
from wind_physics import force_features, wind_features

BASE_WEIGHT = {"A": .5, "B": .5, "C": .4, "D": .5}


def view_features(raw, extended, platform):
    physics = flight_physics(raw, extended)
    core = pd.concat([extended, physics, wind_features(raw, extended)], axis=1)
    if platform != "C":
        return {"specialist": core}
    rotor = pd.concat([extended, physics[DISK_COLUMNS + HISTORY_COLUMNS + ROUTE_COLUMNS]], axis=1)
    force = pd.concat([core, force_features(raw, core)], axis=1)
    return {"rotor": rotor, "force": force}


def train_platform(raw, extended, platform, model_dir: Path):
    powered = raw["power"].to_numpy(dtype=np.float32) >= POWERED_W
    counts = raw.groupby("flight")["flight"].transform("size").to_numpy()
    weights = 1. / counts[powered]
    weights /= weights.mean()
    target = raw["power"].to_numpy(dtype=np.float32)[powered]
    views = view_features(raw, extended, platform)
    model_dir.mkdir(exist_ok=True, parents=True)
    anchors = {}
    for name, view in views.items():
        train = view.loc[powered]
        fitted = fit_anchor(train, target, weights)
        residual = target - anchor_value(train, fitted)
        rounds = 520 if platform == "C" else 290
        model = fit_booster(train, residual, weights, rounds, wide=True)
        save_tree(platform, {name: model}, model_dir)
        anchors[name] = [value.tolist() for value in fitted]
    (model_dir / f"{platform}.json").write_text(
        json.dumps(anchors, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def predict_hybrid(raw, extended, baseline, gate, model_dir: Path):
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    for platform in PLATFORMS:
        rows = raw["platform"].eq(platform).to_numpy()
        if not rows.any():
            continue
        views = view_features(raw.loc[rows], extended.loc[rows], platform)
        metadata = json.loads((model_dir / f"{platform}.json").read_text(encoding="utf-8"))
        models = load_tree(model_dir, {name: f"{platform}_{name}.txt" for name in views})
        predictions = []
        for name, view in views.items():
            fit = tuple(np.asarray(value) for value in metadata[name])
            model = models[name]
            estimate = anchor_value(view, fit) + model.predict(
                view[model.feature_name()], num_threads=6, validate_features=True,
            )
            predictions.append(gate[rows] * np.maximum(0., estimate))
        base_weight = BASE_WEIGHT[platform]
        prediction[rows] = base_weight * baseline[rows] + (1. - base_weight) * np.mean(predictions, axis=0)
    return prediction
