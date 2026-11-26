"""飞行功率候选及其训练与推理逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from extra_features import phase_weights
from features import PLATFORMS
from tree_models import COMMON, POWERED_W, fit_booster, load_tree, save_tree

FAMILIES = ("long_lgb", "physical", "phase", "c_focus", "xgb", "cat", "xgb_anchor", "cat_anchor")
XGB_MODELS = ("xgb", "xgb_anchor")
CAT_MODELS = ("cat", "cat_anchor")
MAX_BOOST = 780
PRIOR_INPUTS = [
    "payload_level", "horizontal_speed", "wind_speed", "induced_proxy",
    "vertical_work_proxy", "prefer_vertical", "accel_horizontal_rot5", "yaw_abs",
]


def physical_matrix(x: pd.DataFrame, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    raw = x[PRIOR_INPUTS].to_numpy(dtype=np.float64)
    z = np.clip(np.nan_to_num((raw - means) / scales), -6, 6)
    return np.column_stack((np.ones(len(z)), z))


def fit_prior(x: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> dict:
    raw = x[PRIOR_INPUTS].to_numpy(dtype=np.float64)
    means = np.nanmean(raw, axis=0)
    scales = np.maximum(1e-3, np.nanstd(raw, axis=0))
    design = physical_matrix(x, means, scales)
    coefficients = {}
    for platform in PLATFORMS:
        mask = x[f"platform_{platform}"].eq(1).to_numpy()
        d, target, weight = design[mask], y[mask], w[mask]
        reg = np.diag([0, *[0.01 * weight.sum()] * (d.shape[1] - 1)])
        robust = weight
        for _ in range(5):
            beta = np.linalg.solve(d.T @ (d * robust[:, None]) + reg, d.T @ (robust * target))
            distance = np.maximum(np.abs(target - d @ beta), 25)
            robust = weight * (25 / distance)
        coefficients[platform] = beta.tolist()
    return {"mean": means.tolist(), "scale": scales.tolist(), "coefficients": coefficients}


def predict_prior(x: pd.DataFrame, prior: dict) -> np.ndarray:
    design = physical_matrix(x, np.array(prior["mean"]), np.array(prior["scale"]))
    output = np.empty(len(x))
    for platform in PLATFORMS:
        mask = x[f"platform_{platform}"].eq(1).to_numpy()
        output[mask] = design[mask] @ np.asarray(prior["coefficients"][platform])
    return output


def fit_lgb(
    x: pd.DataFrame, y: np.ndarray, w: np.ndarray,
    valid: tuple | None, rounds: int,
) -> lgb.Booster:
    params = {
        **COMMON, "objective": "regression_l1", "metric": "l1",
        "num_leaves": 63, "max_depth": 9, "min_data_in_leaf": 95,
        "feature_fraction": 0.74, "num_threads": 6,
    }
    dataset = lgb.Dataset(x, label=y, weight=w)
    if valid is None:
        return lgb.train(params, dataset, num_boost_round=rounds)
    vx, vy, vw = valid
    validation = lgb.Dataset(vx, label=vy, weight=vw, reference=dataset)
    return lgb.train(
        params, dataset, num_boost_round=rounds, valid_sets=[validation],
        callbacks=[lgb.early_stopping(55, verbose=False)],
    )


def focus_set(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, membership: np.ndarray) -> tuple:
    keep = membership >= 0.1
    return x.loc[keep], y[keep], w[keep] * membership[keep]


def fit_offsets(x: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> dict:
    offsets = {}
    for platform in PLATFORMS:
        keep = x[f"platform_{platform}"].eq(1).to_numpy()
        target, weights = y[keep], w[keep]
        order = np.argsort(target)
        median = np.searchsorted(np.cumsum(weights[order]), 0.5 * weights.sum())
        offsets[platform] = float(target[order[median]])
    return offsets


def power_offsets(x: pd.DataFrame, offsets: dict) -> np.ndarray:
    return sum(x[f"platform_{platform}"].to_numpy() * value for platform, value in offsets.items())


def fit_active(
    family: str, x: pd.DataFrame, y: np.ndarray, w: np.ndarray,
    valid: tuple | None = None, frozen: dict | None = None, device: str = "cpu",
) -> tuple[dict, dict]:
    config = frozen or {}
    default = config.get("main", MAX_BOOST)
    models = {}
    settings = {}
    if family.endswith("_anchor"):
        offsets = fit_offsets(x, y, w)
        models["offsets"] = offsets
        y = y - power_offsets(x, offsets)
        if valid is not None:
            vx, vy, vw = valid
            valid = vx, vy - power_offsets(vx, offsets), vw
    if family in XGB_MODELS:
        import xgboost as xgb

        dx = xgb.QuantileDMatrix(x, label=y, weight=w, max_bin=128, nthread=6)
        use_stop = valid is not None
        evaluations = []
        if use_stop:
            vx, vy, vw = valid
            dv = xgb.QuantileDMatrix(vx, label=vy, weight=vw, ref=dx, max_bin=128, nthread=6)
            evaluations = [(dv, "validation")]
        params = {
            "tree_method": "hist", "device": device, "nthread": 6, "max_bin": 128,
            "objective": "reg:squarederror" if family == "xgb_anchor" else "reg:pseudohubererror",
            "eval_metric": "mae",
            # 固定的鲁棒初值使梯度在尚未拟合时保持有效。
            "base_score": float(np.median(y)), "eta": 0.045, "max_depth": 8,
            "grow_policy": "lossguide", "max_leaves": 63, "subsample": 0.84,
            "colsample_bytree": 0.78, "lambda": 16.0, "min_child_weight": 35, "seed": 156,
        }
        if family == "xgb":
            params["huber_slope"] = 24.0
        booster = xgb.train(
            params, dx, num_boost_round=default, evals=evaluations, verbose_eval=False,
            early_stopping_rounds=55 if use_stop else None,
        )
        best = booster.best_iteration + 1 if use_stop else default
        booster = booster[:best]
        booster.set_param({"device": "cpu", "nthread": 6})
        return {**models, "main": booster}, {"main": best}
    if family in CAT_MODELS:
        from catboost import CatBoostRegressor, Pool

        train = Pool(x, label=y, weight=w)
        test = None
        if valid is not None:
            vx, vy, vw = valid
            test = Pool(vx, label=vy, weight=vw)
        model = CatBoostRegressor(
            loss_function="RMSE" if family == "cat_anchor" else "MAE",
            eval_metric="MAE", iterations=config.get("main", 900), learning_rate=0.07,
            depth=7, l2_leaf_reg=8.0, random_seed=156, thread_count=6,
            task_type="GPU" if device == "cuda" else "CPU",
            **({"random_strength": 0.4} if device == "cpu" else {}),
            allow_writing_files=False, verbose=False,
        )
        model.fit(train, eval_set=test, early_stopping_rounds=65 if valid is not None else None)
        return {**models, "main": model}, {"main": model.tree_count_}
    if family == "physical":
        prior = fit_prior(x, y, w)
        models["prior"] = prior
        y = y - predict_prior(x, prior)
        if valid is not None:
            vx, vy, vw = valid
            valid = vx, vy - predict_prior(vx, prior), vw
    elif family == "phase":
        gates = phase_weights(x)
        val_gates = phase_weights(valid[0]) if valid is not None else None
        for mode, gate in gates.items():
            phase_valid = focus_set(*valid, val_gates[mode]) if valid is not None else None
            branch = fit_lgb(*focus_set(x, y, w, gate), phase_valid, config.get(mode, 540))
            models[mode] = branch
            settings[mode] = branch.best_iteration or branch.current_iteration()
        return models, settings
    elif family == "c_focus":
        train_mask = x["platform_C"].eq(1).to_numpy()
        x, y, w = x.loc[train_mask], y[train_mask], w[train_mask]
        if valid is not None:
            vx, vy, vw = valid
            val_mask = vx["platform_C"].eq(1).to_numpy()
            valid = vx.loc[val_mask], vy[val_mask], vw[val_mask]
    elif family != "long_lgb":
        raise ValueError(family)
    model = fit_lgb(x, y, w, valid, default)
    models["main"] = model
    settings["main"] = model.best_iteration or model.current_iteration()
    return models, settings


def predict_active(family: str, models: dict, x: pd.DataFrame) -> np.ndarray:
    if family in XGB_MODELS or family in CAT_MODELS:
        if family in XGB_MODELS:
            result = models["main"].inplace_predict(x)
        else:
            result = models["main"].predict(x, thread_count=6, task_type="CPU")
        return result + power_offsets(x, models["offsets"]) if "offsets" in models else result
    if family == "phase":
        return sum(models[mode].predict(x, num_threads=6) * gate
                   for mode, gate in phase_weights(x).items())
    result = models["main"].predict(x, num_threads=6)
    if family == "physical":
        result += predict_prior(x, models["prior"])
    return result


def gate_fraction(gate: lgb.Booster, base: pd.DataFrame) -> np.ndarray:
    probability = gate.predict(base, num_threads=6)
    return np.clip((probability - 0.35) / 0.30, 0, 1)


def train_gate(base: pd.DataFrame, y: np.ndarray, weights: np.ndarray, rounds: int) -> lgb.Booster:
    return fit_booster(base, y >= POWERED_W, weights, rounds, classifier=True)


def save_active(family: str, models: dict, directory: Path) -> dict:
    sidecar = {}
    if "offsets" in models:
        sidecar["offsets"] = f"{family}_offsets.json"
        (directory / sidecar["offsets"]).write_text(json.dumps(models["offsets"]), encoding="utf-8")
    if family in XGB_MODELS:
        filename = f"{family}.ubj"
        (directory / filename).write_bytes(bytes(models["main"].save_raw(raw_format="ubj")))
        return {**sidecar, "main": filename}
    if family in CAT_MODELS:
        from tempfile import TemporaryDirectory

        filename = f"{family}.cbm"
        with TemporaryDirectory(prefix="uav-cat-") as temp:
            ascii_path = Path(temp) / "model.cbm"
            models["main"].save_model(str(ascii_path))
            (directory / filename).write_bytes(ascii_path.read_bytes())
        return {**sidecar, "main": filename}
    trees = {key: value for key, value in models.items() if key != "prior"}
    files = save_tree(family, trees, directory)
    if "prior" in models:
        files["prior"] = f"{family}_prior.json"
        (directory / files["prior"]).write_text(json.dumps(models["prior"]), encoding="utf-8")
    return files


def load_active(family: str, files: dict, directory: Path) -> dict:
    sidecar = {}
    if "offsets" in files:
        sidecar["offsets"] = json.loads((directory / files["offsets"]).read_text(encoding="utf-8"))
    if family in XGB_MODELS:
        import xgboost as xgb

        booster = xgb.Booster(params={"device": "cpu", "nthread": 6})
        booster.load_model(bytearray((directory / files["main"]).read_bytes()))
        return {**sidecar, "main": booster}
    if family in CAT_MODELS:
        from tempfile import TemporaryDirectory
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        with TemporaryDirectory(prefix="uav-cat-") as temp:
            ascii_path = Path(temp) / "model.cbm"
            ascii_path.write_bytes((directory / files["main"]).read_bytes())
            model.load_model(str(ascii_path))
        return {**sidecar, "main": model}
    models = load_tree(directory, {k: v for k, v in files.items() if k != "prior"})
    if "prior" in files:
        models["prior"] = json.loads((directory / files["prior"]).read_text(encoding="utf-8"))
    return models
