"""LightGBM 模型与冻结配置共用的推理路径。"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import BASE_FEATURES, PLATFORMS

POWERED_W = 30
TREE_NAMES = (
    "snapshot", "temporal", "temporal_wide", "history_only",
    "per_platform", "flight_state", "flight_state_mechanics",
)
DIRECT = ("snapshot", "temporal", "temporal_wide", "history_only")
PHYSICS_COLUMNS = [
    "cos_tilt", "wind_cubed", "ambient_fwd", "ambient_right",
    "induced_proxy", "vertical_work_proxy",
]
COMMON = {
    "learning_rate": 0.045, "min_gain_to_split": 0.02,
    "feature_fraction": 0.84, "bagging_fraction": 0.85, "bagging_freq": 1,
    "lambda_l1": 0.3, "lambda_l2": 14.0, "verbosity": -1, "num_threads": 6,
    "seed": 156, "deterministic": True, "force_col_wise": True,
}


def fit_booster(
    x: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
    iterations: int, wide: bool = False, classifier: bool = False,
    valid: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
) -> lgb.Booster:
    parameters = {
        **COMMON, "objective": "binary" if classifier else "regression_l1",
        "metric": "binary_logloss" if classifier else "l1",
        "num_leaves": 63 if wide else 31,
        "max_depth": 9 if wide else 7,
        "min_data_in_leaf": 105 if wide else 145,
    }
    train = lgb.Dataset(x, label=y, weight=weights)
    if valid is None:
        return lgb.train(parameters, train, num_boost_round=iterations)
    val_x, val_y, val_weights = valid
    validation = lgb.Dataset(val_x, label=val_y, weight=val_weights, reference=train)
    return lgb.train(
        parameters, train, num_boost_round=iterations, valid_sets=[validation],
        callbacks=[lgb.early_stopping(45, first_metric_only=True, verbose=False)],
    )


def add_body_mechanics(x: pd.DataFrame) -> pd.DataFrame:
    # 文档规定偏航向机体右侧转为正。旋转坐标的速度导数需加上转动项。
    yaw = np.deg2rad(x["yaw_rate"]).clip(-4.0, 4.0)
    accel_fwd = x["v_fwd_rate3"] - yaw * x["v_right"]
    accel_right = x["v_right_rate3"] + yaw * x["v_fwd"]
    planar = np.hypot(accel_fwd, accel_right)
    accel_up = x["v_up_rate3"].clip(-7.0, 7.0)
    acceleration_load = np.hypot(1 + accel_up / 9.81, planar / 9.81)
    wind_head, wind_right = x["wind_head"], x["wind_right"]
    return x.assign(
        inertial_accel_fwd=accel_fwd,
        inertial_accel_right=accel_right,
        inertial_planar_accel=planar,
        inertial_load_power=acceleration_load.pow(1.5),
        horizontal_specific_work=(accel_fwd * x["v_fwd"] + accel_right * x["v_right"]),
        wind_accel_alignment=(accel_fwd * wind_head + accel_right * wind_right),
        height_velocity_disagreement=x["h_rel_rate3"] - x["v_up_mean4"],
        tilt_fluctuation=x["tilt_std12"] + x["tilt_std30"],
        horizontal_vertical_wind=np.hypot(x["wind_speed"], x["v_up"]),
    )


def train_tree(
    name: str, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
    valid: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
    rounds: dict[str, int] | None = None,
) -> dict[str, lgb.Booster]:
    rounds = rounds or {}
    if name == "snapshot":
        x = x[BASE_FEATURES]
        if valid is not None:
            vx, vy, vw = valid
            valid = vx[BASE_FEATURES], vy, vw
    if name == "history_only":
        x = x.drop(columns=PHYSICS_COLUMNS)
        if valid is not None:
            vx, vy, vw = valid
            valid = vx.drop(columns=PHYSICS_COLUMNS), vy, vw
    if name in DIRECT:
        return {"direct": fit_booster(
            x, y, weights, rounds.get("direct", 550),
            wide=name in ("temporal_wide", "history_only"), valid=valid,
        )}
    if name == "per_platform":
        by_platform = {}
        for platform in PLATFORMS:
            mask = x[f"platform_{platform}"].eq(1).to_numpy()
            validation = None
            if valid is not None:
                vx, vy, vw = valid
                val_mask = vx[f"platform_{platform}"].eq(1).to_numpy()
                validation = vx.loc[val_mask], vy[val_mask], vw[val_mask]
            by_platform[platform] = fit_booster(
                x.loc[mask], y[mask], weights[mask], rounds.get(platform, 530),
                wide=False, valid=validation,
            )
        return by_platform
    if name == "flight_state_mechanics":
        x = add_body_mechanics(x)
        if valid is not None:
            vx, vy, vw = valid
            valid = add_body_mechanics(vx), vy, vw
    if name not in ("flight_state", "flight_state_mechanics"):
        raise ValueError(f"Unknown tree candidate: {name}")
    powered = y >= POWERED_W
    gate_val = power_val = None
    if valid is not None:
        vx, vy, vw = valid
        val_powered = vy >= POWERED_W
        gate_val = vx, val_powered, vw
        power_val = vx.loc[val_powered], vy[val_powered], vw[val_powered]
    return {
        "gate": fit_booster(
            x, powered, weights, rounds.get("gate", 430), classifier=True, valid=gate_val,
        ),
        "powered": fit_booster(
            x.loc[powered], y[powered], weights[powered], rounds.get("powered", 550),
            wide=True, valid=power_val,
        ),
    }


def predict_tree(name: str, models: dict, x: pd.DataFrame) -> np.ndarray:
    if name in DIRECT:
        return np.maximum(0.0, models["direct"].predict(x[models["direct"].feature_name()]))
    if name == "per_platform":
        prediction = np.full(len(x), np.nan)
        for platform, model in models.items():
            mask = x[f"platform_{platform}"].eq(1).to_numpy()
            if mask.any():
                prediction[mask] = model.predict(x.loc[mask])
        return np.maximum(0.0, prediction)
    if name == "flight_state_mechanics":
        x = add_body_mechanics(x)
    gate = models["gate"].predict(x)
    conditional = models["powered"].predict(x)
    # 远离过渡区时分别输出待机或飞行功率。门限固定，测试阶段不估计。
    powered_fraction = np.clip((gate - 0.35) / 0.30, 0.0, 1.0)
    return powered_fraction * np.maximum(0.0, conditional)


def save_tree(name: str, models: dict, directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for head, model in models.items():
        filename = f"{name}_{head}.txt"
        # Python 打开文件以兼容 Windows 的中文目录。
        (directory / filename).write_text(model.model_to_string(), encoding="utf-8")
        files[head] = filename
    return files


def load_tree(directory: Path, files: dict[str, str]) -> dict:
    return {
        head: lgb.Booster(model_str=(directory / path).read_text(encoding="utf-8"))
        for head, path in files.items()
    }
