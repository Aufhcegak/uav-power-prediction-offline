"""任务输入、物理代理量和仅依赖当前与历史样本的特征。"""

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["flight", "segment", "t"]
NUMERIC = [
    "payload_level", "cmd_speed", "cmd_alt", "speed_varies", "alt_varies",
    "v_fwd", "v_right", "v_up", "h_rel", "pitch", "roll", "yaw_rate", "wind_speed",
]
PLATFORMS = ("A", "B", "C", "D")
BASE_FEATURES = [
    *NUMERIC, *(f"platform_{p}" for p in PLATFORMS),
    "wind_head", "wind_right", "horizontal_speed", "tilt",
]
DT = 0.2
GRAVITY = 9.81


def read_flight(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"flight": "string", "t": "string"})


def ordered_flight(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(_order=pd.to_numeric(frame["t"])).sort_values(
        ["flight", "segment", "_order"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def build_features(ordered: pd.DataFrame) -> pd.DataFrame:
    """按已排序的航次构造特征，片段间不共享历史。"""
    frame = ordered[NUMERIC].astype(np.float32)
    angle = np.deg2rad(ordered["wind_dir"].to_numpy(dtype=np.float32))
    v = {column: frame[column].to_numpy() for column in frame}
    platform = ordered["platform"]
    frame = frame.assign(
        **{f"platform_{p}": platform.eq(p).astype(np.float32) for p in PLATFORMS},
        wind_head=v["wind_speed"] * np.cos(angle),
        wind_right=v["wind_speed"] * np.sin(angle),
        horizontal_speed=np.hypot(v["v_fwd"], v["v_right"]),
        tilt=np.hypot(v["pitch"], v["roll"]),
    )
    cos_tilt = np.cos(np.deg2rad(v["pitch"])) * np.cos(np.deg2rad(v["roll"]))
    frame = frame.assign(
        cos_tilt=cos_tilt,
        vertical_abs=np.abs(v["v_up"]),
        yaw_abs=np.abs(v["yaw_rate"]),
        wind_cubed=np.power(np.maximum(v["wind_speed"], 0), 3),
        ambient_fwd=v["v_fwd"] - frame["wind_head"],
        ambient_right=v["v_right"] - frame["wind_right"],
        horizontal_speed_error=frame["horizontal_speed"] - v["cmd_speed"],
    )
    extra = []
    group_keys = [ordered["flight"], ordered["segment"]]
    for _, group in frame.groupby(group_keys, sort=False):
        dynamics = {}
        signals = [
            "v_fwd", "v_right", "v_up", "h_rel", "pitch", "roll", "yaw_rate",
            "wind_head", "wind_right", "horizontal_speed", "tilt",
        ]
        for signal in signals:
            value = group[signal]
            for lag in (1, 3):
                delta = (value - value.shift(lag)) / (lag * DT)
                dynamics[f"{signal}_rate{lag}"] = delta

        a_z = dynamics["v_up_rate3"].clip(-7.0, 7.0).fillna(0)
        tilt_vertical = group["cos_tilt"].clip(lower=0.35)
        thrust = (1 + a_z / GRAVITY) / tilt_vertical
        dynamics["induced_proxy"] = thrust.pow(1.5)
        dynamics["vertical_work_proxy"] = thrust * group["v_up"]
        for window in (4, 12, 30):
            columns = ["v_up", "tilt", "wind_speed", "horizontal_speed", "yaw_abs"]
            rolling = group[columns].rolling(window, min_periods=1)
            for name, values in rolling.mean().items():
                dynamics[f"{name}_mean{window}"] = values
            if window != 4:
                for name, values in rolling.std(ddof=0).items():
                    dynamics[f"{name}_std{window}"] = values
            heights = group["h_rel"].rolling(window, min_periods=1)
            dynamics[f"relative_climb{window}"] = group["h_rel"] - heights.min()
            dynamics[f"height_range{window}"] = heights.max() - heights.min()

        # ---- 双向上下文：同一片段内对称时间窗（批量预测时天然可得），永不触碰 power ----
        columns = ["v_up", "tilt", "wind_speed", "horizontal_speed", "yaw_abs"]
        for window in (4, 12, 30):
            centered = group[columns].rolling(window, min_periods=1, center=True)
            for name, values in centered.mean().items():
                dynamics[f"{name}_cmean{window}"] = values
            if window != 4:
                for name, values in centered.std(ddof=0).items():
                    dynamics[f"{name}_cstd{window}"] = values
        for window in (12, 30):
            heights_c = group["h_rel"].rolling(window, min_periods=1, center=True)
            dynamics[f"relative_climb_c{window}"] = group["h_rel"] - heights_c.min()
            dynamics[f"height_range_c{window}"] = heights_c.max() - heights_c.min()
        for column in ("v_up", "h_rel"):
            span = group[column].rolling(12, min_periods=1, center=True)
            dynamics[f"{column}_cmin12"] = span.min()
            dynamics[f"{column}_cmax12"] = span.max()
        for signal in signals:
            value = group[signal]
            for lag in (1, 3):
                dynamics[f"{signal}_symrate{lag}"] = (
                    value.shift(-lag) - value.shift(lag)
                ) / (2 * lag * DT)
        for column in ("v_up", "h_rel", "horizontal_speed"):
            for step in (10, 30):
                dynamics[f"{column}_future{step}"] = group[column].shift(-step)
        extra.append(pd.DataFrame(dynamics, index=group.index, dtype=np.float32))
    history = pd.concat(extra).reindex(frame.index)
    return pd.concat([frame, history], axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)
