"""离线纯 CPU 最优无损加速版无人机功率预测入口（q05 全树 + TabM + 批推理 ResTCN_v2 + 树共识熔断护栏）。"""

from __future__ import annotations

import os

# 强制纯 CPU 执行，严格符合离线课堂验收要求
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "w1_lib"))

from airborne import gate_fraction, physical_matrix, predict_active  # noqa: E402
from extra_features import extended_features  # noqa: E402
from specialist_models import BASE_WEIGHT, predict_hybrid  # noqa: E402

KEYS = ["flight", "segment", "t"]
PLATFORMS = ("A", "B", "C", "D")
MISSION_COLS = {"payload_level", "cmd_speed", "cmd_alt", "speed_varies", "alt_varies"}
REQUIRED_TELEMETRY = [
    "payload_level", "cmd_speed", "cmd_alt", "speed_varies", "alt_varies",
    "v_fwd", "v_right", "v_up", "h_rel", "pitch", "roll", "yaw_rate",
    "wind_speed", "wind_dir",
]


def discover_csvs(input_path: Path) -> list[Path]:
    """Discover telemetry CSV files from a single CSV, a flight directory, or nested folders."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    direct = sorted(p for p in input_path.glob("*.csv") if p.is_file())
    candidates = direct if direct else sorted(p for p in input_path.rglob("*.csv") if p.is_file())
    telemetry_files = []
    required_min = {"flight", "segment", "t", "v_fwd", "v_up", "h_rel"}
    for p in candidates:
        try:
            header = set(pd.read_csv(p, nrows=0).columns.tolist())
        except Exception:
            continue
        if required_min.issubset(header):
            telemetry_files.append(p)
    if not telemetry_files:
        raise ValueError(f"No valid UAV telemetry CSV files found under: {input_path}")
    return telemetry_files


def load_and_sanitize_inputs(csv_paths: list[Path]) -> pd.DataFrame:
    """Read telemetry CSVs preserving lossless string `t`, original row order, and intra-segment hygiene."""
    frames = []
    for f_idx, p in enumerate(csv_paths):
        df = pd.read_csv(p, dtype={"flight": str, "t": str, "platform": str})
        if "power" in df.columns:
            df = df.drop(columns=["power"])
        if "platform" not in df.columns:
            df["platform"] = "C"
        df["platform"] = df["platform"].fillna("C").astype(str).str.strip().str.upper()
        for col in REQUIRED_TELEMETRY:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["segment"] = pd.to_numeric(df["segment"], errors="coerce").fillna(0).astype(np.int64)
        df["_orig_idx"] = f_idx * 10_000_000 + np.arange(len(df), dtype=np.int64)
        df["_t_num"] = pd.to_numeric(df["t"], errors="coerce").fillna(0.0).astype(np.float64)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["flight", "segment", "_t_num", "_orig_idx"], kind="stable").reset_index(drop=True)
    for col in REQUIRED_TELEMETRY:
        if raw[col].isna().any():
            raw[col] = (
                raw.groupby(["flight", "segment"], sort=False)[col]
                .transform(lambda s: s.ffill().bfill().fillna(0.0))
                .astype(np.float32)
            )
    return raw


def build_phys_bidir_features_q05(plat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spike-resistant q05 drift-corrected bidirectional + attitude micro-oscillation features."""
    df = plat_df.reset_index(drop=True)
    n = len(df)
    idx = np.arange(n, dtype=np.int64)

    lower = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=np.int64)
    for _, block in df.groupby(["flight", "segment"], sort=False):
        b_idx = block.index.to_numpy()
        lower[b_idx] = b_idx[0]
        upper[b_idx] = b_idx[-1]

    def shift_arr(arr: np.ndarray, k: int) -> np.ndarray:
        return arr[np.clip(idx + k, lower, upper)]

    def cdiff(arr: np.ndarray, k: int) -> np.ndarray:
        return (shift_arr(arr, k) - shift_arr(arr, -k)) / (0.4 * k)

    def cdiff2(arr: np.ndarray, k: int) -> np.ndarray:
        dt = 0.2 * k
        return (shift_arr(arr, k) - 2.0 * arr + shift_arr(arr, -k)) / (dt * dt)

    v_fwd = df["v_fwd"].to_numpy(dtype=np.float32)
    v_right = df["v_right"].to_numpy(dtype=np.float32)
    v_up = df["v_up"].to_numpy(dtype=np.float32)
    h_rel = df["h_rel"].to_numpy(dtype=np.float32)
    pitch = df["pitch"].to_numpy(dtype=np.float32)
    roll = df["roll"].to_numpy(dtype=np.float32)
    yaw_rate = df["yaw_rate"].to_numpy(dtype=np.float32)
    wind_speed = df["wind_speed"].to_numpy(dtype=np.float32)
    wind_dir = df["wind_dir"].to_numpy(dtype=np.float32)
    payload_level = df["payload_level"].to_numpy(dtype=np.float32)

    seg_q05_h = df.groupby(["flight", "segment"], sort=False)["h_rel"].transform(
        lambda s: float(np.quantile(s, 0.05))
    ).to_numpy(dtype=np.float32)
    h_base = np.clip(seg_q05_h, -20.0, 0.0).astype(np.float32)
    h_agl = np.maximum(0.0, h_rel - h_base).astype(np.float32)
    h_log = np.log1p(h_agl).astype(np.float32)
    h_rel_clip = np.clip(h_rel, -20.0, 200.0).astype(np.float32)

    v_horiz = np.hypot(v_fwd, v_right).astype(np.float32)
    v_total = np.sqrt(v_fwd ** 2 + v_right ** 2 + v_up ** 2).astype(np.float32)
    tilt = np.hypot(pitch, roll).astype(np.float32)

    pitch_rad, roll_rad, yaw_rad = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw_rate)
    cos_pitch, sin_pitch = np.cos(pitch_rad).astype(np.float32), np.sin(pitch_rad).astype(np.float32)
    cos_roll, sin_roll = np.cos(roll_rad).astype(np.float32), np.sin(roll_rad).astype(np.float32)
    load_factor = (1.0 / np.maximum(0.5, cos_pitch * cos_roll)).astype(np.float32)

    wind_rad = np.deg2rad(wind_dir)
    cos_wind, sin_wind = np.cos(wind_rad).astype(np.float32), np.sin(wind_rad).astype(np.float32)
    wind_x, wind_y = (wind_speed * cos_wind).astype(np.float32), (wind_speed * sin_wind).astype(np.float32)

    load_pow15 = np.power(load_factor, 1.5).astype(np.float32)
    load_vup = (load_factor * v_up).astype(np.float32)
    vup_abs_vup = (v_up * np.abs(v_up)).astype(np.float32)
    v_horiz_sq, v_horiz_cu = (v_horiz ** 2).astype(np.float32), (v_horiz ** 3).astype(np.float32)
    wind_sq, wind_cu = (wind_speed ** 2).astype(np.float32), (wind_speed ** 3).astype(np.float32)
    induced_proxy = (np.sqrt(np.maximum(0.0, 0.25 * (v_up ** 2 + v_horiz_sq) + 25.0)) - 0.5 * v_up).astype(np.float32)
    dh_dt_minus_vup = (cdiff(h_rel, 1) - v_up).astype(np.float32)
    spec_fx = (cdiff(v_fwd, 1) - yaw_rad * v_right).astype(np.float32)
    spec_fy = (cdiff(v_right, 1) + yaw_rad * v_fwd).astype(np.float32)
    spec_f_norm = np.hypot(spec_fx, spec_fy).astype(np.float32)

    cols = {
        "v_fwd": v_fwd, "v_right": v_right, "v_up": v_up, "v_horiz": v_horiz, "v_total": v_total,
        "h_agl": h_agl, "h_rel_clip": h_rel_clip, "h_log": h_log, "pitch": pitch, "roll": roll, "tilt": tilt,
        "load_factor": load_factor, "yaw_rate": yaw_rate, "yaw_abs": np.abs(yaw_rate),
        "wind_speed": wind_speed, "wind_x": wind_x, "wind_y": wind_y,
        "sin_wind": sin_wind, "cos_wind": cos_wind,
        "sin_pitch": sin_pitch, "cos_pitch": cos_pitch, "sin_roll": sin_roll, "cos_roll": cos_roll,
        "load_pow15": load_pow15, "load_vup": load_vup, "vup_abs_vup": vup_abs_vup,
        "v_horiz_sq": v_horiz_sq, "v_horiz_cu": v_horiz_cu, "wind_sq": wind_sq, "wind_cu": wind_cu,
        "v_horiz_wind_x": (v_horiz * wind_x).astype(np.float32),
        "v_horiz_wind_y": (v_horiz * wind_y).astype(np.float32),
        "induced_proxy": induced_proxy, "dh_dt_minus_vup": dh_dt_minus_vup,
        "spec_fx": spec_fx, "spec_fy": spec_fy, "spec_f_norm": spec_f_norm,
        "payload_level": payload_level,
        "cmd_speed": df["cmd_speed"].to_numpy(dtype=np.float32),
        "cmd_alt": df["cmd_alt"].to_numpy(dtype=np.float32),
        "speed_varies": df["speed_varies"].to_numpy(dtype=np.float32),
        "alt_varies": df["alt_varies"].to_numpy(dtype=np.float32),
    }

    lag_channels = {
        "v_up": v_up, "h_agl": h_agl, "v_horiz": v_horiz, "pitch": pitch,
        "roll": roll, "wind_speed": wind_speed, "wind_x": wind_x, "load_factor": load_factor,
    }
    for name, arr in lag_channels.items():
        for k in (-15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15):
            cols[f"{name}_shift_{k}"] = shift_arr(arr, k)

    diff_channels = {
        "v_up": v_up, "h_rel": h_rel, "v_fwd": v_fwd, "v_right": v_right,
        "pitch": pitch, "roll": roll, "wind_x": wind_x, "wind_y": wind_y,
    }
    for name, arr in diff_channels.items():
        for k in (1, 2, 5):
            cols[f"{name}_cdiff_{k}"] = cdiff(arr, k)
        for k in (1, 2):
            cols[f"{name}_cdiff2_{k}"] = cdiff2(arr, k)

    roll_df = pd.DataFrame({
        "v_horiz": v_horiz, "v_up": v_up, "tilt": tilt,
        "wind_speed": wind_speed, "wind_x": wind_x, "load_factor": load_factor,
    })
    gb_roll = roll_df.groupby([df["flight"], df["segment"]], sort=False)
    for w in (5, 11, 25):
        r_mean = gb_roll.rolling(w, min_periods=1, center=True).mean().droplevel([0, 1]).reindex(df.index)
        r_std = gb_roll.rolling(w, min_periods=1, center=True).std(ddof=0).fillna(0.0).droplevel([0, 1]).reindex(df.index)
        r_min = gb_roll.rolling(w, min_periods=1, center=True).min().droplevel([0, 1]).reindex(df.index)
        r_max = gb_roll.rolling(w, min_periods=1, center=True).max().droplevel([0, 1]).reindex(df.index)
        for c in roll_df.columns:
            cols[f"{c}_rmean_{w}"] = r_mean[c].to_numpy(dtype=np.float32)
            cols[f"{c}_rstd_{w}"] = r_std[c].to_numpy(dtype=np.float32)
            cols[f"{c}_rmin_{w}"] = r_min[c].to_numpy(dtype=np.float32)
            cols[f"{c}_rmax_{w}"] = r_max[c].to_numpy(dtype=np.float32)

    att_df = pd.DataFrame({"pitch": pitch, "roll": roll, "yaw": yaw_rate})
    gb_att = att_df.groupby([df["flight"], df["segment"]], sort=False)
    for w in (5, 11):
        a_std = gb_att.rolling(w, min_periods=1, center=True).std(ddof=0).fillna(0.0).droplevel([0, 1]).reindex(df.index)
        for c in att_df.columns:
            cols[f"{c}_rstd_{w}"] = a_std[c].to_numpy(dtype=np.float32)

    seg_df = pd.DataFrame({
        "v_horiz": v_horiz, "v_up": v_up, "tilt": tilt,
        "wind_speed": wind_speed, "wind_x": wind_x, "h_agl": h_agl,
    })
    gb_seg = seg_df.groupby([df["flight"], df["segment"]], sort=False)
    for c in seg_df.columns:
        cols[f"{c}_seg_mean"] = gb_seg[c].transform("mean").to_numpy(dtype=np.float32)
        cols[f"{c}_seg_std"] = gb_seg[c].transform("std").fillna(0.0).to_numpy(dtype=np.float32)
        cols[f"{c}_seg_q10"] = gb_seg[c].transform("quantile", 0.10).to_numpy(dtype=np.float32)
        cols[f"{c}_seg_q50"] = gb_seg[c].transform("quantile", 0.50).to_numpy(dtype=np.float32)
        cols[f"{c}_seg_q90"] = gb_seg[c].transform("quantile", 0.90).to_numpy(dtype=np.float32)

    del roll_df, gb_roll, att_df, gb_att, seg_df, gb_seg
    phys_basis = np.column_stack([
        np.ones(n, dtype=np.float64),
        payload_level, load_factor, load_pow15, v_up, load_vup, vup_abs_vup,
        v_horiz, v_horiz_sq, wind_speed, wind_sq, tilt, h_log, induced_proxy,
    ]).astype(np.float64)
    phys_basis = np.nan_to_num(phys_basis, nan=0.0, posinf=0.0, neginf=0.0)

    reg_names = list(cols.keys())
    X_reg = np.nan_to_num(np.column_stack(list(cols.values())).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    del cols
    gc.collect()
    gate_idx = [i for i, name in enumerate(reg_names) if name not in MISSION_COLS and "_seg_" not in name]
    X_gate = X_reg[:, gate_idx]
    return X_reg, X_gate, phys_basis


def build_phys_bidir_features_v1(plat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Within-segment 279+1 bidirectional features matching frozen TabM v1 checkpoints (in_dim=280)."""
    df = plat_df.reset_index(drop=True)
    n = len(df)
    idx = np.arange(n, dtype=np.int64)

    lower = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=np.int64)
    for _, block in df.groupby(["flight", "segment"], sort=False):
        b_idx = block.index.to_numpy()
        lower[b_idx] = b_idx[0]
        upper[b_idx] = b_idx[-1]

    def shift_arr(arr: np.ndarray, k: int) -> np.ndarray:
        return arr[np.clip(idx + k, lower, upper)]

    def cdiff(arr: np.ndarray, k: int) -> np.ndarray:
        return (shift_arr(arr, k) - shift_arr(arr, -k)) / (0.4 * k)

    def cdiff2(arr: np.ndarray, k: int) -> np.ndarray:
        dt = 0.2 * k
        return (shift_arr(arr, k) - 2.0 * arr + shift_arr(arr, -k)) / (dt * dt)

    v_fwd = df["v_fwd"].to_numpy(dtype=np.float32)
    v_right = df["v_right"].to_numpy(dtype=np.float32)
    v_up = df["v_up"].to_numpy(dtype=np.float32)
    h_rel = df["h_rel"].to_numpy(dtype=np.float32)
    pitch = df["pitch"].to_numpy(dtype=np.float32)
    roll = df["roll"].to_numpy(dtype=np.float32)
    yaw_rate = df["yaw_rate"].to_numpy(dtype=np.float32)
    wind_speed = df["wind_speed"].to_numpy(dtype=np.float32)
    wind_dir = df["wind_dir"].to_numpy(dtype=np.float32)
    payload_level = df["payload_level"].to_numpy(dtype=np.float32)

    v_horiz = np.hypot(v_fwd, v_right).astype(np.float32)
    v_total = np.sqrt(v_fwd ** 2 + v_right ** 2 + v_up ** 2).astype(np.float32)
    h_log = np.log1p(np.clip(h_rel, 0.0, None)).astype(np.float32)
    tilt = np.hypot(pitch, roll).astype(np.float32)

    pitch_rad, roll_rad, yaw_rad = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw_rate)
    cos_pitch, sin_pitch = np.cos(pitch_rad).astype(np.float32), np.sin(pitch_rad).astype(np.float32)
    cos_roll, sin_roll = np.cos(roll_rad).astype(np.float32), np.sin(roll_rad).astype(np.float32)
    load_factor = (1.0 / np.maximum(0.5, cos_pitch * cos_roll)).astype(np.float32)

    wind_rad = np.deg2rad(wind_dir)
    cos_wind, sin_wind = np.cos(wind_rad).astype(np.float32), np.sin(wind_rad).astype(np.float32)
    wind_x, wind_y = (wind_speed * cos_wind).astype(np.float32), (wind_speed * sin_wind).astype(np.float32)

    load_pow15 = np.power(load_factor, 1.5).astype(np.float32)
    load_vup = (load_factor * v_up).astype(np.float32)
    vup_abs_vup = (v_up * np.abs(v_up)).astype(np.float32)
    v_horiz_sq, v_horiz_cu = (v_horiz ** 2).astype(np.float32), (v_horiz ** 3).astype(np.float32)
    wind_sq, wind_cu = (wind_speed ** 2).astype(np.float32), (wind_speed ** 3).astype(np.float32)
    induced_proxy = (np.sqrt(np.maximum(0.0, 0.25 * (v_up ** 2 + v_horiz_sq) + 25.0)) - 0.5 * v_up).astype(np.float32)
    dh_dt_minus_vup = (cdiff(h_rel, 1) - v_up).astype(np.float32)
    spec_fx = (cdiff(v_fwd, 1) - yaw_rad * v_right).astype(np.float32)
    spec_fy = (cdiff(v_right, 1) + yaw_rad * v_fwd).astype(np.float32)
    spec_f_norm = np.hypot(spec_fx, spec_fy).astype(np.float32)

    cols = {
        "v_fwd": v_fwd, "v_right": v_right, "v_up": v_up, "v_horiz": v_horiz, "v_total": v_total,
        "h_rel": h_rel, "h_log": h_log, "pitch": pitch, "roll": roll, "tilt": tilt,
        "load_factor": load_factor, "yaw_rate": yaw_rate, "yaw_abs": np.abs(yaw_rate),
        "wind_speed": wind_speed, "wind_x": wind_x, "wind_y": wind_y,
        "sin_wind": sin_wind, "cos_wind": cos_wind,
        "sin_pitch": sin_pitch, "cos_pitch": cos_pitch, "sin_roll": sin_roll, "cos_roll": cos_roll,
        "load_pow15": load_pow15, "load_vup": load_vup, "vup_abs_vup": vup_abs_vup,
        "v_horiz_sq": v_horiz_sq, "v_horiz_cu": v_horiz_cu, "wind_sq": wind_sq, "wind_cu": wind_cu,
        "v_horiz_wind_x": (v_horiz * wind_x).astype(np.float32),
        "v_horiz_wind_y": (v_horiz * wind_y).astype(np.float32),
        "induced_proxy": induced_proxy, "dh_dt_minus_vup": dh_dt_minus_vup,
        "spec_fx": spec_fx, "spec_fy": spec_fy, "spec_f_norm": spec_f_norm,
        "payload_level": payload_level,
        "cmd_speed": df["cmd_speed"].to_numpy(dtype=np.float32),
        "cmd_alt": df["cmd_alt"].to_numpy(dtype=np.float32),
        "speed_varies": df["speed_varies"].to_numpy(dtype=np.float32),
        "alt_varies": df["alt_varies"].to_numpy(dtype=np.float32),
    }

    lag_channels = {
        "v_up": v_up, "h_rel": h_rel, "v_horiz": v_horiz, "pitch": pitch,
        "roll": roll, "wind_speed": wind_speed, "wind_x": wind_x, "load_factor": load_factor,
    }
    for name, arr in lag_channels.items():
        for k in (-15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15):
            cols[f"{name}_shift_{k}"] = shift_arr(arr, k)

    diff_channels = {
        "v_up": v_up, "h_rel": h_rel, "v_fwd": v_fwd, "v_right": v_right,
        "pitch": pitch, "roll": roll, "wind_x": wind_x, "wind_y": wind_y,
    }
    for name, arr in diff_channels.items():
        for k in (1, 2, 5):
            cols[f"{name}_cdiff_{k}"] = cdiff(arr, k)
        for k in (1, 2):
            cols[f"{name}_cdiff2_{k}"] = cdiff2(arr, k)

    roll_df = pd.DataFrame({
        "v_horiz": v_horiz, "v_up": v_up, "tilt": tilt,
        "wind_speed": wind_speed, "wind_x": wind_x, "load_factor": load_factor,
    })
    gb_roll = roll_df.groupby([df["flight"], df["segment"]], sort=False)
    for w in (5, 11, 25):
        r_mean = gb_roll.rolling(w, min_periods=1, center=True).mean().droplevel([0, 1]).reindex(df.index)
        r_std = gb_roll.rolling(w, min_periods=1, center=True).std(ddof=0).fillna(0.0).droplevel([0, 1]).reindex(df.index)
        r_min = gb_roll.rolling(w, min_periods=1, center=True).min().droplevel([0, 1]).reindex(df.index)
        r_max = gb_roll.rolling(w, min_periods=1, center=True).max().droplevel([0, 1]).reindex(df.index)
        for c in roll_df.columns:
            cols[f"{c}_rmean_{w}"] = r_mean[c].to_numpy(dtype=np.float32)
            cols[f"{c}_rstd_{w}"] = r_std[c].to_numpy(dtype=np.float32)
            cols[f"{c}_rmin_{w}"] = r_min[c].to_numpy(dtype=np.float32)
            cols[f"{c}_rmax_{w}"] = r_max[c].to_numpy(dtype=np.float32)

    seg_df = pd.DataFrame({
        "v_horiz": v_horiz, "v_up": v_up, "tilt": tilt,
        "wind_speed": wind_speed, "wind_x": wind_x, "h_rel": h_rel,
    })
    gb_seg = seg_df.groupby([df["flight"], df["segment"]], sort=False)
    for c in seg_df.columns:
        cols[f"{c}_seg_mean"] = gb_seg[c].transform("mean").to_numpy(dtype=np.float32)
        cols[f"{c}_seg_std"] = gb_seg[c].transform("std").fillna(0.0).to_numpy(dtype=np.float32)
        cols[f"{c}_seg_q10"] = gb_seg[c].transform("quantile", 0.10).to_numpy(dtype=np.float32)
        cols[f"{c}_seg_q50"] = gb_seg[c].transform("quantile", 0.50).to_numpy(dtype=np.float32)
        cols[f"{c}_seg_q90"] = gb_seg[c].transform("quantile", 0.90).to_numpy(dtype=np.float32)

    del roll_df, gb_roll, seg_df, gb_seg
    phys_basis = np.column_stack([
        np.ones(n, dtype=np.float64),
        payload_level, load_factor, load_pow15, v_up, load_vup, vup_abs_vup,
        v_horiz, v_horiz_sq, wind_speed, wind_sq, tilt, h_log, induced_proxy,
    ]).astype(np.float64)
    phys_basis = np.nan_to_num(phys_basis, nan=0.0, posinf=0.0, neginf=0.0)

    feat_mat = np.nan_to_num(np.column_stack(list(cols.values())).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    del cols
    gc.collect()
    return feat_mat, phys_basis


def predict_irls_anchor(basis: np.ndarray, loc: np.ndarray, scale: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(basis)), (basis[:, 1:] - loc) / scale])
    return (design @ coeff).astype(np.float32)


def build_segment_features_and_basis_v2(seg_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Spike-resistant q05 h_agl + attitude micro-oscillation features for ResTCN_v2."""
    n = len(seg_df)
    idx = np.arange(n, dtype=np.int64)

    def at(arr: np.ndarray, k: int) -> np.ndarray:
        return arr[np.clip(idx + k, 0, n - 1)]

    def cdiff(arr: np.ndarray, k: int) -> np.ndarray:
        return (at(arr, k) - at(arr, -k)) / (0.4 * k)

    v_fwd = seg_df["v_fwd"].to_numpy(dtype=np.float32)
    v_right = seg_df["v_right"].to_numpy(dtype=np.float32)
    v_up = seg_df["v_up"].to_numpy(dtype=np.float32)
    h_rel = seg_df["h_rel"].to_numpy(dtype=np.float32)
    pitch = seg_df["pitch"].to_numpy(dtype=np.float32)
    roll = seg_df["roll"].to_numpy(dtype=np.float32)
    yaw_rate = seg_df["yaw_rate"].to_numpy(dtype=np.float32)
    wind_speed = seg_df["wind_speed"].to_numpy(dtype=np.float32)
    wind_dir = seg_df["wind_dir"].to_numpy(dtype=np.float32)
    payload_level = seg_df["payload_level"].to_numpy(dtype=np.float32)

    h_base = float(np.clip(np.quantile(h_rel, 0.05), -20.0, 0.0))
    h_agl = np.maximum(0.0, h_rel - h_base).astype(np.float32)
    h_log = np.log1p(h_agl).astype(np.float32)
    h_rel_clip = np.clip(h_rel, -20.0, 200.0).astype(np.float32)

    v_horiz = np.hypot(v_fwd, v_right).astype(np.float32)
    v_total = np.sqrt(v_fwd ** 2 + v_right ** 2 + v_up ** 2).astype(np.float32)
    tilt = np.hypot(pitch, roll).astype(np.float32)

    pitch_rad, roll_rad, yaw_rad = np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw_rate)
    load_factor = (1.0 / np.maximum(0.5, np.cos(pitch_rad) * np.cos(roll_rad))).astype(np.float32)

    wind_rad = np.deg2rad(wind_dir)
    wind_x, wind_y = (wind_speed * np.cos(wind_rad)).astype(np.float32), (wind_speed * np.sin(wind_rad)).astype(np.float32)

    dv_up_1, dv_up_2 = cdiff(v_up, 1), cdiff(v_up, 2)
    dh_1, dh_2 = cdiff(h_rel, 1), cdiff(h_rel, 2)
    dv_fwd_1, dv_fwd_2 = cdiff(v_fwd, 1), cdiff(v_fwd, 2)
    dv_right_1, dv_right_2 = cdiff(v_right, 1), cdiff(v_right, 2)
    dpitch_1, dpitch_2 = cdiff(pitch, 1), cdiff(pitch, 2)
    droll_1, droll_2 = cdiff(roll, 1), cdiff(roll, 2)
    dwx_1, dwx_2 = cdiff(wind_x, 1), cdiff(wind_x, 2)
    dwy_1, dwy_2 = cdiff(wind_y, 1), cdiff(wind_y, 2)

    att_df = pd.DataFrame({"pitch": pitch, "roll": roll, "yaw": yaw_rate})
    att_cols = []
    for w in (5, 11):
        rstd = att_df.rolling(w, min_periods=1, center=True).std(ddof=0).fillna(0.0)
        for c in ("pitch", "roll", "yaw"):
            att_cols.append(rstd[c].to_numpy(dtype=np.float32))

    dh_dt_minus_vup = (dh_1 - v_up).astype(np.float32)
    spec_fx = (dv_fwd_1 - yaw_rad * v_right).astype(np.float32)
    spec_fy = (dv_right_1 + yaw_rad * v_fwd).astype(np.float32)
    load_vup = (load_factor * v_up).astype(np.float32)
    load_pow15 = np.power(load_factor, 1.5).astype(np.float32)
    vup_abs_vup = (v_up * np.abs(v_up)).astype(np.float32)
    v_horiz_sq = (v_horiz ** 2).astype(np.float32)
    wind_sq = (wind_speed ** 2).astype(np.float32)
    v_horiz_wind_x = (v_horiz * wind_x).astype(np.float32)
    induced_proxy = (np.sqrt(np.maximum(0.0, 0.25 * (v_up ** 2 + v_horiz_sq) + 25.0)) - 0.5 * v_up).astype(np.float32)

    feats = np.stack(
        [
            v_fwd, v_right, v_up, v_horiz, v_total, h_agl, h_rel_clip, h_log,
            pitch, roll, tilt, load_factor, yaw_rate, np.abs(yaw_rate),
            wind_speed, wind_x, wind_y,
            dv_up_1, dv_up_2, dh_1, dh_2, dv_fwd_1, dv_fwd_2, dv_right_1, dv_right_2,
            dpitch_1, dpitch_2, droll_1, droll_2, dwx_1, dwx_2, dwy_1, dwy_2,
            *att_cols,
            dh_dt_minus_vup, spec_fx, spec_fy, load_vup, load_pow15, wind_sq, v_horiz_wind_x,
            payload_level,
            seg_df["cmd_speed"].to_numpy(dtype=np.float32),
            seg_df["cmd_alt"].to_numpy(dtype=np.float32),
            seg_df["speed_varies"].to_numpy(dtype=np.float32),
            seg_df["alt_varies"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    basis = np.column_stack([
        np.ones(n, dtype=np.float64),
        payload_level, load_factor, load_pow15, v_up, load_vup, vup_abs_vup,
        v_horiz, v_horiz_sq, wind_speed, wind_sq, tilt, h_log, induced_proxy,
    ]).astype(np.float64)
    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, basis


class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, k_heads: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.r = nn.Parameter(1.0 + 0.05 * torch.randn(1, k_heads, in_features))
        self.s = nn.Parameter(1.0 + 0.05 * torch.randn(1, k_heads, out_features))
        self.bias = nn.Parameter(torch.zeros(1, k_heads, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x * self.r, self.weight, bias=None) * self.s + self.bias


class TabMResidual(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, k_heads: int = 16, dropout: float = 0.15):
        super().__init__()
        self.k_heads = k_heads
        self.fourier = nn.Linear(in_dim, 64)
        self.stem = nn.Sequential(nn.Linear(in_dim + 128, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"lin": BatchEnsembleLinear(hidden, hidden, k_heads), "norm": nn.LayerNorm(hidden), "drop": nn.Dropout(dropout)})
            for _ in range(3)
        ])
        self.head = BatchEnsembleLinear(hidden, 1, k_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * self.fourier(x)
        emb = torch.cat([x, torch.sin(ang), torch.cos(ang)], dim=-1)
        h = self.stem(emb).unsqueeze(1).expand(-1, self.k_heads, -1)
        for blk in self.blocks:
            h = h + blk["drop"](F.silu(blk["norm"](blk["lin"](h))))
        return self.head(h).squeeze(-1)


class DilatedResConvBlock(nn.Module):
    def __init__(self, hidden: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.norm = nn.LayerNorm(hidden)
        self.conv_glu = nn.Conv1d(hidden, hidden * 2, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        h = F.glu(self.conv_glu((self.norm(x) * m).transpose(1, 2)), dim=1)
        return (x + self.proj(self.drop(h)).transpose(1, 2)) * m


class ResPhysTCNFiLM(nn.Module):
    def __init__(self, in_dim: int = 52, hidden: int = 160, nominal_res_std: float = 40.0, dropout: float = 0.15):
        super().__init__()
        self.nominal_res_std = float(max(1.0, nominal_res_std))
        self.stem = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.conv_blocks = nn.ModuleList([DilatedResConvBlock(hidden, 5, d, dropout) for d in (1, 2, 4, 8, 16, 32)])
        self.att_score = nn.Linear(hidden, 1)
        self.film_mlp = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden * 2))
        self.refine_blocks = nn.ModuleList([DilatedResConvBlock(hidden, 3, d, dropout) for d in (1, 2)])
        self.out_norm = nn.LayerNorm(hidden)
        self.gate_head = nn.Linear(hidden, 1)
        self.delta_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, y_phys: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1)
        h = self.stem(x) * m
        for blk in self.conv_blocks:
            h = blk(h, mask)
        att_logits = self.att_score(h).squeeze(-1).masked_fill(mask < 0.5, -1e4)
        att_w = torch.softmax(att_logits, dim=1).unsqueeze(-1)
        att_mean = torch.sum(att_w * h, dim=1)
        att_std = torch.sqrt(torch.sum(att_w * (h - att_mean.unsqueeze(1)) ** 2, dim=1) + 1e-5)
        gamma, beta = self.film_mlp(torch.cat([att_mean, att_std], dim=1)).chunk(2, dim=1)
        h = (h * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)) * m
        for blk in self.refine_blocks:
            h = blk(h, mask)
        h = self.out_norm(h) * m
        gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        delta = self.delta_head(h).squeeze(-1) * self.nominal_res_std
        return gate * F.relu(y_phys + delta)


def load_catboost_buffered(cbm_path: Path, ascii_buf: Path, n_threads: int) -> CatBoostRegressor:
    shutil.copy2(cbm_path, ascii_buf)
    m = CatBoostRegressor(thread_count=n_threads)
    m.load_model(str(ascii_buf))
    return m


def load_xgb_buffered(json_path: Path, ascii_buf: Path, is_clf: bool, n_threads: int):
    shutil.copy2(json_path, ascii_buf)
    cls = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
    m = cls(n_jobs=n_threads, device="cpu")
    m.load_model(str(ascii_buf))
    return m


def predict_w1_5fold(plat_df: pd.DataFrame, model_plat: str, models_dir: Path, n_threads: int) -> np.ndarray:
    telemetry = plat_df.drop(columns=["_orig_idx", "_t_num"], errors="ignore").copy()
    telemetry["platform"] = model_plat
    base, ext = extended_features(telemetry)
    ones = np.ones(len(telemetry), dtype=np.float64)
    zeros = np.zeros(len(telemetry), dtype=np.float64)
    weight = BASE_WEIGHT[model_plat]

    preds = np.zeros(len(telemetry), dtype=np.float64)
    for fold in range(5):
        mdir = models_dir / "w1_models" / f"fold{fold}" / model_plat
        meta = json.loads((mdir / "bundle_meta.json").read_text(encoding="utf-8"))
        own_gate = lgb.Booster(model_str=(mdir / "own_gate.txt").read_text(encoding="utf-8"))
        phys_tree = lgb.Booster(model_str=(mdir / "phys_tree.txt").read_text(encoding="utf-8"))
        phase_dict = {
            k: lgb.Booster(model_str=(mdir / fn).read_text(encoding="utf-8"))
            for k, fn in meta["phase_files"].items()
        }
        loc = np.asarray(meta["loc"], dtype=np.float64)
        scale = np.asarray(meta["scale"], dtype=np.float64)
        coeff = np.asarray(meta["coeff"], dtype=np.float64)

        gate = gate_fraction(own_gate, base)
        specialist = predict_hybrid(telemetry, ext, zeros, ones, mdir) / (1.0 - weight)
        physical_score = physical_matrix(ext, loc, scale) @ coeff + phys_tree.predict(ext, num_threads=n_threads)
        phase_score = predict_active("phase", phase_dict, ext)
        own_assembly = 0.5 * (np.maximum(0.0, phase_score) + np.maximum(0.0, physical_score))
        preds += gate * (weight * own_assembly + (1.0 - weight) * specialist)
    return (preds / 5.0).astype(np.float64)


def predict_restcn_v2_batched(
    plat_df: pd.DataFrame, model_plat: str, models_dir: Path, batch_size: int
) -> np.ndarray:
    """Lossless length-bucketed batched CPU inference for 5x2 ResTCN_v2 models."""
    n = len(plat_df)
    segs = []
    for _, seg_df in plat_df.groupby(["flight", "segment"], sort=False):
        feats, basis = build_segment_features_and_basis_v2(seg_df)
        segs.append((seg_df.index.to_numpy(), feats, basis))

    # 按长度降序分桶，减少 padding 浪费，与 batch=1 数学严格等价（全过程严格乘以 mask）
    segs.sort(key=lambda item: len(item[0]), reverse=True)
    p_tcn2 = np.zeros(n, dtype=np.float64)
    count = 0

    for fold in range(5):
        for seed in (42, 1042):
            ckpt = torch.load(
                models_dir / "res_tcn_v2_models" / f"fold{fold}_{model_plat}_seed{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            net = ResPhysTCNFiLM(
                in_dim=int(ckpt["in_dim"]),
                hidden=int(ckpt["hidden"]),
                nominal_res_std=float(ckpt["nominal_res_std"]),
                dropout=0.15,
            )
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            med = ckpt["med"]
            scale = ckpt["scale"]
            loc, sc_b, coeff = ckpt["loc"], ckpt["sc_b"], ckpt["coeff"]

            with torch.no_grad():
                for start in range(0, len(segs), batch_size):
                    batch = segs[start : start + batch_size]
                    bsz = len(batch)
                    max_len = max(len(b[0]) for b in batch)
                    in_dim = int(ckpt["in_dim"])
                    x_pad = torch.zeros((bsz, max_len, in_dim), dtype=torch.float32)
                    p_pad = torch.zeros((bsz, max_len), dtype=torch.float32)
                    m_pad = torch.zeros((bsz, max_len), dtype=torch.float32)
                    for i, (_, feats, basis) in enumerate(batch):
                        L = len(feats)
                        yp = predict_irls_anchor(basis, loc, sc_b, coeff)
                        aug = np.column_stack([feats, yp]).astype(np.float32)
                        x_std = np.clip((aug - med) / scale, -10.0, 10.0).astype(np.float32)
                        x_pad[i, :L] = torch.from_numpy(x_std)
                        p_pad[i, :L] = torch.from_numpy(yp)
                        m_pad[i, :L] = 1.0
                    out = net(x_pad, p_pad, m_pad).numpy()
                    for i, (row_idx, _, _) in enumerate(batch):
                        L = len(row_idx)
                        p_tcn2[row_idx] += out[i, :L]
            count += 1
    return (p_tcn2 / float(count)).astype(np.float64)


def predict_platform(
    plat_df: pd.DataFrame, model_plat: str, models_dir: Path, manifest: dict, n_threads: int, tcn_batch_size: int
) -> np.ndarray:
    n = len(plat_df)
    weights = manifest["frozen_weights_per_platform"][model_plat]

    # 1. W1 5-Fold (participates in both blend and tree guard)
    p_w1 = predict_w1_5fold(plat_df, model_plat, models_dir, n_threads)

    # 2. q05 bidirectional features shared by CatBoost_q05, XGBoost_q05, LGB_q05_L1, LGB_q05_ET
    X_reg_q05, X_gate_q05, basis_q05 = build_phys_bidir_features_q05(plat_df)
    need_tabm = float(weights.get("tabm_phys", 0.0)) >= 1e-5
    if need_tabm:
        X_v1, basis_v1 = build_phys_bidir_features_v1(plat_df)

    p_lgb_l1 = np.zeros(n, dtype=np.float64)
    p_lgb_et = np.zeros(n, dtype=np.float64)
    p_cat = np.zeros(n, dtype=np.float64)
    p_xgb = np.zeros(n, dtype=np.float64)
    p_tabm = np.zeros(n, dtype=np.float64)

    # 全平台复用单个 ASCII 临时目录缓冲，零冗余目录建删开销
    with TemporaryDirectory(prefix="uav_opt_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cbm_buf = tmp_path / "buf.cbm"
        xgb_buf = tmp_path / "buf.json"

        for fold in range(5):
            # 2a. LightGBM q05 L1 + ET
            lgb_bundle = joblib.load(models_dir / "lgb_q05_models" / f"fold{fold}_{model_plat}.pkl")
            g_lgb = lgb_bundle["gate"].predict_proba(X_gate_q05)[:, 1].astype(np.float64)
            g_lgb = np.where(g_lgb < 0.02, 0.0, g_lgb)
            phys_lgb = predict_irls_anchor(basis_q05, lgb_bundle["loc"], lgb_bundle["scale"], lgb_bundle["coeff"])
            p_lgb_l1 += g_lgb * np.clip(phys_lgb + lgb_bundle["reg_l1"].predict(X_reg_q05), 0.0, None)
            p_lgb_et += g_lgb * np.clip(phys_lgb + lgb_bundle["reg_et"].predict(X_reg_q05), 0.0, None)

            # 2b. 3-Depth Multi-Seed CatBoost q05 + XGBoost q05 (both always needed for 5-tree consensus guard)
            q05_dir = models_dir / "trees_q05_models" / f"fold{fold}_{model_plat}"
            anchor = joblib.load(q05_dir / "anchor.pkl")
            gate_xgb = load_xgb_buffered(q05_dir / "gate_xgb.json", xgb_buf, is_clf=True, n_threads=n_threads)
            g_xgb = gate_xgb.predict_proba(X_gate_q05)[:, 1].astype(np.float64)
            g_xgb = np.where(g_xgb < 0.02, 0.0, g_xgb)
            phys_q05 = predict_irls_anchor(basis_q05, anchor["loc"], anchor["scale"], anchor["coeff"])

            cat_fold = np.zeros(n, dtype=np.float64)
            xgb_fold = np.zeros(n, dtype=np.float64)
            for si in range(3):
                m_cat = load_catboost_buffered(q05_dir / f"reg_cat_s{si}.cbm", cbm_buf, n_threads=n_threads)
                cat_fold += g_xgb * np.clip(phys_q05 + m_cat.predict(X_reg_q05), 0.0, None)
                m_xgb = load_xgb_buffered(q05_dir / f"reg_xgb_s{si}.json", xgb_buf, is_clf=False, n_threads=n_threads)
                xgb_fold += g_xgb * np.clip(phys_q05 + m_xgb.predict(X_reg_q05), 0.0, None)
            p_cat += cat_fold / 3.0
            p_xgb += xgb_fold / 3.0

            # 2c. TabM v1 CPU fp32 (skipped cleanly if frozen weight < 1e-5)
            if need_tabm:
                ckpt_tabm = torch.load(
                    models_dir / "tabm_v1_models" / f"fold{fold}_{model_plat}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                phys_tabm = predict_irls_anchor(basis_v1, ckpt_tabm["loc"], ckpt_tabm["scale"], ckpt_tabm["coeff"])
                X_full = np.column_stack([X_v1, phys_tabm]).astype(np.float32)
                X_std = np.clip((X_full - ckpt_tabm["med"]) / ckpt_tabm["sc"], -8.0, 8.0).astype(np.float32)
                model_tabm = TabMResidual(in_dim=int(ckpt_tabm["in_dim"]), hidden=256, k_heads=16, dropout=0.15)
                model_tabm.load_state_dict(ckpt_tabm["state_dict"])
                model_tabm.eval()
                preds_t = []
                with torch.no_grad():
                    for start in range(0, n, 8192):
                        xb = torch.from_numpy(X_std[start : start + 8192])
                        pred_bk = model_tabm(xb)
                        preds_t.append((torch.median(pred_bk, dim=1).values * float(ckpt_tabm["res_scale"])).numpy())
                res_tabm = np.concatenate(preds_t, axis=0)
                p_tabm += g_xgb * np.clip(phys_tabm + res_tabm, 0.0, None)

    p_lgb_l1 /= 5.0
    p_lgb_et /= 5.0
    p_cat /= 5.0
    p_xgb /= 5.0
    p_tabm /= 5.0

    del X_reg_q05, X_gate_q05, basis_q05
    if need_tabm:
        del X_v1, basis_v1
    gc.collect()

    # 3. Length-bucketed batched ResTCN_v2 (strictly masked within each segment)
    need_tcn2 = float(weights.get("res_tcn_v2", 0.0)) >= 1e-5
    if need_tcn2:
        p_tcn2 = predict_restcn_v2_batched(plat_df, model_plat, models_dir, batch_size=tcn_batch_size)
    else:
        p_tcn2 = np.zeros(n, dtype=np.float64)

    # 4. Tree-Consensus Safety Guard
    tree_med = np.median(np.vstack([p_w1, p_cat, p_xgb, p_lgb_l1, p_lgb_et]), axis=0)
    t_low = float(manifest["guard"]["t_low"])
    t_high = float(manifest["guard"]["t_high"])

    def guard(arr: np.ndarray) -> np.ndarray:
        alpha = np.clip((np.abs(arr - tree_med) - t_low) / max(1e-6, t_high - t_low), 0.0, 1.0)
        return (1.0 - alpha) * arr + alpha * tree_med

    member_preds = {
        "w1_isolated": p_w1,
        "catboost_q05_ms": p_cat,
        "xgboost_q05_ms": p_xgb,
        "lgb_q05_l1": p_lgb_l1,
        "lgb_q05_et": p_lgb_et,
        "tabm_phys": guard(p_tabm) if need_tabm else np.zeros(n, dtype=np.float64),
        "res_tcn_v2": guard(p_tcn2) if need_tcn2 else np.zeros(n, dtype=np.float64),
    }

    blended = np.zeros(n, dtype=np.float64)
    for name, w in weights.items():
        if float(w) > 0.0:
            blended += float(w) * member_preds[name]
    return np.clip(blended, 0.0, None)


def predict(input_path: Path, output_path: Path, tcn_batch_size: int = 16) -> None:
    n_threads = min(16, os.cpu_count() or 4)
    torch.set_num_threads(n_threads)

    models_dir = ROOT_DIR / "models"
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))

    csvs = discover_csvs(input_path)
    raw = load_and_sanitize_inputs(csvs)
    preds = np.empty(len(raw), dtype=np.float64)

    for plat, block in raw.groupby("platform", sort=False):
        model_plat = plat if plat in PLATFORMS else "C"
        plat_df = block.reset_index(drop=True)
        plat_pred = predict_platform(
            plat_df, model_plat, models_dir, manifest, n_threads=n_threads, tcn_batch_size=tcn_batch_size
        )
        preds[block.index.to_numpy()] = plat_pred

    raw["power_pred"] = np.clip(preds, 0.0, None)
    if not np.isfinite(raw["power_pred"]).all():
        raise ValueError("Non-finite power prediction detected")

    out = raw.sort_values("_orig_idx", kind="stable")[KEYS + ["power_pred"]].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True, type=Path, help="输入测试集目录或单条 CSV 路径")
    cli.add_argument("--output", required=True, type=Path, help="输出预测 CSV 路径")
    cli.add_argument("--tcn-batch-size", type=int, default=16, help="ResTCN_v2 长度分桶批大小（默认 16）")
    parsed = cli.parse_args()
    predict(parsed.input, parsed.output, tcn_batch_size=parsed.tcn_batch_size)
