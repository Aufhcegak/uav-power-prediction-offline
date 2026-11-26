"""Train and export Spike-Resistant q05 3-Seed Multi-Depth CatBoost + XGBoost models locally on RTX 4060."""

import argparse
import gc
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import warnings

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
CV5_DIR = ROOT / "无人机功率_C平台结果/cv5_unified"
DATA_GZ = ROOT / "无人机功率_C平台结果/colab_seq_film/uav_train.csv.gz"
FOLDS_CSV = CV5_DIR / "folds_5fold_stratified.csv"
MODELS_DIR = CV5_DIR / "trees_q05_models"

KEYS = ["flight", "segment", "t"]
PLATFORMS = ["A", "B", "C", "D"]
MISSION_COLS = {"payload_level", "cmd_speed", "cmd_alt", "speed_varies", "alt_varies"}
CONFIGS = [(0, 6), (1000, 7), (2000, 8)]


def fit_irls_anchor(basis_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray):
    loc = np.mean(basis_tr[:, 1:], axis=0)
    scale = np.maximum(1e-3, np.std(basis_tr[:, 1:], axis=0))
    design = np.column_stack([np.ones(len(basis_tr)), (basis_tr[:, 1:] - loc) / scale])
    penalty = np.diag([0.0, *[0.01 * float(w_tr.sum())] * (design.shape[1] - 1)])
    robust = w_tr.astype(np.float64)
    coeff = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(5):
        lhs = design.T @ (design * robust[:, None]) + penalty
        rhs = design.T @ (robust * y_tr)
        coeff = np.linalg.solve(lhs, rhs)
        resid = np.abs(y_tr - design @ coeff)
        robust = w_tr * (25.0 / np.maximum(resid, 25.0))
    return loc, scale, coeff


def predict_irls_anchor(basis: np.ndarray, loc: np.ndarray, scale: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(basis)), (basis[:, 1:] - loc) / scale])
    return (design @ coeff).astype(np.float32)


def build_phys_bidir_features_q05(plat_df: pd.DataFrame):
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

    seg_q05_h = df.groupby(["flight", "segment"], sort=False)["h_rel"].transform(lambda s: float(np.quantile(s, 0.05))).to_numpy(dtype=np.float32)
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


def save_cat_safe(model: CatBoostRegressor, dst: Path):
    with TemporaryDirectory(prefix="uav_q05_") as tmp:
        p = Path(tmp) / "m.cbm"
        model.save_model(str(p))
        tmp_dst = dst.with_suffix(".cbm.tmp")
        shutil.copy2(p, tmp_dst)
        tmp_dst.replace(dst)


def save_xgb_safe(model, dst: Path):
    with TemporaryDirectory(prefix="uav_q05_") as tmp:
        p = Path(tmp) / "m.json"
        model.save_model(str(p))
        tmp_dst = dst.with_suffix(".json.tmp")
        shutil.copy2(p, tmp_dst)
        tmp_dst.replace(dst)


def load_xgb_safe(src: Path, is_clf: bool, device: str):
    with TemporaryDirectory(prefix="uav_q05_") as tmp:
        p = Path(tmp) / "m.json"
        shutil.copy2(src, p)
        cls = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
        m = cls(device=device)
        m.load_model(str(p))
        return m


def ensure_gate_and_anchor(mdir: Path, fold: int, Xg_tr, Xg_va, y_tr, w_tr, b_tr, b_va):
    gate_path = mdir / "gate_xgb.json"
    anchor_path = mdir / "anchor.pkl"
    if gate_path.exists() and anchor_path.exists():
        gate_clf = load_xgb_safe(gate_path, is_clf=True, device="cpu")
        g_prob = gate_clf.predict_proba(Xg_va)[:, 1].astype(np.float32)
        g_clean = np.where(g_prob < 0.02, 0.0, g_prob)
        anc = joblib.load(anchor_path)
        loc, scale, coeff = anc["loc"], anc["scale"], anc["coeff"]
    else:
        gate_clf = xgb.XGBClassifier(
            n_estimators=320, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cpu", n_jobs=12, eval_metric="logloss", random_state=42 + fold,
        )
        gate_clf.fit(Xg_tr, (y_tr >= 30.0).astype(np.int32), sample_weight=w_tr)
        g_prob = gate_clf.predict_proba(Xg_va)[:, 1].astype(np.float32)
        g_clean = np.where(g_prob < 0.02, 0.0, g_prob)
        save_xgb_safe(gate_clf, gate_path)

        pow_tr = y_tr >= 30.0
        w_pow = (w_tr[pow_tr] / np.mean(w_tr[pow_tr])).astype(np.float32)
        loc, scale, coeff = fit_irls_anchor(b_tr[pow_tr], y_tr[pow_tr], w_pow)
        tmp_a = anchor_path.with_suffix(".pkl.tmp")
        joblib.dump({"loc": loc, "scale": scale, "coeff": coeff}, tmp_a)
        tmp_a.replace(anchor_path)

    pow_tr = y_tr >= 30.0
    w_pow = (w_tr[pow_tr] / np.mean(w_tr[pow_tr])).astype(np.float32)
    phys_tr = predict_irls_anchor(b_tr[pow_tr], loc, scale, coeff)
    phys_va = predict_irls_anchor(b_va, loc, scale, coeff)
    res_tr = (y_tr[pow_tr] - phys_tr).astype(np.float32)
    return g_clean, phys_va, res_tr, w_pow, pow_tr


def run_stage_model(stage: str, plats: list[str]):
    t0 = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_GZ, dtype={"flight": str, "t": str, "platform": str})
    df = df.loc[df["platform"].isin(plats)].reset_index(drop=True)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"flight": str, "platform": str})
    flight_to_fold = dict(zip(folds_df["flight"], folds_df["fold"]))
    plat_dfs = {p: df.loc[df["platform"] == p].reset_index(drop=True) for p in plats}
    del df
    gc.collect()

    col_name = "catboost_q05_ms" if stage == "cat" else "xgboost_q05_ms"

    for plat in plats:
        p_t0 = time.perf_counter()
        plat_df = plat_dfs.pop(plat)
        plat_oof_path = MODELS_DIR / f"oof_{stage}_{plat}.csv.gz"
        done_models = all(
            (MODELS_DIR / f"fold{fold}_{plat}" / f"reg_{stage}_s{si}.{'cbm' if stage == 'cat' else 'json'}").exists()
            and (MODELS_DIR / f"fold{fold}_{plat}" / "gate_xgb.json").exists()
            and (MODELS_DIR / f"fold{fold}_{plat}" / "anchor.pkl").exists()
            for fold in range(5) for si in range(len(CONFIGS))
        )
        if plat_oof_path.exists() and done_models:
            print(f"[Skip {stage}_q05 | Plat {plat}] already complete", flush=True)
            del plat_df
            gc.collect()
            continue

        X_reg, X_gate, basis_all = build_phys_bidir_features_q05(plat_df)
        y_all = plat_df["power"].to_numpy(dtype=np.float32)
        f_all = plat_df["flight"].map(flight_to_fold).to_numpy(dtype=np.int32)
        flight_arr = plat_df["flight"].to_numpy()
        _, inv, counts = np.unique(flight_arr, return_inverse=True, return_counts=True)
        w_raw = (1.0 / counts[inv]).astype(np.float64)
        w_all = (w_raw / np.mean(w_raw)).astype(np.float32)
        oof_pred = np.zeros(len(plat_df), dtype=np.float32)

        for fold in range(5):
            f_t0 = time.perf_counter()
            mdir = MODELS_DIR / f"fold{fold}_{plat}"
            mdir.mkdir(parents=True, exist_ok=True)
            fold_npy = mdir / f"oof_{stage}.npy"
            val_mask = f_all == fold
            ext = "cbm" if stage == "cat" else "json"
            fold_ready = (
                fold_npy.exists()
                and (mdir / "gate_xgb.json").exists()
                and (mdir / "anchor.pkl").exists()
                and all((mdir / f"reg_{stage}_s{si}.{ext}").exists() for si in range(len(CONFIGS)))
            )
            if fold_ready:
                oof_pred[val_mask] = np.load(fold_npy)
                print(f"[Resume {stage}_q05 | Plat {plat} | Fold {fold}] loaded", flush=True)
                continue

            tr_mask = ~val_mask
            y_tr = y_all[tr_mask]
            w_tr = (w_all[tr_mask] / np.mean(w_all[tr_mask])).astype(np.float32)
            g_clean, phys_va, res_tr, w_pow, pow_tr = ensure_gate_and_anchor(
                mdir, fold, X_gate[tr_mask], X_gate[val_mask], y_tr, w_tr, basis_all[tr_mask], basis_all[val_mask],
            )
            Xr_tr_all = X_reg[tr_mask]
            Xr_pow = Xr_tr_all[pow_tr]
            Xr_va = X_reg[val_mask]
            del Xr_tr_all

            pred_sum = np.zeros(val_mask.sum(), dtype=np.float32)
            for si, (seed_off, depth) in enumerate(CONFIGS):
                if stage == "cat":
                    m = CatBoostRegressor(
                        loss_function="Huber:delta=15.0", iterations=2600, depth=depth,
                        learning_rate=0.045, l2_leaf_reg=3.0, random_seed=42 + fold + seed_off,
                        task_type="GPU", gpu_ram_part=0.35, devices="0", verbose=0,
                    )
                    m.fit(Xr_pow, res_tr, sample_weight=w_pow)
                    pred_sum += g_clean * np.clip(phys_va + m.predict(Xr_va).astype(np.float32), 0.0, None)
                    save_cat_safe(m, mdir / f"reg_cat_s{si}.cbm")
                else:
                    m = xgb.XGBRegressor(
                        objective="reg:absoluteerror", n_estimators=1800, max_depth=depth,
                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                        min_child_weight=30, reg_lambda=5.0, tree_method="hist", device="cuda",
                        random_state=42 + fold + seed_off,
                    )
                    m.fit(Xr_pow, res_tr, sample_weight=w_pow)
                    pred_sum += g_clean * np.clip(phys_va + m.predict(Xr_va).astype(np.float32), 0.0, None)
                    m.set_params(device="cpu")
                    save_xgb_safe(m, mdir / f"reg_xgb_s{si}.json")
                del m
            fold_pred = (pred_sum / len(CONFIGS)).astype(np.float32)
            oof_pred[val_mask] = fold_pred
            tmp_npy = mdir / f"oof_{stage}.tmp.npy"
            np.save(tmp_npy, fold_pred)
            tmp_npy.replace(fold_npy)
            f_mae = float(pd.DataFrame({"flight": flight_arr[val_mask], "err": np.abs(fold_pred - y_all[val_mask])}).groupby("flight", sort=False)["err"].mean().mean())
            del Xr_pow, Xr_va
            gc.collect()
            print(f"[{stage}_q05 | Plat {plat} | Fold {fold}] MAE={f_mae:.5f} W ({time.perf_counter() - f_t0:.1f}s)", flush=True)

        p_mae = float(pd.DataFrame({"flight": flight_arr, "err": np.abs(oof_pred - y_all)}).groupby("flight", sort=False)["err"].mean().mean())
        out_df = pd.DataFrame({
            "flight": plat_df["flight"], "segment": plat_df["segment"], "t": plat_df["t"], col_name: oof_pred,
        })
        tmp_gz = MODELS_DIR / f"oof_{stage}_{plat}.tmp.csv.gz"
        out_df.to_csv(tmp_gz, index=False, compression="gzip")
        tmp_gz.replace(plat_oof_path)
        print(f"=== [{stage}_q05 | Plat {plat}] DONE MAE={p_mae:.5f} W ({time.perf_counter() - p_t0:.1f}s) ===", flush=True)
        del X_reg, X_gate, basis_all, plat_df
        gc.collect()

    print(f"Finished stage {stage} in {time.perf_counter() - t0:.1f}s", flush=True)


def run_merge():
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_GZ, dtype={"flight": str, "t": str, "platform": str}, usecols=[*KEYS, "platform", "power"])
    cat_parts = [pd.read_csv(MODELS_DIR / f"oof_cat_{p}.csv.gz", dtype={"flight": str, "t": str}) for p in PLATFORMS]
    xgb_parts = [pd.read_csv(MODELS_DIR / f"oof_xgb_{p}.csv.gz", dtype={"flight": str, "t": str}) for p in PLATFORMS]
    cat_df = pd.concat(cat_parts, ignore_index=True)
    xgb_df = pd.concat(xgb_parts, ignore_index=True)
    merged = cat_df.merge(xgb_df, on=KEYS, validate="1:1")
    merged["t_num"] = merged["t"].astype(np.float64)
    merged = merged.sort_values(["flight", "segment", "t_num"], kind="stable").drop(columns=["t_num"]).reset_index(drop=True)
    assert len(merged) == 836547
    assert np.isfinite(merged[["catboost_q05_ms", "xgboost_q05_ms"]]).all().all()

    out_gz = CV5_DIR / "oof_trees_q05_5fold.csv.gz"
    merged.to_csv(out_gz, index=False, compression="gzip")

    m = df.merge(merged, on=KEYS, validate="1:1")
    summary = {"overall": {}, "platforms": {}, "sentinels": {}}
    for col in ("catboost_q05_ms", "xgboost_q05_ms"):
        m[f"err_{col}"] = np.abs(m["power"] - m[col])
        summary["overall"][col] = float(m.groupby("flight", sort=False)[f"err_{col}"].mean().mean())
        summary["platforms"][col] = {
            p: float(m.loc[m["platform"] == p].groupby("flight", sort=False)[f"err_{col}"].mean().mean())
            for p in PLATFORMS
        }
        summary["sentinels"][col] = {
            fl: float(m.loc[m["flight"] == fl, f"err_{col}"].mean())
            for fl in ("F0025", "F0089", "F0332", "F0414", "F0353")
        }
    (CV5_DIR / "trees_q05_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Trees q05 Summary:", json.dumps(summary, indent=2), flush=True)
    print(f"Merged OOF saved to {out_gz} in {time.perf_counter() - t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["cat", "xgb", "merge"])
    parser.add_argument("--plats", default="A,B,C,D")
    args = parser.parse_args()
    if args.stage == "merge":
        run_merge()
    else:
        plats = [p.strip().upper() for p in args.plats.split(",") if p.strip()]
        for p in plats:
            assert p in PLATFORMS, f"Unknown platform {p}"
        run_stage_model(args.stage, plats)


if __name__ == "__main__":
    main()
