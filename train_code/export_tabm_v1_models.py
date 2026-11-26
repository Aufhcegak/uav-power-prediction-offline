"""Export missing frozen CPU weights for TabMResidual (v1) across all 4 platforms and 5 folds (self-contained, no xgboost)."""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
CV5_DIR = ROOT / "无人机功率_C平台结果/cv5_unified"
DATA_GZ = ROOT / "无人机功率_C平台结果/colab_seq_film/uav_train.csv.gz"
FOLDS_CSV = CV5_DIR / "folds_5fold_stratified.csv"
MODELS_DIR = CV5_DIR / "tabm_v1_models"
KEYS = ["flight", "segment", "t"]
PLATFORMS = ["A", "B", "C", "D"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_phys_bidir_features(plat_df: pd.DataFrame):
    """Build strictly within-segment bidirectional + aerodynamic physics features for one platform."""
    df = plat_df.reset_index(drop=True)
    n = len(df)
    idx = np.arange(n, dtype=np.int64)

    lower = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=np.int64)
    for _, block in df.groupby(["flight", "segment"], sort=False):
        b_idx = block.index.to_numpy()
        assert np.array_equal(b_idx, np.arange(b_idx[0], b_idx[-1] + 1))
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

    feat_mat = np.column_stack(list(cols.values())).astype(np.float32)
    del cols
    gc.collect()
    feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_mat, phys_basis


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


def train_and_export_tabm(X_tr, res_tr, w_tr, X_va, seed: int, device: torch.device, epochs: int = 35, batch_size: int = 2048):
    set_seed(seed)
    use_amp = device.type == "cuda"
    res_scale = float(max(1.0, np.std(res_tr)))
    y_norm = (res_tr / res_scale).astype(np.float32)

    model = TabMResidual(in_dim=X_tr.shape[1], hidden=256, k_heads=16, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n = len(X_tr)
    x_cpu = torch.from_numpy(X_tr)
    y_cpu = torch.from_numpy(y_norm)
    w_cpu = torch.from_numpy(w_tr)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_cpu[idx].to(device, non_blocking=use_amp)
            yb = y_cpu[idx].to(device, non_blocking=use_amp)
            wb = w_cpu[idx].to(device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_bk = model(xb)
            pred_f = pred_bk.float()
            loss = ((wb.unsqueeze(1) * torch.abs(pred_f - yb.unsqueeze(1))).sum(dim=0) / (wb.sum() + 1e-8)).mean()

            old_scale = scaler.get_scale() if use_amp else 1.0
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if (not use_amp) or (scaler.get_scale() >= old_scale):
            scheduler.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_va), 8192):
            xb = torch.from_numpy(X_va[start : start + 8192]).to(device, non_blocking=use_amp)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_bk = model(xb)
            med_pred = torch.median(pred_bk.float(), dim=1).values * res_scale
            preds.append(med_pred.cpu().numpy())
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model, optimizer, scheduler, scaler, x_cpu, y_cpu, w_cpu
    gc.collect()
    return np.concatenate(preds, axis=0).astype(np.float32), state_cpu, res_scale


def infer_saved_tabm(ckpt_path: Path, X_va_raw: np.ndarray, b_va: np.ndarray, device: torch.device):
    use_amp = device.type == "cuda"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    phys_va = predict_irls_anchor(b_va, ckpt["loc"], ckpt["scale"], ckpt["coeff"])
    X_va_full = np.column_stack([X_va_raw, phys_va]).astype(np.float32)
    X_va_std = np.clip((X_va_full - ckpt["med"]) / ckpt["sc"], -8.0, 8.0).astype(np.float32)
    del X_va_full
    model = TabMResidual(in_dim=int(ckpt["in_dim"]), hidden=256, k_heads=16, dropout=0.15).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preds = []
    res_scale = float(ckpt["res_scale"])
    with torch.no_grad():
        for start in range(0, len(X_va_std), 8192):
            xb = torch.from_numpy(X_va_std[start : start + 8192]).to(device, non_blocking=use_amp)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_bk = model(xb)
            med_pred = torch.median(pred_bk.float(), dim=1).values * res_scale
            preds.append(med_pred.cpu().numpy())
    del model, ckpt, X_va_std
    gc.collect()
    return phys_va, np.concatenate(preds, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plats", default="A,B,C,D", help="Comma-separated platforms to process")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch execution device")
    args = parser.parse_args()
    selected_plats = [p.strip().upper() for p in args.plats.split(",") if p.strip()]
    for p in selected_plats:
        assert p in PLATFORMS, f"Unknown platform: {p}"
    verify_mode = selected_plats == ["A", "B", "C", "D"]

    t0 = time.perf_counter()
    device = torch.device(args.device)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_GZ, dtype={"flight": str, "t": str, "platform": str})
    df = df.loc[df["platform"].isin(selected_plats)].reset_index(drop=True)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"flight": str, "platform": str})
    cat_ref = pd.read_csv(CV5_DIR / "oof_cat_xgb_5fold.csv.gz", dtype={"flight": str, "t": str}, usecols=[*KEYS, "gate_xgb"])
    df = df.merge(cat_ref, on=KEYS, how="left", validate="1:1")
    del cat_ref
    plat_dfs = {p: df.loc[df["platform"] == p].reset_index(drop=True) for p in selected_plats}
    del df
    gc.collect()
    flight_to_fold = dict(zip(folds_df["flight"], folds_df["fold"]))

    oof_parts = []
    truth_parts = []
    summary = {"platform_mae": {}}

    for plat in selected_plats:
        p_t0 = time.perf_counter()
        plat_df = plat_dfs.pop(plat)
        X_all, basis_all = build_phys_bidir_features(plat_df)
        y_all = plat_df["power"].to_numpy(dtype=np.float32)
        f_all = plat_df["flight"].map(flight_to_fold).to_numpy(dtype=np.int32)
        g_all = plat_df["gate_xgb"].to_numpy(dtype=np.float32)
        flight_arr = plat_df["flight"].to_numpy()
        _, inv, counts = np.unique(flight_arr, return_inverse=True, return_counts=True)
        w_raw = (1.0 / counts[inv]).astype(np.float64)
        w_all = (w_raw / np.mean(w_raw)).astype(np.float32)

        oof_tabm = np.zeros(len(plat_df), dtype=np.float32)
        for fold in range(5):
            f_t0 = time.perf_counter()
            ckpt_path = MODELS_DIR / f"fold{fold}_{plat}.pt"
            val_mask = f_all == fold

            if ckpt_path.exists():
                if not verify_mode:
                    print(f"[Skip tabm_v1 | Plat {plat} | Fold {fold}] exists", flush=True)
                    continue
                phys_va, res_va = infer_saved_tabm(ckpt_path, X_all[val_mask], basis_all[val_mask], device)
                oof_tabm[val_mask] = g_all[val_mask] * np.clip(phys_va + res_va, 0.0, None)
                print(f"[Resume tabm_v1 | Plat {plat} | Fold {fold}] loaded checkpoint", flush=True)
            else:
                tr_mask = ~val_mask
                X_tr_raw, X_va_raw = X_all[tr_mask], X_all[val_mask]
                y_tr = y_all[tr_mask]
                w_tr = (w_all[tr_mask] / np.mean(w_all[tr_mask])).astype(np.float32)
                b_tr, b_va = basis_all[tr_mask], basis_all[val_mask]

                pow_tr = y_tr >= 30.0
                w_pow = (w_tr[pow_tr] / np.mean(w_tr[pow_tr])).astype(np.float32)
                loc, scale, coeff = fit_irls_anchor(b_tr[pow_tr], y_tr[pow_tr], w_pow)
                phys_tr = predict_irls_anchor(b_tr[pow_tr], loc, scale, coeff)
                phys_va = predict_irls_anchor(b_va, loc, scale, coeff)
                res_tr = (y_tr[pow_tr] - phys_tr).astype(np.float32)

                X_pow_full = np.column_stack([X_tr_raw[pow_tr], phys_tr]).astype(np.float32)
                X_va_full = np.column_stack([X_va_raw, phys_va]).astype(np.float32)
                med = np.median(X_pow_full, axis=0).astype(np.float32)
                iqr = np.percentile(X_pow_full, 75, axis=0) - np.percentile(X_pow_full, 25, axis=0)
                std = np.std(X_pow_full, axis=0)
                sc = np.maximum(np.maximum(iqr / 1.349, std), 1e-3).astype(np.float32)

                X_pow_std = np.clip((X_pow_full - med) / sc, -8.0, 8.0).astype(np.float32)
                X_va_std = np.clip((X_va_full - med) / sc, -8.0, 8.0).astype(np.float32)
                del X_tr_raw, X_va_raw, X_pow_full, X_va_full
                gc.collect()

                res_va, state_cpu, res_scale = train_and_export_tabm(
                    X_pow_std, res_tr, w_pow, X_va_std, seed=42 + fold, device=device, epochs=35, batch_size=2048
                )
                oof_tabm[val_mask] = g_all[val_mask] * np.clip(phys_va + res_va, 0.0, None)

                tmp_path = ckpt_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "state_dict": state_cpu,
                        "med": med,
                        "sc": sc,
                        "res_scale": res_scale,
                        "loc": loc,
                        "scale": scale,
                        "coeff": coeff,
                        "in_dim": X_pow_std.shape[1],
                    },
                    tmp_path,
                )
                tmp_path.replace(ckpt_path)
                del X_pow_std, X_va_std, state_cpu, b_tr, b_va, res_tr
                gc.collect()
                print(f"[Export tabm_v1 | Plat {plat} | Fold {fold}] saved ({time.perf_counter() - f_t0:.1f}s)", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        plat_mae = float(pd.DataFrame({"flight": flight_arr, "err": np.abs(oof_tabm - y_all)}).groupby("flight", sort=False)["err"].mean().mean())
        summary["platform_mae"][plat] = plat_mae
        print(f"[Export tabm_v1 | Plat {plat}] MAE={plat_mae:.5f} W ({time.perf_counter() - p_t0:.1f}s)", flush=True)
        if verify_mode:
            oof_parts.append(pd.DataFrame({
                "flight": plat_df["flight"], "segment": plat_df["segment"], "t": plat_df["t"],
                "tabm_repro": oof_tabm,
            }))
            truth_parts.append(plat_df[[*KEYS, "power"]].copy())
        del X_all, basis_all, plat_df
        gc.collect()

    if verify_mode:
        tabm_ref = pd.read_csv(CV5_DIR / "oof_tabm_5fold.csv.gz", dtype={"flight": str, "t": str})
        repro = pd.concat(oof_parts, ignore_index=True)
        truth_df = pd.concat(truth_parts, ignore_index=True)
        m = truth_df.merge(repro, on=KEYS, validate="1:1").merge(tabm_ref, on=KEYS, validate="1:1")
        m["err_new"] = np.abs(m["power"] - m["tabm_repro"])
        m["err_ref"] = np.abs(m["power"] - m["tabm_phys"])
        mae_new = float(m.groupby("flight", sort=False)["err_new"].mean().mean())
        mae_ref = float(m.groupby("flight", sort=False)["err_ref"].mean().mean())
        diff = abs(mae_new - mae_ref)
        print(f"Verify tabm_phys: repro={mae_new:.5f} W vs ref={mae_ref:.5f} W (|diff|={diff:.5f} W)", flush=True)
        assert diff < 0.25, f"tabm diff too large: {diff}"
        (MODELS_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Exported all 20 tabm_v1 .pt files in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
