"""Train spike-resistant ResTCN_v2 (q05 h_agl + attitude micro-oscillation + 20s heartbeat + CPU ckpts) on Colab T4."""

import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

KEYS = ["flight", "segment", "t"]
PLATFORMS = ["A", "B", "C", "D"]
SEEDS = [42, 1042]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_segment_features_and_basis_v2(seg_df: pd.DataFrame):
    """Return 51 segment features (spike-resistant h_agl + attitude rstd) + 14-term IRLS physics basis."""
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

    # Spike-resistant negative-drift barometer baseline (immune to single-point -62.77m spikes)
    h_base = float(np.clip(np.quantile(h_rel, 0.05), -20.0, 0.0))
    h_agl = np.maximum(0.0, h_rel - h_base).astype(np.float32)
    h_log = np.log1p(h_agl).astype(np.float32)
    h_rel_clip = np.clip(h_rel, -20.0, 200.0).astype(np.float32)

    v_horiz = np.hypot(v_fwd, v_right).astype(np.float32)
    v_total = np.sqrt(v_fwd ** 2 + v_right ** 2 + v_up ** 2).astype(np.float32)
    tilt = np.hypot(pitch, roll).astype(np.float32)

    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    yaw_rad = np.deg2rad(yaw_rate)
    cos_pr = np.maximum(0.5, np.cos(pitch_rad) * np.cos(roll_rad))
    load_factor = (1.0 / cos_pr).astype(np.float32)

    wind_rad = np.deg2rad(wind_dir)
    wind_x = (wind_speed * np.cos(wind_rad)).astype(np.float32)
    wind_y = (wind_speed * np.sin(wind_rad)).astype(np.float32)

    dv_up_1 = cdiff(v_up, 1)
    dv_up_2 = cdiff(v_up, 2)
    dh_1 = cdiff(h_rel, 1)
    dh_2 = cdiff(h_rel, 2)
    dv_fwd_1 = cdiff(v_fwd, 1)
    dv_fwd_2 = cdiff(v_fwd, 2)
    dv_right_1 = cdiff(v_right, 1)
    dv_right_2 = cdiff(v_right, 2)
    dpitch_1 = cdiff(pitch, 1)
    dpitch_2 = cdiff(pitch, 2)
    droll_1 = cdiff(roll, 1)
    droll_2 = cdiff(roll, 2)
    dwx_1 = cdiff(wind_x, 1)
    dwx_2 = cdiff(wind_x, 2)
    dwy_1 = cdiff(wind_y, 1)
    dwy_2 = cdiff(wind_y, 2)

    # Attitude micro-oscillation features within segment
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

    cmd_speed = seg_df["cmd_speed"].to_numpy(dtype=np.float32)
    cmd_alt = seg_df["cmd_alt"].to_numpy(dtype=np.float32)
    speed_varies = seg_df["speed_varies"].to_numpy(dtype=np.float32)
    alt_varies = seg_df["alt_varies"].to_numpy(dtype=np.float32)

    feats = np.stack(
        [
            v_fwd, v_right, v_up, v_horiz, v_total, h_agl, h_rel_clip, h_log,
            pitch, roll, tilt, load_factor, yaw_rate, np.abs(yaw_rate),
            wind_speed, wind_x, wind_y,
            dv_up_1, dv_up_2, dh_1, dh_2, dv_fwd_1, dv_fwd_2, dv_right_1, dv_right_2,
            dpitch_1, dpitch_2, droll_1, droll_2, dwx_1, dwx_2, dwy_1, dwy_2,
            *att_cols,
            dh_dt_minus_vup, spec_fx, spec_fy, load_vup, load_pow15, wind_sq, v_horiz_wind_x,
            payload_level, cmd_speed, cmd_alt, speed_varies, alt_varies,
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


class DilatedResConvBlock(nn.Module):
    def __init__(self, hidden: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.norm = nn.LayerNorm(hidden)
        self.conv_glu = nn.Conv1d(
            hidden, hidden * 2, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        h = self.norm(x) * m
        h = h.transpose(1, 2)
        h = F.glu(self.conv_glu(h), dim=1)
        h = self.drop(h)
        h = self.proj(h).transpose(1, 2)
        return (x + h) * m


class ResPhysTCNFiLM(nn.Module):
    def __init__(self, in_dim: int = 52, hidden: int = 160, nominal_res_std: float = 40.0, dropout: float = 0.15):
        super().__init__()
        self.nominal_res_std = float(max(1.0, nominal_res_std))
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.conv_blocks = nn.ModuleList(
            [
                DilatedResConvBlock(hidden, kernel_size=5, dilation=d, dropout=dropout)
                for d in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.att_score = nn.Linear(hidden, 1)
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden * 2),
        )
        self.refine_blocks = nn.ModuleList(
            [
                DilatedResConvBlock(hidden, kernel_size=3, dilation=d, dropout=dropout)
                for d in (1, 2)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.gate_head = nn.Linear(hidden, 1)
        self.delta_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, y_phys: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1)
        h = self.stem(x) * m
        for blk in self.conv_blocks:
            h = blk(h, mask)

        att_logits = self.att_score(h).squeeze(-1)
        att_logits = att_logits.masked_fill(mask < 0.5, -1e4)
        att_w = torch.softmax(att_logits, dim=1).unsqueeze(-1)
        att_mean = torch.sum(att_w * h, dim=1)
        att_var = torch.sum(att_w * (h - att_mean.unsqueeze(1)) ** 2, dim=1)
        att_std = torch.sqrt(att_var + 1e-5)
        ctx = torch.cat([att_mean, att_std], dim=1)

        gamma, beta = self.film_mlp(ctx).chunk(2, dim=1)
        h = (h * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)) * m
        for blk in self.refine_blocks:
            h = blk(h, mask)
        h = self.out_norm(h) * m

        gate_logit = self.gate_head(h).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        delta = self.delta_head(h).squeeze(-1) * self.nominal_res_std
        pred = gate * F.relu(y_phys + delta)
        return pred, gate_logit, gate


def collate_segments(batch):
    lengths = [len(b["x"]) for b in batch]
    max_len = max(lengths)
    in_dim = batch[0]["x"].shape[1]
    bsz = len(batch)

    x_pad = torch.zeros((bsz, max_len, in_dim), dtype=torch.float32)
    phys_pad = torch.zeros((bsz, max_len), dtype=torch.float32)
    y_pad = torch.zeros((bsz, max_len), dtype=torch.float32)
    w_pad = torch.zeros((bsz, max_len), dtype=torch.float32)
    mask = torch.zeros((bsz, max_len), dtype=torch.float32)

    for i, b in enumerate(batch):
        L = len(b["x"])
        x_pad[i, :L] = torch.from_numpy(b["x"])
        phys_pad[i, :L] = torch.from_numpy(b["y_phys"])
        y_pad[i, :L] = torch.from_numpy(b["y"])
        w_pad[i, :L] = b["flight_weight"]
        mask[i, :L] = 1.0
    return x_pad, phys_pad, y_pad, w_pad, mask


def prepare_segments(df: pd.DataFrame):
    flight_rows = df.groupby("flight", sort=False).size().to_dict()
    segments = []
    for (flight, segment), seg_df in df.groupby(KEYS[:2], sort=False):
        feats, basis = build_segment_features_and_basis_v2(seg_df)
        power = seg_df["power"].to_numpy(dtype=np.float32)
        segments.append(
            {
                "flight": flight,
                "segment": int(segment),
                "t": seg_df["t"].to_numpy(),
                "x_raw": feats,
                "basis": basis,
                "y": power,
                "flight_weight": float(1.0 / flight_rows[flight]),
            }
        )
    return segments


def attach_anchor_and_norm(train_segs, val_segs):
    b_list, y_list, w_list = [], [], []
    for s in train_segs:
        pow_m = s["y"] >= 30.0
        if pow_m.any():
            b_list.append(s["basis"][pow_m])
            y_list.append(s["y"][pow_m])
            w_list.append(np.full(pow_m.sum(), s["flight_weight"], dtype=np.float64))
    b_tr = np.concatenate(b_list, axis=0)
    y_tr = np.concatenate(y_list, axis=0)
    w_tr = np.concatenate(w_list, axis=0)
    w_tr = w_tr / np.mean(w_tr)
    loc, sc_b, coeff = fit_irls_anchor(b_tr, y_tr, w_tr)

    phys_tr_cat = predict_irls_anchor(b_tr, loc, sc_b, coeff)
    nominal_res_std = float(max(1.0, np.std(y_tr - phys_tr_cat)))

    x_tr_aug = []
    for s in train_segs:
        yp = predict_irls_anchor(s["basis"], loc, sc_b, coeff)
        s["y_phys"] = yp
        x_tr_aug.append(np.column_stack([s["x_raw"], yp]).astype(np.float32))
    for s in val_segs:
        yp = predict_irls_anchor(s["basis"], loc, sc_b, coeff)
        s["y_phys"] = yp

    x_cat = np.concatenate(x_tr_aug, axis=0)
    med = np.median(x_cat, axis=0).astype(np.float32)
    iqr = np.percentile(x_cat, 75, axis=0) - np.percentile(x_cat, 25, axis=0)
    std = np.std(x_cat, axis=0)
    scale = np.maximum(np.maximum(iqr / 1.349, std), 1e-3).astype(np.float32)

    def apply(segs):
        out = []
        for s in segs:
            aug = np.column_stack([s["x_raw"], s["y_phys"]]).astype(np.float32)
            item = dict(s)
            item["x"] = np.clip((aug - med) / scale, -10.0, 10.0).astype(np.float32)
            out.append(item)
        return out

    return apply(train_segs), apply(val_segs), nominal_res_std, loc, sc_b, coeff, med, scale


def compute_flight_mae(preds_list, segs):
    flight_abs, flight_cnt = {}, {}
    for pred_arr, s in zip(preds_list, segs):
        f = s["flight"]
        flight_abs[f] = flight_abs.get(f, 0.0) + float(np.abs(pred_arr - s["y"]).sum())
        flight_cnt[f] = flight_cnt.get(f, 0) + len(s["y"])
    return float(np.mean([flight_abs[f] / flight_cnt[f] for f in flight_abs]))


def train_one_seed(
    train_segs, val_segs, in_dim, nominal_res_std,
    seed: int, epochs: int, batch_size: int, lr: float, hidden: int,
    device: torch.device, use_amp: bool,
):
    set_seed(seed)
    model = ResPhysTCNFiLM(in_dim=in_dim, hidden=hidden, nominal_res_std=nominal_res_std, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_train = len(train_segs)
    seg_lengths = np.array([len(s["x"]) for s in train_segs], dtype=np.float32)

    for epoch in range(epochs):
        model.train()
        order = np.argsort(seg_lengths + np.random.uniform(-300.0, 300.0, size=n_train))
        batches = [order[start : start + batch_size] for start in range(0, n_train, batch_size)]
        random.shuffle(batches)
        old_scale = scaler.get_scale() if use_amp else 1.0
        for batch_idx in batches:
            batch = [train_segs[i] for i in batch_idx]
            x, y_phys, y, w, mask = collate_segments(batch)
            x = x.to(device, non_blocking=True)
            y_phys = y_phys.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred, gate_logit, _ = model(x, y_phys, mask)

            pred_f = pred.float()
            gate_logit_f = gate_logit.float()
            w_sum = w.sum() + 1e-8
            l1_loss = (w * torch.abs(pred_f - y)).sum() / w_sum

            if pred_f.size(1) > 1:
                m_diff = mask[:, 1:] * mask[:, :-1]
                w_diff = 0.5 * (w[:, 1:] + w[:, :-1]) * m_diff
                diff_loss = (w_diff * torch.abs((pred_f[:, 1:] - pred_f[:, :-1]) - (y[:, 1:] - y[:, :-1]))).sum() / (w_diff.sum() + 1e-8)
            else:
                diff_loss = torch.zeros((), device=device)

            bce_raw = F.binary_cross_entropy_with_logits(gate_logit_f, (y >= 30.0).float(), reduction="none")
            bce_loss = (w * bce_raw).sum() / w_sum

            loss = l1_loss + 0.25 * diff_loss + 0.15 * bce_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        if (not use_amp) or (scaler.get_scale() >= old_scale):
            scheduler.step()
        if (epoch + 1) in (30, epochs):
            print(f"    seed={seed} epoch={epoch+1}/{epochs} last_loss={loss.item():.4f}", flush=True)

    model.eval()
    val_preds, val_gates = [], []
    with torch.no_grad():
        for start in range(0, len(val_segs), batch_size):
            batch = val_segs[start : start + batch_size]
            x, y_phys, _, _, mask = collate_segments(batch)
            x = x.to(device, non_blocking=True)
            y_phys = y_phys.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred, _, gate = model(x, y_phys, mask)
            pred_np = pred.float().cpu().numpy()
            gate_np = gate.float().cpu().numpy()
            for i, s in enumerate(batch):
                L = len(s["x"])
                val_preds.append(pred_np[i, :L].copy())
                val_gates.append(gate_np[i, :L].copy())
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return val_preds, val_gates, state_cpu


def start_heartbeat(stop_event: threading.Event, t0: float):
    def _run():
        while not stop_event.wait(20.0):
            print(f". [{time.perf_counter() - t0:.0f}s]", flush=True)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/content/uav_train.csv.gz")
    parser.add_argument("--folds", default="/content/folds_5fold_stratified.csv")
    parser.add_argument("--out", default="/content/out_cv5_res_tcn_v2")
    parser.add_argument("--fold-list", default=os.environ.get("FOLD_LIST", "0,1,2,3,4"))
    parser.add_argument("--plat-list", default=os.environ.get("PLAT_LIST", "A,B,C,D"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    args, _ = parser.parse_known_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop_ev = threading.Event()
    start_heartbeat(stop_ev, t0)

    try:
        df = pd.read_csv(args.data, dtype={"flight": str, "t": str, "platform": str})
        folds_df = pd.read_csv(args.folds, dtype={"flight": str, "platform": str})
        assert len(df) == 836547 and len(folds_df) == 394

        plats_to_run = [p.strip().upper() for p in args.plat_list.split(",") if p.strip()]
        segs_by_plat = {p: prepare_segments(df.loc[df["platform"] == p]) for p in plats_to_run}
        folds_to_run = [int(f) for f in args.fold_list.split(",") if f.strip()]

        for fold in folds_to_run:
            oof_path = out_dir / f"fold_{fold}_oof.csv.gz"
            zip_path = out_dir / f"ckpt_fold{fold}.zip"
            if oof_path.exists() and zip_path.exists():
                print(f"=== ResTCN_v2 Fold {fold} ALREADY DONE (skipping) ===", flush=True)
                continue

            f_t0 = time.perf_counter()
            val_ids = set(folds_df.loc[folds_df["fold"] == fold, "flight"])
            train_ids = set(folds_df.loc[folds_df["fold"] != fold, "flight"])
            ckpt_dir = out_dir / f"ckpt_fold{fold}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            fold_frames, plat_maes = [], {}

            for plat in plats_to_run:
                sub_t0 = time.perf_counter()
                all_segs = segs_by_plat[plat]
                tr_raw = [dict(s) for s in all_segs if s["flight"] in train_ids]
                va_raw = [dict(s) for s in all_segs if s["flight"] in val_ids]
                tr_segs, va_segs, nominal_res_std, loc, sc_b, coeff, med, scale = attach_anchor_and_norm(tr_raw, va_raw)
                in_dim = tr_segs[0]["x"].shape[1]

                seed_preds, seed_gates = [], []
                for seed in SEEDS:
                    vp, vg, state_cpu = train_one_seed(
                        tr_segs, va_segs, in_dim, nominal_res_std,
                        seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, hidden=args.hidden, device=device, use_amp=use_amp,
                    )
                    torch.save(
                        {
                            "state_dict": state_cpu,
                            "loc": loc,
                            "sc_b": sc_b,
                            "coeff": coeff,
                            "med": med,
                            "scale": scale,
                            "nominal_res_std": nominal_res_std,
                            "in_dim": in_dim,
                            "hidden": args.hidden,
                        },
                        ckpt_dir / f"fold{fold}_{plat}_seed{seed}.pt",
                    )
                    seed_preds.append(vp)
                    seed_gates.append(vg)

                ens_preds = [np.mean([seed_preds[si][i] for si in range(len(SEEDS))], axis=0) for i in range(len(va_segs))]
                ens_gates = [np.mean([seed_gates[si][i] for si in range(len(SEEDS))], axis=0) for i in range(len(va_segs))]
                plat_mae = compute_flight_mae(ens_preds, va_segs)
                plat_maes[plat] = plat_mae
                print(f"[ResTCN_v2 | Fold {fold} | Plat {plat}] MAE={plat_mae:.5f} W ({time.perf_counter() - sub_t0:.1f}s)", flush=True)

                for s, pa, ga in zip(va_segs, ens_preds, ens_gates):
                    fold_frames.append(pd.DataFrame({
                        "flight": s["flight"], "segment": s["segment"], "t": s["t"],
                        "res_tcn_v2": pa.astype(np.float32), "res_tcn_v2_gate": ga.astype(np.float32),
                    }))
                gc.collect()
                if use_amp:
                    torch.cuda.empty_cache()

            fold_df = pd.concat(fold_frames, ignore_index=True)
            fold_df["t_num"] = fold_df["t"].astype(np.float64)
            fold_df = fold_df.sort_values(["flight", "segment", "t_num"], kind="stable").drop(columns=["t_num"])
            assert np.isfinite(fold_df[["res_tcn_v2", "res_tcn_v2_gate"]]).all().all()

            val_true = df.loc[df["flight"].isin(val_ids), [*KEYS, "power"]]
            m = val_true.merge(fold_df, on=KEYS, how="left", validate="1:1")
            m["abs_err"] = np.abs(m["res_tcn_v2"] - m["power"])
            f_mae = float(m.groupby("flight", sort=False)["abs_err"].mean().mean())
            print(
                f"=== ResTCN_v2 Fold {fold} DONE: MAE={f_mae:.5f} W | "
                + " ".join(f"{p}={v:.4f}" for p, v in plat_maes.items())
                + f" ({time.perf_counter() - f_t0:.1f}s) ===",
                flush=True,
            )
            shutil.make_archive(str(out_dir / f"ckpt_fold{fold}"), "zip", root_dir=ckpt_dir)
            fold_df.to_csv(oof_path, index=False, compression="gzip")

        print(f"Finished ResTCN_v2 folds {folds_to_run} in {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        stop_ev.set()


if __name__ == "__main__":
    main()
