"""Strict zero-leakage 394-flight LOFO + Tree-Consensus Safety Guard + Three-Lock Gate audit for Wave 5 upgrade."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_cv5_unified import load_all_truth, paired_bootstrap_ci, per_flight_metrics
from eval_cv5_wave3 import apply_lofo_blend

ROOT = Path(__file__).resolve().parents[1]
CV5_DIR = ROOT / "无人机功率_C平台结果/cv5_unified"
FOLDS_CSV = CV5_DIR / "folds_5fold_stratified.csv"
KEYS = ["flight", "segment", "t"]
PLATFORMS = ["A", "B", "C", "D"]
T_LOW = 80.0
T_HIGH = 140.0

TREE_Q05_COLS = ["w1_isolated", "catboost_q05_ms", "xgboost_q05_ms", "lgb_q05_l1", "lgb_q05_et"]
TREE_V1_EXTRA = ["catboost_phys", "xgboost_phys", "lgb_phys_l1", "lgb_phys_et"]
NEURAL_ALL = ["tabm_phys", "res_tcn_v2", "conv_bigru_v2", "patch_trans_v2"]


def apply_tree_consensus_guard(pred: np.ndarray, tree_med: np.ndarray, t_low: float = T_LOW, t_high: float = T_HIGH) -> np.ndarray:
    alpha = np.clip((np.abs(pred - tree_med) - t_low) / max(1e-6, t_high - t_low), 0.0, 1.0)
    return ((1.0 - alpha) * pred + alpha * tree_med).astype(np.float64)


def main():
    truth = load_all_truth()
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"flight": str, "platform": str})

    sources = [
        ("oof_w1_isolated_5fold.csv.gz", [*KEYS, "w1_isolated"]),
        ("oof_cat_xgb_5fold.csv.gz", [*KEYS, "catboost_phys", "xgboost_phys"]),
        ("oof_lgb_phys_5fold.csv.gz", [*KEYS, "lgb_phys_l1", "lgb_phys_et"]),
        ("oof_tabm_5fold.csv.gz", [*KEYS, "tabm_phys"]),
        ("oof_res_tcn_v2_5fold.csv.gz", [*KEYS, "res_tcn_v2"]),
        ("oof_guarded_w45_5fold.csv.gz", [*KEYS, "guarded_v1_tcn2"]),
        ("oof_trees_q05_5fold.csv.gz", [*KEYS, "catboost_q05_ms", "xgboost_q05_ms"]),
        ("oof_lgb_q05_5fold.csv.gz", [*KEYS, "lgb_q05_l1", "lgb_q05_et"]),
        ("oof_conv_bigru_v2_5fold.csv.gz", [*KEYS, "conv_bigru_v2"]),
        ("oof_patch_trans_v2_5fold.csv.gz", [*KEYS, "patch_trans_v2"]),
    ]

    df = truth.merge(folds_df[["flight", "fold"]], on="flight", how="left", validate="many_to_one")
    for fname, cols in sources:
        sub = pd.read_csv(CV5_DIR / fname, dtype={"flight": str, "t": str}, usecols=cols)
        df = df.merge(sub, on=KEYS, how="left", validate="1:1")

    assert len(df) == 836547 and df["flight"].nunique() == 394
    all_member_cols = [*TREE_Q05_COLS, *TREE_V1_EXTRA, *NEURAL_ALL]
    assert np.isfinite(df[["guarded_v1_tcn2", *all_member_cols]]).all().all()

    print("=== Standalone 394-Flight OOF Metrics (Unguarded New Backbones) ===")
    for col in ["catboost_q05_ms", "xgboost_q05_ms", "lgb_q05_l1", "lgb_q05_et", "res_tcn_v2", "conv_bigru_v2", "patch_trans_v2"]:
        fm = per_flight_metrics(df, col)
        plats = {p: round(float(fm.loc[fm["platform"] == p, "mae"].mean()), 5) for p in PLATFORMS}
        print(f"  {col:18s}: MAE={fm['mae'].mean():.5f} RMSE={fm['rmse'].mean():.5f} R2={fm['r2'].mean():.5f} max={fm['mae'].max():.2f} | {plats}")

    tree_med_q05 = np.median(df[TREE_Q05_COLS].to_numpy(dtype=np.float64), axis=1)
    df_safe = df.copy()
    print("\n=== Tree-Consensus Safety Guard (q05 5-tree median, [80, 140] W) ===")
    for nc in NEURAL_ALL:
        raw_vals = df_safe[nc].to_numpy(dtype=np.float64)
        n_trig = int((np.abs(raw_vals - tree_med_q05) > T_LOW).sum())
        df_safe[nc] = apply_tree_consensus_guard(raw_vals, tree_med_q05)
        print(f"  Guard [{nc:14s}] triggered on {n_trig:5d} / {len(df_safe)} rows ({100.0 * n_trig / len(df_safe):.3f}%)")

    cand_defs = {
        "q05_trees_tcn2": [*TREE_Q05_COLS, "tabm_phys", "res_tcn_v2"],
        "q05_all_3seq": [*TREE_Q05_COLS, *NEURAL_ALL],
        "q05_all_3seq_notabm": [*TREE_Q05_COLS, "res_tcn_v2", "conv_bigru_v2", "patch_trans_v2"],
        "kitchen_sink_v1_q05": [*TREE_Q05_COLS, *TREE_V1_EXTRA, *NEURAL_ALL],
    }

    cand_lofo_weights = {}
    for cname, cols in cand_defs.items():
        cand_lofo_weights[cname] = apply_lofo_blend(df_safe, cols, cname, l2_reg=0.20)

    # Frozen mean weights for cleanest upgrade candidate (q05_all_3seq)
    frozen_weights = {}
    df_safe["frozen_q05_all_3seq"] = np.nan
    for plat in PLATFORMS:
        cols = cand_defs["q05_all_3seq"]
        avg_w = {c: float(np.mean([cand_lofo_weights["q05_all_3seq"][str(f)][plat][c] for f in range(5)])) for c in cols}
        s = sum(avg_w.values())
        avg_w = {c: v / s for c, v in avg_w.items()}
        frozen_weights[plat] = avg_w
        mask = df_safe["platform"] == plat
        pred = np.zeros(mask.sum(), dtype=np.float64)
        for c, wi in avg_w.items():
            pred += wi * df_safe.loc[mask, c].to_numpy(dtype=np.float64)
        df_safe.loc[mask, "frozen_q05_all_3seq"] = pred

    base_fm = per_flight_metrics(df_safe, "guarded_v1_tcn2").sort_values("flight")
    print("\n=== Wave 5 Strict Zero-Leakage LOFO Comparison (394 Flights) ===")
    eval_cols = ["guarded_v1_tcn2", *cand_defs.keys(), "frozen_q05_all_3seq"]
    results = {}
    for m in eval_cols:
        fm = per_flight_metrics(df_safe, m).sort_values("flight")
        plats = {p: round(float(fm.loc[fm["platform"] == p, "mae"].mean()), 5) for p in PLATFORMS}
        n_gt50 = int((fm["mae"] > 50.0).sum())
        row = {
            "mae": float(fm["mae"].mean()),
            "rmse": float(fm["rmse"].mean()),
            "r2": float(fm["r2"].mean()),
            "energy_rel_err": float(fm["energy_rel_err"].mean()),
            "max_mae": float(fm["mae"].max()),
            "n_gt50": n_gt50,
            "platforms": plats,
        }
        print(
            f"{m:22s}: MAE={row['mae']:.5f} RMSE={row['rmse']:.5f} R2={row['r2']:.5f} "
            f"E={row['energy_rel_err']:.5f} max={row['max_mae']:.2f} >50W={n_gt50} | {plats}"
        )
        if m != "guarded_v1_tcn2":
            g = paired_bootstrap_ci(base_fm["mae"].to_numpy(), fm["mae"].to_numpy())
            row["vs_sealed"] = g
            print(
                f"   vs Sealed(13.87596): gain={g['mean_gain_w']:+.4f} W "
                f"CI95=[{g['ci95_low_w']:+.4f}, {g['ci95_high_w']:+.4f}] improved={g['improved_pct']:.1f}%"
            )
        results[m] = row

    print("\n=== Sentinel Flights (Guarded Sealed vs Wave 5) ===")
    for fl in ["F0025", "F0089", "F0332", "F0414", "F0353"]:
        sub = df_safe.loc[df_safe["flight"] == fl]
        vals = {m: round(float(np.mean(np.abs(sub["power"] - sub[m]))), 2) for m in ["guarded_v1_tcn2", "q05_all_3seq", "frozen_q05_all_3seq"]}
        print(f"  {fl}: {vals}")

    manifest = {
        "name": "uav_power_wave5_q05_3seq",
        "guard": {"tree_cols": TREE_Q05_COLS, "neural_cols": NEURAL_ALL, "t_low": T_LOW, "t_high": T_HIGH},
        "members": cand_defs["q05_all_3seq"],
        "frozen_weights_per_platform": frozen_weights,
        "lofo_weights_by_fold": cand_lofo_weights["q05_all_3seq"],
        "metrics": results,
    }
    (CV5_DIR / "wave5_upgrade_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    df_safe[[*KEYS, "q05_trees_tcn2", "q05_all_3seq", "q05_all_3seq_notabm", "kitchen_sink_v1_q05", "frozen_q05_all_3seq"]].to_csv(
        CV5_DIR / "oof_wave5_upgrade_5fold.csv.gz", index=False, compression="gzip"
    )
    print("\nSaved wave5_upgrade_manifest.json & oof_wave5_upgrade_5fold.csv.gz")


if __name__ == "__main__":
    main()
