"""Train Dual-LightGBM q05 (GBDT L1 + ExtraTrees L1) with spike-resistant h_agl across 5 unified CV folds on local CPU."""

import gc
import json
from pathlib import Path
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from local_cv5_trees_q05 import build_phys_bidir_features_q05, fit_irls_anchor, predict_irls_anchor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
CV5_DIR = ROOT / "无人机功率_C平台结果/cv5_unified"
DATA_GZ = ROOT / "无人机功率_C平台结果/colab_seq_film/uav_train.csv.gz"
FOLDS_CSV = CV5_DIR / "folds_5fold_stratified.csv"
MODELS_DIR = CV5_DIR / "lgb_q05_models"
KEYS = ["flight", "segment", "t"]
PLATFORMS = ["A", "B", "C", "D"]


def main():
    t0 = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_GZ, dtype={"flight": str, "t": str, "platform": str})
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"flight": str, "platform": str})
    assert len(df) == 836547 and len(folds_df) == 394
    flight_to_fold = dict(zip(folds_df["flight"], folds_df["fold"]))

    oof_parts = []
    summary = {"platform_mae": {}}

    for plat in PLATFORMS:
        p_t0 = time.perf_counter()
        plat_df = df.loc[df["platform"] == plat].reset_index(drop=True)
        X_reg, X_gate, basis_all = build_phys_bidir_features_q05(plat_df)
        y_all = plat_df["power"].to_numpy(dtype=np.float32)
        f_all = plat_df["flight"].map(flight_to_fold).to_numpy(dtype=np.int32)
        flight_arr = plat_df["flight"].to_numpy()
        _, inv, counts = np.unique(flight_arr, return_inverse=True, return_counts=True)
        w_raw = (1.0 / counts[inv]).astype(np.float64)
        w_all = (w_raw / np.mean(w_raw)).astype(np.float32)

        oof_l1 = np.zeros(len(plat_df), dtype=np.float32)
        oof_et = np.zeros(len(plat_df), dtype=np.float32)
        oof_gate = np.zeros(len(plat_df), dtype=np.float32)

        for fold in range(5):
            val_mask = f_all == fold
            tr_mask = ~val_mask
            Xg_tr, Xg_va = X_gate[tr_mask], X_gate[val_mask]
            Xr_tr, Xr_va = X_reg[tr_mask], X_reg[val_mask]
            y_tr = y_all[tr_mask]
            w_tr = (w_all[tr_mask] / np.mean(w_all[tr_mask])).astype(np.float32)
            b_tr, b_va = basis_all[tr_mask], basis_all[val_mask]

            gate_clf = lgb.LGBMClassifier(
                n_estimators=250, num_leaves=31, learning_rate=0.05,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=42 + fold, n_jobs=12, verbose=-1,
            )
            gate_clf.fit(Xg_tr, (y_tr >= 30.0).astype(np.int32), sample_weight=w_tr)
            g_prob = gate_clf.predict_proba(Xg_va)[:, 1].astype(np.float32)
            g_clean = np.where(g_prob < 0.02, 0.0, g_prob)

            pow_tr = y_tr >= 30.0
            w_pow = (w_tr[pow_tr] / np.mean(w_tr[pow_tr])).astype(np.float32)
            loc, scale, coeff = fit_irls_anchor(b_tr[pow_tr], y_tr[pow_tr], w_pow)
            phys_tr = predict_irls_anchor(b_tr[pow_tr], loc, scale, coeff)
            phys_va = predict_irls_anchor(b_va, loc, scale, coeff)
            res_tr = (y_tr[pow_tr] - phys_tr).astype(np.float32)

            reg_l1 = lgb.LGBMRegressor(
                objective="regression_l1", n_estimators=1200, num_leaves=63,
                learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                min_child_samples=40, reg_lambda=5.0, random_state=42 + fold, n_jobs=12, verbose=-1,
            )
            reg_l1.fit(Xr_tr[pow_tr], res_tr, sample_weight=w_pow)
            oof_l1[val_mask] = g_clean * np.clip(phys_va + reg_l1.predict(Xr_va).astype(np.float32), 0.0, None)

            reg_et = lgb.LGBMRegressor(
                objective="regression_l1", extra_trees=True, n_estimators=1400, num_leaves=95,
                learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                min_child_samples=30, reg_lambda=5.0, random_state=1042 + fold, n_jobs=12, verbose=-1,
            )
            reg_et.fit(Xr_tr[pow_tr], res_tr, sample_weight=w_pow)
            oof_et[val_mask] = g_clean * np.clip(phys_va + reg_et.predict(Xr_va).astype(np.float32), 0.0, None)
            oof_gate[val_mask] = g_clean

            pkl_path = MODELS_DIR / f"fold{fold}_{plat}.pkl"
            tmp_pkl = pkl_path.with_suffix(".pkl.tmp")
            joblib.dump(
                {"gate": gate_clf, "loc": loc, "scale": scale, "coeff": coeff, "reg_l1": reg_l1, "reg_et": reg_et},
                tmp_pkl,
            )
            tmp_pkl.replace(pkl_path)

        l1_mae = float(pd.DataFrame({"flight": flight_arr, "err": np.abs(oof_l1 - y_all)}).groupby("flight", sort=False)["err"].mean().mean())
        et_mae = float(pd.DataFrame({"flight": flight_arr, "err": np.abs(oof_et - y_all)}).groupby("flight", sort=False)["err"].mean().mean())
        summary["platform_mae"][plat] = {"lgb_q05_l1": l1_mae, "lgb_q05_et": et_mae}
        print(f"[Local LGB q05 CPU | Plat {plat}] L1={l1_mae:.5f} W | ET={et_mae:.5f} W ({time.perf_counter() - p_t0:.1f}s)", flush=True)

        oof_parts.append(pd.DataFrame({
            "flight": plat_df["flight"], "segment": plat_df["segment"], "t": plat_df["t"],
            "lgb_q05_l1": oof_l1, "lgb_q05_et": oof_et, "gate_lgb_q05": oof_gate,
        }))
        del X_reg, X_gate, basis_all, plat_df
        gc.collect()

    oof = pd.concat(oof_parts, ignore_index=True)
    oof["t_num"] = oof["t"].astype(np.float64)
    oof = oof.sort_values(["flight", "segment", "t_num"], kind="stable").drop(columns=["t_num"]).reset_index(drop=True)
    assert len(oof) == 836547 and oof["flight"].nunique() == 394
    assert np.isfinite(oof[["lgb_q05_l1", "lgb_q05_et", "gate_lgb_q05"]]).all().all()
    out_gz = CV5_DIR / "oof_lgb_q05_5fold.csv.gz"
    oof.to_csv(out_gz, index=False, compression="gzip")

    m = df[[*KEYS, "power"]].merge(oof, on=KEYS, validate="1:1")
    summary["overall"] = {}
    summary["sentinels"] = {}
    for col in ("lgb_q05_l1", "lgb_q05_et"):
        m[f"err_{col}"] = np.abs(m["power"] - m[col])
        summary["overall"][col] = float(m.groupby("flight", sort=False)[f"err_{col}"].mean().mean())
        summary["sentinels"][col] = {
            fl: float(m.loc[m["flight"] == fl, f"err_{col}"].mean())
            for fl in ("F0025", "F0089", "F0332", "F0414", "F0353")
        }
    summary["elapsed_s"] = time.perf_counter() - t0
    (CV5_DIR / "lgb_q05_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("LGB q05 ALL DONE:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
