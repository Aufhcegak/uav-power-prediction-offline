"""Export missing own_gate, phase, and local physical bundles for W1 platform-isolated experts across 5 folds."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from isolated_w1_experts import (
    COMMON, CONFIG, DATA, KEYS, PLATFORMS, fit_one, predict_one, read_source,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "无人机功率_C平台结果/cv5_unified"
FOLDS_CSV = OUT_DIR / "folds_5fold_stratified.csv"


def main():
    t0 = time.perf_counter()
    COMMON["num_threads"] = 12
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"flight": str, "platform": str})
    ref = pd.read_csv(OUT_DIR / "oof_w1_isolated_5fold.csv.gz", dtype={"flight": str, "t": str})
    parts = []

    for fold in range(5):
        for plat in PLATFORMS:
            train_ids = sorted(
                folds_df.loc[(folds_df["fold"] != fold) & (folds_df["platform"] == plat), "flight"].tolist()
            )
            val_ids = sorted(
                folds_df.loc[(folds_df["fold"] == fold) & (folds_df["platform"] == plat), "flight"].tolist()
            )
            model_dir = OUT_DIR / "w1_models" / f"fold{fold}" / plat
            model_dir.mkdir(parents=True, exist_ok=True)
            own_gate, phase_dict, (phys_tree, loc, scale, coeff) = fit_one(model_dir, plat, train_ids, config)

            # Save pure-text LightGBM boosters + JSON anchor (immune to Chinese path issues)
            (model_dir / "own_gate.txt").write_text(own_gate.model_to_string(), encoding="utf-8")
            (model_dir / "phys_tree.txt").write_text(phys_tree.model_to_string(), encoding="utf-8")
            phase_files = {}
            for k, booster in phase_dict.items():
                fname = f"phase_{k}.txt"
                (model_dir / fname).write_text(booster.model_to_string(), encoding="utf-8")
                phase_files[k] = fname
            (model_dir / "bundle_meta.json").write_text(
                json.dumps(
                    {
                        "phase_files": phase_files,
                        "loc": loc.tolist(),
                        "scale": scale.tolist(),
                        "coeff": coeff.tolist(),
                    }
                ),
                encoding="utf-8",
            )

            actual = read_source(DATA, val_ids)
            pred = predict_one(actual.drop(columns="power"), plat, model_dir, (own_gate, phase_dict, (phys_tree, loc, scale, coeff)), own_gate, config)
            parts.append(pred[[*KEYS, "isolated_full"]].rename(columns={"isolated_full": "w1_repro"}))
        print(f"[Export W1 bundles | Fold {fold}] done ({time.perf_counter() - t0:.1f}s)", flush=True)

    repro = pd.concat(parts, ignore_index=True)
    m = ref.merge(repro, on=KEYS, how="inner", validate="1:1")
    max_diff = float(np.max(np.abs(m["w1_isolated"] - m["w1_repro"])))
    print(f"Verify W1 exported bundles: max |repro - ref| = {max_diff:.6f} W (elapsed {time.perf_counter() - t0:.1f}s)")
    assert max_diff < 1e-3


if __name__ == "__main__":
    main()
