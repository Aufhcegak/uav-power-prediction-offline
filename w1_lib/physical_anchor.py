"""Robust power anchor shared by research and offline inference."""

import numpy as np

ANCHOR_COLUMNS = [
    "payload_level", "disk_induced", "disk_lift_work", "disk_edgewise_sq",
    "edgewise_mean40", "rotor_mean10", "yaw_rms40",
]

def anchor_basis(x):
    return x[ANCHOR_COLUMNS].fillna(0).to_numpy(dtype=np.float64)


def fit_anchor(x, y, w):
    matrix = anchor_basis(x)
    mean, std = matrix.mean(axis=0), np.maximum(1e-3, matrix.std(axis=0))
    predictors = np.column_stack([np.ones(len(y)), np.clip((matrix - mean) / std, -6, 6)])
    ridge = np.diag([0, *[0.01 * w.sum()] * matrix.shape[1]])
    effective = w
    for _ in range(5):
        beta = np.linalg.solve(
            predictors.T @ (predictors * effective[:, None]) + ridge,
            predictors.T @ (effective * y),
        )
        effective = w * (22 / np.maximum(22, np.abs(y - predictors @ beta)))
    return mean, std, beta


def anchor_value(x, fit):
    mean, std, beta = fit
    return beta[0] + np.clip((anchor_basis(x) - mean) / std, -6, 6) @ beta[1:]
