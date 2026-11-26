"""仅依赖当前片段历史的风、运动和状态特征。"""

import numpy as np
import pandas as pd

from features import build_features, DT


def extended_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = build_features(frame)
    added = []
    for _, block in base.groupby([frame["flight"], frame["segment"]], sort=False):
        yaw = np.deg2rad(block["yaw_rate"].clip(-220, 220))
        turn = yaw.cumsum() * DT
        forward = block["v_fwd"]
        right = block["v_right"]
        additions = {}
        for lag in (2, 5):
            delta = turn - turn.shift(lag)
            cosine, sine = np.cos(delta), np.sin(delta)
            forward_old = forward.shift(lag) * cosine + right.shift(lag) * sine
            right_old = -forward.shift(lag) * sine + right.shift(lag) * cosine
            accel_fwd = (forward - forward_old) / (lag * DT)
            accel_side = (right - right_old) / (lag * DT)
            magnitude = np.hypot(accel_fwd, accel_side)
            name = str(lag)
            additions[f"accel_fwd_rot{name}"] = accel_fwd
            additions[f"accel_side_rot{name}"] = accel_side
            additions[f"accel_horizontal_rot{name}"] = magnitude
            additions[f"specific_horizontal_work{name}"] = (
                forward * accel_fwd + right * accel_side
            )
            ambient_fwd = block["ambient_fwd"].shift(lag)
            ambient_right = block["ambient_right"].shift(lag)
            additions[f"ambient_fwd_rot_diff{name}"] = (
                block["ambient_fwd"] - ambient_fwd * cosine - ambient_right * sine
            )
            additions[f"ambient_right_rot_diff{name}"] = (
                block["ambient_right"] + ambient_fwd * sine - ambient_right * cosine
            )
        for column in (
            "v_up", "horizontal_speed", "wind_speed", "pitch", "roll", "induced_proxy",
        ):
            signal = block[column]
            ema = signal.ewm(span=55, adjust=False).mean()
            additions[f"{column}_ema55"] = ema
            additions[f"{column}_highpass55"] = signal - ema
            additions[f"{column}_lag2"] = signal.shift(2)
            additions[f"{column}_lag5"] = signal.shift(5)
        vert = block["v_up"]
        additions["vertical_jerk"] = (vert - 2 * vert.shift(2) + vert.shift(4)) / (2 * DT)**2
        additions["vertical_disagree"] = block["h_rel_rate3"] - block["v_up_mean4"]
        additions["airspeed_vertical"] = np.hypot(block["wind_speed"], vert)
        additions["ambient_magnitude"] = np.hypot(block["ambient_fwd"], block["ambient_right"])
        additions["wind_tilt_load"] = block["wind_speed"] * block["tilt"]
        additions["prefer_vertical"] = vert.ewm(span=4, adjust=False).mean()
        added.append(pd.DataFrame(additions, index=block.index, dtype=np.float32))
    extra = pd.concat(added).reindex(base.index)
    result = pd.concat([base, extra], axis=1).replace([np.inf, -np.inf], np.nan)
    return base, result


def phase_weights(x: pd.DataFrame) -> dict[str, np.ndarray]:
    vertical = x["prefer_vertical"].fillna(0).to_numpy()
    climb = np.clip((vertical - 0.18) / 0.85, 0, 1)
    descent = np.clip((-vertical - 0.18) / 0.85, 0, 1)
    return {"climb": climb, "level": 1 - climb - descent, "descent": descent}
