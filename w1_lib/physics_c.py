"""C 平台的旋翼盘来流与同片段航迹特征。"""

import numpy as np
import pandas as pd

from features import DT, GRAVITY


DISK_COLUMNS = [
    "disk_axial", "disk_edgewise", "disk_edgewise_sq", "tilt_load", "dynamic_load",
    "disk_induced", "disk_lift_work", "frame_tilt_agreement", "altitude_to_command",
    "ground_to_air_gap", "air_ground_alignment",
]
HISTORY_COLUMNS = [
    "edgewise_mean10", "edgewise_mean40", "axial_mean10", "axial_mean40",
    "vertical_mean10", "vertical_mean40", "load_mean10", "load_mean40",
    "rotor_mean10", "rotor_mean40", "yaw_rms40", "pitch_variation10",
    "roll_variation10", "velocity_variation10", "vertical_variation10",
    "vertical_accel_slow", "vertical_accel_fast", "air_axial_change",
    "asset_wind_mismatch", "ambient_x_mean50", "ambient_y_mean50",
]
ROUTE_COLUMNS = [
    "segment_displacement", "segment_path_length", "range_rate",
    "return_alignment", "relative_height_peak", "height_below_peak",
    "heading_match", "height_rate_long", "recent_wind_exposure",
    "recent_ascended_m", "recent_descended_m",
]
TRANSITION_COLUMNS = [
    "vertical_odometry", "odometry_height_gap", "odometry_drop_from_top",
    "lowest_recent_height", "height_above_recent_floor", "height_quiet_share",
    "recent_motion_share", "stationary_streak", "abs_pitch_mean20", "abs_roll_mean20",
]


def flight_physics(raw, ext):
    by_segment = []
    for _, x in ext.groupby([raw["flight"], raw["segment"]], sort=False):
        pitch = np.deg2rad(x["pitch"].clip(-80, 80))
        roll = np.deg2rad(x["roll"].clip(-80, 80))
        n_forward = -np.sin(pitch) * np.cos(roll)
        n_side = np.sin(roll)
        n_up = (np.cos(pitch) * np.cos(roll)).clip(0.35, 1)
        relative_up = x["v_up"]
        axial = n_forward * x["wind_head"] + n_side * x["wind_right"] + n_up * relative_up
        square = x["wind_speed"].pow(2) + relative_up.pow(2)
        edgewise_sq = (square - axial.pow(2)).clip(lower=0)
        edgewise = np.sqrt(edgewise_sq)
        tilt_load = 1 / n_up
        slow_accel = (relative_up - relative_up.shift(6)) / (6 * DT)
        fast_accel = x["v_up_rate3"]
        vertical_accel = (0.65 * fast_accel + 0.35 * slow_accel).fillna(fast_accel).fillna(0)
        dynamic_load = ((1 + vertical_accel.clip(-6.0, 7.0) / GRAVITY) * tilt_load)
        # Dimensionless hover-load proxy. Fixed velocity scale is not a measurement of C.
        hover_scale = 5.0
        mu_sq = edgewise_sq / hover_scale**2
        hover_ratio = np.sqrt((np.sqrt(mu_sq.pow(2) + 4 * dynamic_load.pow(2)) - mu_sq) / 2)
        vertical_proxy = (axial / hover_scale).clip(-0.5, 1.4)
        induced = dynamic_load * (
            np.sqrt(hover_ratio.pow(2) + 0.25 * vertical_proxy.pow(2)) - 0.5 * vertical_proxy
        )
        lift_work = dynamic_load * axial
        speed = x["horizontal_speed"]
        wind = x["wind_speed"]
        projection = x["v_fwd"] * x["wind_head"] + x["v_right"] * x["wind_right"]
        heading = np.deg2rad(x["yaw_rate"].clip(-220, 220).fillna(0)).cumsum() * DT
        cos_yaw, sin_yaw = np.cos(heading), np.sin(heading)
        vx = cos_yaw * x["v_fwd"] - sin_yaw * x["v_right"]
        vy = sin_yaw * x["v_fwd"] + cos_yaw * x["v_right"]
        dx = vx.fillna(0).shift(1, fill_value=0).cumsum() * DT
        dy = vy.fillna(0).shift(1, fill_value=0).cumsum() * DT
        distance = np.hypot(dx, dy)
        range_rate = (dx * vx + dy * vy) / distance.clip(lower=2)
        wx = cos_yaw * x["ambient_fwd"] - sin_yaw * x["ambient_right"]
        wy = sin_yaw * x["ambient_fwd"] + cos_yaw * x["ambient_right"]
        wx_mean = wx.rolling(50, min_periods=1).mean()
        wy_mean = wy.rolling(50, min_periods=1).mean()
        height = x["h_rel"]
        peak = height.cummax()
        features = {
            "disk_axial": axial, "disk_edgewise": edgewise, "disk_edgewise_sq": edgewise_sq,
            "tilt_load": tilt_load, "dynamic_load": dynamic_load, "disk_induced": induced,
            "disk_lift_work": lift_work,
            "frame_tilt_agreement": (
                x["accel_fwd_rot5"] * n_forward + x["accel_side_rot5"] * n_side
            ),
            "altitude_to_command": (x["cmd_alt"] - height).where(x["cmd_alt"] >= 0),
            "ground_to_air_gap": (speed - wind).abs(),
            "air_ground_alignment": projection / (speed * wind).clip(lower=0.5),
            "yaw_rms40": np.sqrt(x["yaw_rate"].clip(-180, 180).pow(2).rolling(40, min_periods=1).mean()),
            "vertical_accel_slow": slow_accel, "vertical_accel_fast": fast_accel,
            "air_axial_change": axial - axial.shift(5),
            "asset_wind_mismatch": np.hypot(wx - wx_mean, wy - wy_mean),
            "ambient_x_mean50": cos_yaw * wx_mean + sin_yaw * wy_mean,
            "ambient_y_mean50": -sin_yaw * wx_mean + cos_yaw * wy_mean,
            "segment_displacement": distance,
            "segment_path_length": speed.clip(0, 25).shift(1, fill_value=0).cumsum() * DT,
            "range_rate": range_rate,
            "return_alignment": range_rate / speed.clip(lower=1),
            "relative_height_peak": peak,
            "height_below_peak": peak - height,
            "heading_match": cos_yaw,
            "height_rate_long": (height - height.shift(20)) / (20 * DT),
            "recent_wind_exposure": edgewise.pow(2).rolling(80, min_periods=1).mean(),
            "recent_ascended_m": relative_up.clip(lower=0).rolling(45, min_periods=1).sum() * DT,
            "recent_descended_m": -relative_up.clip(upper=0).rolling(45, min_periods=1).sum() * DT,
        }
        for label, series in [
            ("edgewise", edgewise), ("axial", axial), ("vertical", relative_up),
            ("load", dynamic_load), ("rotor", induced + lift_work / hover_scale),
        ]:
            for window in (10, 40):
                features[f"{label}_mean{window}"] = series.rolling(window, min_periods=1).mean()
        for label, series in [
            ("pitch", x["pitch"]), ("roll", x["roll"]), ("velocity", speed), ("vertical", relative_up),
        ]:
            features[f"{label}_variation10"] = series.rolling(10, min_periods=1).std(ddof=0)
        by_segment.append(pd.DataFrame(features, index=x.index, dtype=np.float32))
    return pd.concat(by_segment).reindex(ext.index)[DISK_COLUMNS + HISTORY_COLUMNS + ROUTE_COLUMNS]


def transition_features(raw, ext):
    by_segment = []
    for _, x in ext.groupby([raw["flight"], raw["segment"]], sort=False):
        height = x["h_rel"]
        vertical = x["v_up"]
        integrated = vertical.clip(-8, 8).fillna(0).cumsum() * DT
        odometry = height.iloc[0] + integrated
        recent_floor = height.rolling(40, min_periods=1).min()
        moving = x["horizontal_speed"].gt(0.45) | vertical.abs().gt(0.28)
        steady = ~moving & x["yaw_rate"].abs().lt(5)
        run_start = steady.ne(steady.shift()).cumsum()
        streak = steady.groupby(run_start, sort=False).cumsum()
        height_quiet = (height - height.shift(3)).abs().lt(0.2)
        by_segment.append(pd.DataFrame({
            "vertical_odometry": odometry,
            "odometry_height_gap": height - odometry,
            "odometry_drop_from_top": integrated.cummax() - integrated,
            "lowest_recent_height": recent_floor,
            "height_above_recent_floor": height - recent_floor,
            "height_quiet_share": height_quiet.rolling(16, min_periods=1).mean(),
            "recent_motion_share": moving.rolling(20, min_periods=1).mean(),
            "stationary_streak": streak.clip(0, 120),
            "abs_pitch_mean20": x["pitch"].abs().rolling(20, min_periods=1).mean(),
            "abs_roll_mean20": x["roll"].abs().rolling(20, min_periods=1).mean(),
        }, index=x.index, dtype=np.float32))
    return pd.concat(by_segment).reindex(ext.index)[TRANSITION_COLUMNS]
