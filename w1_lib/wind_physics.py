"""Airflow and aerodynamic force features for the current flight segment."""

import numpy as np
import pandas as pd

from features import DT, GRAVITY


def wind_features(raw: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, x in ext.groupby([raw["flight"], raw["segment"]], sort=False):
        yaw = np.deg2rad(x["yaw_rate"].fillna(0).clip(-220, 220)).cumsum() * DT
        cs, sn = np.cos(yaw), np.sin(yaw)
        wind_x = cs * x["ambient_fwd"] - sn * x["ambient_right"]
        wind_y = sn * x["ambient_fwd"] + cs * x["ambient_right"]
        reliability = (
            (1 / (1 + x["yaw_abs"] / 15))
            * (0.25 + 0.75 * x["h_rel"].clip(0, 10) / 10)
        ).where(wind_x.notna() & wind_y.notna(), 0.)
        smoothed = {}
        for span in (13, 65, 150):
            mass = reliability.ewm(span=span, adjust=False).mean().replace(0, np.nan)
            wx = (wind_x * reliability).fillna(0).ewm(span=span, adjust=False).mean() / mass
            wy = (wind_y * reliability).fillna(0).ewm(span=span, adjust=False).mean() / mass
            smoothed[span] = (cs * wx + sn * wy, -sn * wx + cs * wy)
        ambient_fwd, ambient_side = smoothed[65]
        air_fwd = x["v_fwd"] - ambient_fwd
        air_side = x["v_right"] - ambient_side
        air_norm = np.hypot(air_fwd, air_side)
        gust_fwd = x["wind_head"] - air_fwd
        gust_side = x["wind_right"] - air_side
        pitch = np.deg2rad(x["pitch"].clip(-80, 80))
        roll = np.deg2rad(x["roll"].clip(-80, 80))
        nx = -np.sin(pitch) * np.cos(roll)
        ny = np.sin(roll)
        nz = np.cos(pitch) * np.cos(roll)
        disk_axial = nx * air_fwd + ny * air_side + nz * x["v_up"]
        disk_plane = np.sqrt((air_norm.pow(2) + x["v_up"].pow(2) - disk_axial.pow(2)).clip(lower=0))
        side_drag = air_norm * air_side
        climb_velocity = x["v_up"]
        fast_load = x["induced_proxy"].ewm(span=7, adjust=False).mean()
        slow_load = x["induced_proxy"].ewm(span=45, adjust=False).mean()
        new = {
            "air_side_steady": air_side,
            "air_forward_steady": air_fwd,
            "gust_side": gust_side,
            "gust_forward": gust_fwd,
            "gust_energy": (gust_fwd.pow(2) + gust_side.pow(2)).rolling(15, min_periods=1).mean(),
            "crosswind_drag": side_drag,
            "drag_power_forward": air_norm * air_fwd.pow(2),
            "drag_power_side": air_norm * air_side.pow(2),
            "flow_disk_axial": disk_axial,
            "flow_disk_plane": disk_plane,
            "air_change_along_track": (
                gust_fwd * x["v_fwd"] + gust_side * x["v_right"]
            ) / x["horizontal_speed"].clip(lower=0.75),
            "side_controller_load": ny - 0.008 * side_drag,
            "forward_controller_load": nx - 0.008 * air_norm * air_fwd,
            "ambient_long_side": smoothed[150][1],
            "ambient_long_fwd": smoothed[150][0],
            "ambient_side_change": smoothed[13][1] - smoothed[150][1],
            "ambient_fwd_change": smoothed[13][0] - smoothed[150][0],
            "flow_confidence": reliability.ewm(span=65, adjust=False).mean(),
            "load_change": fast_load - slow_load,
            "filtered_axial": disk_axial.ewm(span=12, adjust=False).mean(),
            "filtered_plane": disk_plane.ewm(span=12, adjust=False).mean(),
            "vertical_control_change": climb_velocity - climb_velocity.ewm(span=18, adjust=False).mean(),
        }
        parts.append(pd.DataFrame(new, dtype=np.float32, index=x.index))
    return pd.concat(parts).reindex(ext.index)


def force_features(raw: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Combine body tilt and acceleration into a drag-per-mass proxy."""
    parts = []
    for _, x in ext.groupby([raw["flight"], raw["segment"]], sort=False):
        pitch = np.deg2rad(x["pitch"].clip(-70, 70))
        roll = np.deg2rad(x["roll"].clip(-70, 70))
        nz = (np.cos(pitch) * np.cos(roll)).clip(0.35, 1)
        thrust_per_mass = (GRAVITY + x["vertical_accel_slow"].clip(-6, 7)) / nz
        drag_fwd = -thrust_per_mass * np.sin(pitch) * np.cos(roll) - x["accel_fwd_rot5"]
        drag_side = thrust_per_mass * np.sin(roll) - x["accel_side_rot5"]
        yaw = np.deg2rad(x["yaw_rate"].fillna(0).clip(-220, 220)).cumsum() * DT
        cs, sn = np.cos(yaw), np.sin(yaw)
        dx = (cs * drag_fwd - sn * drag_side).ewm(span=16, adjust=False).mean()
        dy = (sn * drag_fwd + cs * drag_side).ewm(span=16, adjust=False).mean()
        mean_fwd, mean_side = cs * dx + sn * dy, -sn * dx + cs * dy
        force_size = np.hypot(mean_fwd, mean_side)
        parts.append(pd.DataFrame({
            "dynamic_drag_fwd": drag_fwd,
            "dynamic_drag_side": drag_side,
            "drag_per_mass": force_size,
            "drag_work_steady_air": mean_fwd * x["air_forward_steady"] + mean_side * x["air_side_steady"],
            "drag_work_current_air": mean_fwd * x["wind_head"] + mean_side * x["wind_right"],
            "drag_side_change": drag_side - mean_side,
            "drag_fwd_change": drag_fwd - mean_fwd,
            "drag_side_relative": mean_side / force_size.clip(lower=0.15),
            "airflow_fwd_proxy": mean_fwd / np.sqrt(force_size.clip(lower=0.15)),
            "airflow_side_proxy": mean_side / np.sqrt(force_size.clip(lower=0.15)),
        }, dtype=np.float32, index=x.index))
    return pd.concat(parts).reindex(ext.index)
