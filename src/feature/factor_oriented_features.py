from __future__ import annotations
import numpy as np

def _safe_norm(x, axis=-1):
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=axis)

def _rolling_mean_std(values, window):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = np.zeros(n, dtype=np.float64)
    std = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - int(window) + 1)
        segment = values[start:i + 1]
        mean[i] = float(np.mean(segment))
        std[i] = float(np.std(segment))
    return mean, std

def _difference(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        result[1:] = values[1:] - values[:-1]
    return result

def _matrix_rotation_angle_deg(transforms):
    transforms = np.asarray(transforms, dtype=np.float64)
    result = np.zeros(len(transforms), dtype=np.float64)
    for i, T in enumerate(transforms):
        R = T[:3, :3]
        value = (np.trace(R) - 1.0) * 0.5
        result[i] = np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))
    return result

def build_gps_factor_features(base_features, degraded_gps_local, timestamps, window=10):
    base = np.asarray(base_features, dtype=np.float64)
    position = np.asarray(degraded_gps_local, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    n = min(len(base), len(position), len(timestamps))
    base, position, timestamps = base[:n], position[:n], timestamps[:n]

    dt = np.full(n, 0.1, dtype=np.float64)
    if n > 1:
        dt[1:] = np.diff(timestamps)
        bad = (~np.isfinite(dt)) | (dt <= 0.0) | (dt > 1.0)
        dt[bad] = 0.1

    velocity = np.zeros_like(position)
    if n > 1:
        velocity[1:] = (position[1:] - position[:-1]) / dt[1:, None]

    speed = _safe_norm(velocity)

    acceleration = np.zeros_like(velocity)
    if n > 1:
        acceleration[1:] = (velocity[1:] - velocity[:-1]) / dt[1:, None]
    acc_norm = _safe_norm(acceleration)

    step = np.zeros(n, dtype=np.float64)
    if n > 1:
        step[1:] = _safe_norm(position[1:] - position[:-1])

    speed_mean, speed_std = _rolling_mean_std(speed, window)
    step_mean, step_std = _rolling_mean_std(step, window)

    extras = np.column_stack([
        speed,
        acc_norm,
        step,
        _difference(speed),
        _difference(step),
        speed_mean,
        speed_std,
        step_mean,
        step_std,
    ])
    return np.concatenate([base, extras], axis=1).astype(np.float32)

def build_imu_factor_features(base_features, degraded_imu_acc, degraded_imu_gyro, timestamps, window=10):
    base = np.asarray(base_features, dtype=np.float64)
    acc = np.asarray(degraded_imu_acc, dtype=np.float64)
    gyro = np.asarray(degraded_imu_gyro, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    n = min(len(base), len(acc), len(gyro), len(timestamps))
    base, acc, gyro, timestamps = base[:n], acc[:n], gyro[:n], timestamps[:n]

    dt = np.full(n, 0.1, dtype=np.float64)
    if n > 1:
        dt[1:] = np.diff(timestamps)
        bad = (~np.isfinite(dt)) | (dt <= 0.0) | (dt > 1.0)
        dt[bad] = 0.1

    acc_norm = _safe_norm(acc)
    gyro_norm = _safe_norm(gyro)
    rotation_increment = gyro_norm * dt

    jerk = np.zeros_like(acc)
    if n > 1:
        jerk[1:] = (acc[1:] - acc[:-1]) / dt[1:, None]
    jerk_norm = _safe_norm(jerk)

    gyro_diff = np.zeros_like(gyro)
    if n > 1:
        gyro_diff[1:] = gyro[1:] - gyro[:-1]
    gyro_diff_norm = _safe_norm(gyro_diff)

    acc_mean, acc_std = _rolling_mean_std(acc_norm, window)
    gyro_mean, gyro_std = _rolling_mean_std(gyro_norm, window)

    extras = np.column_stack([
        acc_norm,
        gyro_norm,
        rotation_increment,
        jerk_norm,
        gyro_diff_norm,
        _difference(acc_norm),
        _difference(gyro_norm),
        acc_mean,
        acc_std,
        gyro_mean,
        gyro_std,
    ])
    return np.concatenate([base, extras], axis=1).astype(np.float32)

def _build_relative_factor_features(base_features, factor_npz, window=10):
    base = np.asarray(base_features, dtype=np.float64)
    between = np.asarray(factor_npz["between_measurements"], dtype=np.float64)
    valid = np.asarray(factor_npz["valid"], dtype=bool)
    quality = np.asarray(factor_npz["quality"], dtype=np.float64)

    n = min(len(base), len(between) + 1)
    base = base[:n]

    step_translation = np.zeros(n, dtype=np.float64)
    step_rotation = np.zeros(n, dtype=np.float64)
    valid_frame = np.ones(n, dtype=np.float64)
    quality_frame = np.ones(n, dtype=np.float64)

    step_translation[1:] = _safe_norm(between[:n-1, :3, 3])
    step_rotation[1:] = _matrix_rotation_angle_deg(between[:n-1])
    valid_frame[1:] = valid[:n-1].astype(np.float64)
    quality_frame[1:] = quality[:n-1]

    t_mean, t_std = _rolling_mean_std(step_translation, window)
    r_mean, r_std = _rolling_mean_std(step_rotation, window)
    q_mean, q_std = _rolling_mean_std(quality_frame, window)

    extras = np.column_stack([
        valid_frame,
        quality_frame,
        step_translation,
        step_rotation,
        _difference(quality_frame),
        _difference(step_translation),
        _difference(step_rotation),
        t_mean,
        t_std,
        r_mean,
        r_std,
        q_mean,
        q_std,
    ])
    return np.concatenate([base, extras], axis=1).astype(np.float32)

def build_lidar_factor_features(base_features, factor_npz, window=10):
    return _build_relative_factor_features(base_features, factor_npz, window)

def build_camera_factor_features(base_features, factor_npz, window=10):
    return _build_relative_factor_features(base_features, factor_npz, window)
