from __future__ import annotations

import numpy as np


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def fill_nan(keypoints: np.ndarray) -> np.ndarray:
    x = keypoints.astype(np.float32).copy()
    if x.ndim != 3:
        raise ValueError("keypoints must be [T, J, C]")
    t, j, c = x.shape
    for jj in range(j):
        for cc in range(c):
            col = x[:, jj, cc]
            mask = np.isnan(col)
            if not np.any(mask):
                continue
            valid = np.where(~mask)[0]
            if valid.size == 0:
                x[:, jj, cc] = 0.0
                continue
            x[: valid[0], jj, cc] = col[valid[0]]
            x[valid[-1] + 1 :, jj, cc] = col[valid[-1]]
            miss = np.where(mask & (np.arange(t) >= valid[0]) & (np.arange(t) <= valid[-1]))[0]
            if miss.size > 0:
                x[miss, jj, cc] = np.interp(miss, valid, col[valid])
    return x


def smooth_keypoints(keypoints: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return keypoints.astype(np.float32)
    x = keypoints.astype(np.float32)
    pad = kernel_size // 2
    w = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    out = np.empty_like(x)
    for jj in range(x.shape[1]):
        for cc in range(x.shape[2]):
            s = np.pad(x[:, jj, cc], (pad, pad), mode="edge")
            out[:, jj, cc] = np.convolve(s, w, mode="valid")
    return out


def _torso_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shoulder_mid = 0.5 * (x[:, LEFT_SHOULDER, :3] + x[:, RIGHT_SHOULDER, :3])
    hip_mid = 0.5 * (x[:, LEFT_HIP, :3] + x[:, RIGHT_HIP, :3])
    center = 0.5 * (shoulder_mid + hip_mid)
    scale = np.linalg.norm(shoulder_mid - hip_mid, axis=1, keepdims=True)
    scale = np.clip(scale, 1e-4, None)
    return center, scale


def normalize_keypoints(keypoints: np.ndarray, align_orientation: bool = True) -> np.ndarray:
    x = keypoints.astype(np.float32).copy()
    if x.shape[-1] < 3:
        raise ValueError("need at least x,y,z channels")
    center, scale = _torso_center_scale(x)
    x[:, :, :3] = (x[:, :, :3] - center[:, None, :]) / scale[:, None, :]
    if align_orientation:
        lr = x[:, RIGHT_SHOULDER, 0] - x[:, LEFT_SHOULDER, 0]
        flip_mask = lr < 0
        if np.any(flip_mask):
            x[flip_mask, :, 0] *= -1.0
            if x.shape[-1] > 2:
                x[flip_mask, :, 2] *= -1.0
    if x.shape[-1] > 3:
        x[:, :, 3:] = np.nan_to_num(x[:, :, 3:], nan=0.0)
    return x


def preprocess_keypoints(keypoints: np.ndarray, smooth_kernel: int = 5) -> np.ndarray:
    x = fill_nan(keypoints)
    x = smooth_keypoints(x, kernel_size=smooth_kernel)
    x = normalize_keypoints(x)
    return np.nan_to_num(x, nan=0.0)
