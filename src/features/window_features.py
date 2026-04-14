from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_preprocess import preprocess_keypoints

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
NOSE = 0


@dataclass
class WindowBatch:
    features: np.ndarray
    labels: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    den = np.clip(den, 1e-6, None)
    cosv = np.sum(ba * bc, axis=1) / den
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def _signals(seq: np.ndarray) -> dict[str, np.ndarray]:
    xyz = seq[:, :, :3]
    s = {}
    s["elbow_l"] = _angle(xyz[:, LEFT_SHOULDER], xyz[:, LEFT_ELBOW], xyz[:, LEFT_WRIST])
    s["elbow_r"] = _angle(xyz[:, RIGHT_SHOULDER], xyz[:, RIGHT_ELBOW], xyz[:, RIGHT_WRIST])
    s["arm_span"] = np.linalg.norm(xyz[:, LEFT_WRIST, :2] - xyz[:, RIGHT_WRIST, :2], axis=1)
    s["ankle_span"] = np.linalg.norm(xyz[:, LEFT_ANKLE, :2] - xyz[:, RIGHT_ANKLE, :2], axis=1)
    s["wrist_height"] = -0.5 * (xyz[:, LEFT_WRIST, 1] + xyz[:, RIGHT_WRIST, 1])
    s["hip_height"] = -0.5 * (xyz[:, LEFT_HIP, 1] + xyz[:, RIGHT_HIP, 1])
    s["torso_tilt"] = _angle(
        xyz[:, NOSE],
        0.5 * (xyz[:, LEFT_SHOULDER] + xyz[:, RIGHT_SHOULDER]),
        0.5 * (xyz[:, LEFT_HIP] + xyz[:, RIGHT_HIP]),
    )
    motion = np.diff(xyz[:, :, :2], axis=0)
    s["motion_energy"] = np.concatenate([[0.0], np.mean(np.linalg.norm(motion, axis=2), axis=1)])
    return s


def _periodic_stats(sig: np.ndarray, fps: float) -> tuple[float, float, float]:
    x = sig - np.mean(sig)
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    ac[0] = 0.0
    lag = int(np.argmax(ac[: max(2, n // 2)]))
    ac_peak = float(ac[lag]) if lag > 0 else 0.0
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=max(1e-6, 1.0 / max(fps, 1e-3)))
    if len(freqs) > 1:
        idx = int(np.argmax(np.abs(fft[1:])) + 1)
        dom_f = float(freqs[idx])
    else:
        dom_f = 0.0
    return float(lag), ac_peak, dom_f


def extract_window_feature_dict(window: np.ndarray, fps: float) -> dict[str, float]:
    sigs = _signals(window)
    out: dict[str, float] = {}
    for name, sig in sigs.items():
        d1 = np.diff(sig) if len(sig) > 1 else np.array([0.0], dtype=np.float32)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0], dtype=np.float32)
        out[f"{name}_mean"] = float(np.mean(sig))
        out[f"{name}_std"] = float(np.std(sig))
        out[f"{name}_min"] = float(np.min(sig))
        out[f"{name}_max"] = float(np.max(sig))
        out[f"{name}_range"] = float(np.max(sig) - np.min(sig))
        out[f"{name}_slope"] = float(sig[-1] - sig[0])
        out[f"{name}_vel_mean"] = float(np.mean(np.abs(d1)))
        out[f"{name}_acc_mean"] = float(np.mean(np.abs(d2)))
        lag, ac_peak, dom_f = _periodic_stats(sig, fps=fps)
        out[f"{name}_ac_lag"] = lag
        out[f"{name}_ac_peak"] = ac_peak
        out[f"{name}_dom_freq"] = dom_f
    return out


def extract_window_feature_vector(window: np.ndarray, fps: float) -> tuple[np.ndarray, list[str]]:
    d = extract_window_feature_dict(window, fps)
    names = sorted(d.keys())
    return np.array([d[k] for k in names], dtype=np.float32), names


def build_video_windows(
    keypoints: np.ndarray,
    fps: float,
    window_size: int,
    stride: int,
    action_label: int,
    background_quantile: float = 0.2,
) -> WindowBatch:
    seq = preprocess_keypoints(keypoints)
    t = seq.shape[0]
    if t < window_size:
        return WindowBatch(
            features=np.zeros((0, 1), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            starts=np.zeros((0,), dtype=np.int64),
            ends=np.zeros((0,), dtype=np.int64),
        )

    feats = []
    names = None
    motion_scores = []
    spans = []
    for s in range(0, t - window_size + 1, stride):
        e = s + window_size
        w = seq[s:e]
        vec, nms = extract_window_feature_vector(w, fps=fps)
        if names is None:
            names = nms
        feats.append(vec)
        sig = _signals(w)["motion_energy"]
        motion_scores.append(float(np.mean(sig)))
        spans.append((s, e))

    x = np.stack(feats).astype(np.float32)
    motion_scores = np.array(motion_scores, dtype=np.float32)
    th = float(np.quantile(motion_scores, background_quantile))
    y = np.where(motion_scores <= th, 0, action_label).astype(np.int64)
    starts = np.array([p[0] for p in spans], dtype=np.int64)
    ends = np.array([p[1] for p in spans], dtype=np.int64)
    return WindowBatch(features=x, labels=y, starts=starts, ends=ends)
