#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


# Assumed Penn 13-joint order mapping to MediaPipe 33 indices.
PENN_TO_MP = {
    0: 0,   # head -> nose
    1: 11,  # left_shoulder
    2: 12,  # right_shoulder
    3: 13,  # left_elbow
    4: 14,  # right_elbow
    5: 15,  # left_wrist
    6: 16,  # right_wrist
    7: 23,  # left_hip
    8: 24,  # right_hip
    9: 25,  # left_knee
    10: 26, # right_knee
    11: 27, # left_ankle
    12: 28, # right_ankle
}


def _to_scalar(x):
    return int(x.item()) if hasattr(x, "item") else int(x)


def _read_penn_label(mat_path: Path):
    d = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    vis = np.asarray(d["visibility"], dtype=np.float32)
    dims = np.asarray(d["dimensions"]).reshape(-1)
    h = float(dims[0]) if len(dims) > 0 else 1.0
    w = float(dims[1]) if len(dims) > 1 else 1.0
    nframes = _to_scalar(d["nframes"])
    return x, y, vis, h, w, nframes


def calibrate_one(mp_npz: Path, mat_path: Path, out_npz: Path, blend_mp: float) -> dict:
    d = np.load(mp_npz, allow_pickle=True)
    keypoints = d["keypoints"].astype(np.float32)  # [T,33,4]
    x, y, vis, h, w, nframes = _read_penn_label(mat_path)

    t_mp = keypoints.shape[0]
    t_gt = x.shape[0]
    t = min(t_mp, t_gt)
    if t <= 0:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **{k: d[k] for k in d.files})
        return {"success": False, "reason": "empty_sequence"}

    corr = keypoints.copy()
    alpha = float(np.clip(blend_mp, 0.0, 1.0))
    beta = 1.0 - alpha

    # Penn coords are pixel coords in [1..W/H], convert to normalized [0..1].
    px = np.clip((x[:t] - 1.0) / max(w - 1.0, 1.0), 0.0, 1.0)
    py = np.clip((y[:t] - 1.0) / max(h - 1.0, 1.0), 0.0, 1.0)
    pv = (vis[:t] > 0).astype(np.float32)

    updated = 0
    for p_idx, mp_idx in PENN_TO_MP.items():
        valid = pv[:, p_idx] > 0
        if not np.any(valid):
            continue
        # blend xy toward Penn for valid frames
        corr[:t, mp_idx, 0][valid] = alpha * corr[:t, mp_idx, 0][valid] + beta * px[:, p_idx][valid]
        corr[:t, mp_idx, 1][valid] = alpha * corr[:t, mp_idx, 1][valid] + beta * py[:, p_idx][valid]
        corr[:t, mp_idx, 3][valid] = np.maximum(corr[:t, mp_idx, 3][valid], pv[:, p_idx][valid])
        updated += int(np.sum(valid))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: d[k] for k in d.files}
    payload["keypoints"] = corr
    payload["calibrated_with_penn"] = np.array([1], dtype=np.int32)
    payload["calibration_blend_mp"] = np.array([alpha], dtype=np.float32)
    payload["penn_label_path"] = np.array([str(mat_path.as_posix())])
    np.savez_compressed(out_npz, **payload)

    return {
        "success": True,
        "t_mp": int(t_mp),
        "t_gt": int(t_gt),
        "t_used": int(t),
        "updated_points": int(updated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MediaPipe keypoints with Penn official joints.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/penn_action/rebuild_dataset/ucf_binary/penn"))
    parser.add_argument("--in-pose-dir", type=Path, default=Path("data/penn_action/pose"))
    parser.add_argument("--out-pose-dir", type=Path, default=Path("data/penn_action/pose_calibrated"))
    parser.add_argument("--penn-labels-dir", type=Path, default=Path("/data1/shiyuqi/Penn_Action/labels"))
    parser.add_argument("--blend-mp", type=float, default=0.2, help="0 means full Penn on mapped joints, 1 means keep MP")
    parser.add_argument("--log-json", type=Path, default=Path("outputs/logs/penn_pose_calibration_log.json"))
    args = parser.parse_args()

    args.out_pose_dir.mkdir(parents=True, exist_ok=True)
    args.log_json.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    for split in ["train", "val", "test"]:
        csv_path = args.split_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"calib-{split}"):
            pose_rel = Path(str(row["pose_path"]))
            in_npz = args.in_pose_dir / pose_rel
            out_npz = args.out_pose_dir / pose_rel
            vid = str(row.get("video_id", Path(str(row["video_path"])).stem))
            mat_path = args.penn_labels_dir / f"{vid}.mat"

            if not in_npz.exists():
                logs.append({"split": split, "video_id": vid, "success": False, "reason": "pose_missing", "pose_path": str(in_npz)})
                continue
            if not mat_path.exists():
                logs.append({"split": split, "video_id": vid, "success": False, "reason": "label_missing", "label_path": str(mat_path)})
                continue

            ret = calibrate_one(in_npz, mat_path, out_npz, blend_mp=float(args.blend_mp))
            ret.update({
                "split": split,
                "video_id": vid,
                "in_pose_path": str(in_npz),
                "out_pose_path": str(out_npz),
                "label_path": str(mat_path),
            })
            logs.append(ret)

    summary = {
        "total": len(logs),
        "success_count": int(sum(1 for x in logs if x.get("success"))),
        "failed_count": int(sum(1 for x in logs if not x.get("success"))),
        "blend_mp": float(args.blend_mp),
        "items": logs,
    }
    args.log_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
