#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


def _to_int(x) -> int:
    return int(x.item()) if hasattr(x, "item") else int(x)


def align_one(in_npz: Path, label_mat: Path, out_npz: Path) -> dict:
    d = np.load(in_npz, allow_pickle=True)
    keypoints = d["keypoints"].astype(np.float32)  # [T,33,4]

    ann = sio.loadmat(label_mat, squeeze_me=True, struct_as_record=False)
    bbox = np.asarray(ann["bbox"], dtype=np.float32)  # [T,4], x1,y1,x2,y2
    dims = np.asarray(ann["dimensions"]).reshape(-1)
    h = float(dims[0]) if len(dims) > 0 else 1.0
    w = float(dims[1]) if len(dims) > 1 else 1.0
    nframes = _to_int(ann["nframes"])

    t = min(keypoints.shape[0], bbox.shape[0], nframes)
    if t <= 0:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **{k: d[k] for k in d.files})
        return {"success": False, "reason": "empty_seq"}

    out_k = keypoints.copy()

    # Convert full-image normalized xy -> pixel -> ROI-normalized xy per frame
    for i in range(t):
        x1, y1, x2, y2 = bbox[i]
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))

        x_norm = out_k[i, :, 0]
        y_norm = out_k[i, :, 1]

        # Penn annotations are 1-based pixels; keep same convention here
        x_pix = x_norm * max(w - 1.0, 1.0) + 1.0
        y_pix = y_norm * max(h - 1.0, 1.0) + 1.0

        xr = (x_pix - x1) / bw
        yr = (y_pix - y1) / bh

        out_k[i, :, 0] = np.clip(xr, 0.0, 1.0)
        out_k[i, :, 1] = np.clip(yr, 0.0, 1.0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: d[k] for k in d.files}
    payload["keypoints"] = out_k
    payload["roi_aligned_with_penn_bbox"] = np.array([1], dtype=np.int32)
    payload["penn_bbox_path"] = np.array([str(label_mat.as_posix())])
    np.savez_compressed(out_npz, **payload)

    return {"success": True, "t_used": int(t)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Align pose keypoints to Penn ROI bbox coordinates.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/penn_action/rebuild_dataset/ucf_binary/penn"))
    parser.add_argument("--in-pose-dir", type=Path, default=Path("data/penn_action/pose_calibrated"))
    parser.add_argument("--out-pose-dir", type=Path, default=Path("data/penn_action/pose_roi_aligned"))
    parser.add_argument("--penn-labels-dir", type=Path, default=Path("/data1/shiyuqi/Penn_Action/labels"))
    parser.add_argument("--log-json", type=Path, default=Path("outputs/logs/penn_pose_roi_align_log.json"))
    args = parser.parse_args()

    args.out_pose_dir.mkdir(parents=True, exist_ok=True)
    args.log_json.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    for split in ["train", "val", "test"]:
        csv_path = args.split_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"roi-{split}"):
            pose_rel = Path(str(row["pose_path"]))
            in_npz = args.in_pose_dir / pose_rel
            out_npz = args.out_pose_dir / pose_rel
            vid = str(row.get("video_id", Path(str(row["video_path"])).stem))
            label_mat = args.penn_labels_dir / f"{vid}.mat"

            if not in_npz.exists():
                logs.append({"split": split, "video_id": vid, "success": False, "reason": "pose_missing"})
                continue
            if not label_mat.exists():
                logs.append({"split": split, "video_id": vid, "success": False, "reason": "label_missing"})
                continue

            ret = align_one(in_npz, label_mat, out_npz)
            ret.update({"split": split, "video_id": vid, "in_pose": str(in_npz), "out_pose": str(out_npz)})
            logs.append(ret)

    summary = {
        "total": len(logs),
        "success_count": int(sum(1 for x in logs if x.get("success"))),
        "failed_count": int(sum(1 for x in logs if not x.get("success"))),
        "items": logs,
    }
    args.log_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
