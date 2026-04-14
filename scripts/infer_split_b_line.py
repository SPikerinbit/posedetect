#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.b_line.full_detector import FullTemporalLocalizer
from src.features.pose_preprocess import preprocess_keypoints


def to_fixed_length(x: np.ndarray, seq_len: int) -> np.ndarray:
    t = x.shape[0]
    if t == 0:
        return np.zeros((seq_len, x.shape[1]), dtype=x.dtype)
    if t == seq_len:
        return x
    if t > seq_len:
        idx = np.linspace(0, t - 1, seq_len).astype(np.int64)
        return x[idx]
    return np.pad(x, ((0, seq_len - t), (0, 0)), mode="edge")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B-line full inference for an entire split CSV.")
    parser.add_argument("--config", type=Path, default=Path("configs/penn_b_line.yaml"))
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=Path("outputs/predictions/penn_b_test.jsonl"))
    parser.add_argument("--save-per-video", action="store_true")
    parser.add_argument("--per-video-dir", type=Path, default=Path("outputs/predictions/penn_b_videos"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    ckpt_path = Path(cfg.get("output_dir", "outputs/checkpoints/b_line/penn")) / str(cfg.get("checkpoint_name", "best_full.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt.get(
        "model_args",
        {
            "input_dim": int(cfg.get("input_dim", 66)),
            "hidden_dim": int(cfg.get("hidden_dim", 256)),
            "num_classes": int(cfg.get("num_classes", 3)),
            "num_transformer_layers": int(cfg.get("num_transformer_layers", 2)),
            "num_heads": int(cfg.get("num_heads", 8)),
        },
    )
    model = FullTemporalLocalizer(**model_args)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = pd.read_csv(args.split_csv)
    pose_root = Path(cfg.get("pose_root_ucf", "data/penn_action/pose"))
    seq_len = int(cfg.get("seq_len", 256))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.save_per_video:
        args.per_video_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    total = 0
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="infer-b-full"):
            total += 1
            pose_rel = Path(str(row["pose_path"]))
            pose_path = pose_root / pose_rel
            if not pose_path.exists():
                continue

            d = np.load(pose_path, allow_pickle=True)
            fps = float(d["fps"][0]) if "fps" in d else 25.0
            x = preprocess_keypoints(d["keypoints"].astype(np.float32))[:, :, :2].reshape(-1, 66)
            x = to_fixed_length(x, seq_len=seq_len)

            schema = model.predict_schema(
                torch.from_numpy(x).float(),
                video_path=str(row["video_path"]),
                fps=fps,
                score_thr=float(cfg.get("score_thr", 0.35)),
                nms_iou_thr=float(cfg.get("nms_iou_thr", 0.5)),
                min_segment_frames=int(cfg.get("min_segment_frames", 4)),
                max_segments=int(cfg.get("max_segments", 200)),
            )
            payload = schema.to_dict()
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if args.save_per_video:
                out_json = args.per_video_dir / f"{Path(str(row['video_path'])).stem}.json"
                out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            success += 1

    print(json.dumps({"total": total, "success": success, "out_jsonl": str(args.out_jsonl)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
