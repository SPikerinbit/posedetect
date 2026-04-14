#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pose.mediapipe_extractor import MediaPipePoseExtractor


def pose_rel_path(video_path: str) -> Path:
    p = Path(video_path)
    return Path(p.parent.name) / f"{p.stem}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose keypoints from videos.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--pose-dir", type=Path, default=Path("data/pose"))
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-json", type=Path, default=Path("outputs/logs/pose_extraction_log.json"))
    args = parser.parse_args()

    args.pose_dir.mkdir(parents=True, exist_ok=True)
    args.log_json.parent.mkdir(parents=True, exist_ok=True)

    extractor = MediaPipePoseExtractor(model_complexity=args.model_complexity)
    logs = []
    max_frames = args.max_frames if args.max_frames > 0 else None

    for split_name in ["train", "val", "test"]:
        split_csv = args.split_dir / f"{split_name}.csv"
        df = pd.read_csv(split_csv)
        pose_paths = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"extract-{split_name}"):
            video_path = Path(row["video_path"])
            rel = pose_rel_path(str(video_path))
            out_path = args.pose_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and out_path.stat().st_size > 0:
                pose_paths.append(str(rel.as_posix()))
                logs.append(
                    {
                        "split": split_name,
                        "video_path": str(video_path),
                        "pose_path": str(rel.as_posix()),
                        "success": True,
                        "num_frames_saved": -1,
                        "elapsed_sec": 0.0,
                        "skipped_existing": True,
                    }
                )
                continue

            t0 = perf_counter()
            res = extractor.extract_video(video_path, max_frames=max_frames)
            elapsed = perf_counter() - t0

            np.savez_compressed(
                out_path,
                keypoints=res.keypoints,
                fps=np.array([res.fps], dtype=np.float32),
                frame_count=np.array([res.frame_count], dtype=np.int32),
                width=np.array([res.width], dtype=np.int32),
                height=np.array([res.height], dtype=np.int32),
                success=np.array([int(res.success)], dtype=np.int32),
                label_name=np.array([row["label_name"]]),
                video_path=np.array([str(video_path)]),
            )
            pose_paths.append(str(rel.as_posix()))
            logs.append(
                {
                    "split": split_name,
                    "video_path": str(video_path),
                    "pose_path": str(rel.as_posix()),
                    "success": bool(res.success),
                    "num_frames_saved": int(res.keypoints.shape[0]),
                    "elapsed_sec": elapsed,
                }
            )

        df["pose_path"] = pose_paths
        df.to_csv(split_csv, index=False, encoding="utf-8")

    summary = {
        "total": len(logs),
        "success_count": int(sum(1 for x in logs if x["success"])),
        "failed_count": int(sum(1 for x in logs if not x["success"])),
        "avg_time_sec": float(np.mean([x["elapsed_sec"] for x in logs])) if logs else 0.0,
        "items": logs,
    }
    args.log_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
