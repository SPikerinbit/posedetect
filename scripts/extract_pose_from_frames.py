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


def infer_pose_rel_path(row: pd.Series) -> Path:
    if "pose_path" in row and isinstance(row["pose_path"], str) and row["pose_path"].strip():
        return Path(row["pose_path"])
    video_path = Path(str(row["video_path"]))
    return Path(str(row["label_name"])) / f"{video_path.stem}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe poses from frame folders.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/penn_action/rebuild_dataset/ucf_binary/penn"))
    parser.add_argument("--pose-dir", type=Path, default=Path("data/penn_action/pose"))
    parser.add_argument("--frames-root", type=Path, default=Path("/data1/shiyuqi/Penn_Action/frames"))
    parser.add_argument("--assume-fps", type=float, default=30.0)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--use-gpu-delegate", action="store_true", help="Use MediaPipe GPU delegate when available.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--log-json", type=Path, default=Path("outputs/logs/penn_pose_extraction_log.json"))
    args = parser.parse_args()

    from src.pose.mediapipe_extractor import MediaPipePoseExtractor

    args.pose_dir.mkdir(parents=True, exist_ok=True)
    args.log_json.parent.mkdir(parents=True, exist_ok=True)

    extractor = MediaPipePoseExtractor(
        model_complexity=args.model_complexity,
        model_path=args.model_path,
        use_gpu_delegate=bool(args.use_gpu_delegate),
    )
    max_frames = args.max_frames if args.max_frames > 0 else None

    logs = []
    for split_name in ["train", "val", "test"]:
        split_csv = args.split_dir / f"{split_name}.csv"
        if not split_csv.exists():
            continue

        df = pd.read_csv(split_csv)
        pose_paths = []
        success_flags = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"pose-{split_name}"):
            pose_rel = infer_pose_rel_path(row)
            out_path = args.pose_dir / pose_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            frame_dir = None
            if "frame_dir" in row and isinstance(row["frame_dir"], str) and row["frame_dir"].strip():
                frame_dir = Path(row["frame_dir"])
            elif "video_id" in row:
                frame_dir = args.frames_root / str(row["video_id"])

            if frame_dir is None or not frame_dir.exists():
                logs.append(
                    {
                        "split": split_name,
                        "video_path": str(row.get("video_path", "")),
                        "frame_dir": str(frame_dir) if frame_dir is not None else "",
                        "pose_path": str(pose_rel.as_posix()),
                        "success": False,
                        "num_frames_saved": 0,
                        "elapsed_sec": 0.0,
                        "error": "frame_dir_missing",
                    }
                )
                pose_paths.append(str(pose_rel.as_posix()))
                success_flags.append(False)
                continue

            if out_path.exists() and out_path.stat().st_size > 0:
                logs.append(
                    {
                        "split": split_name,
                        "video_path": str(row.get("video_path", "")),
                        "frame_dir": str(frame_dir.as_posix()),
                        "pose_path": str(pose_rel.as_posix()),
                        "success": True,
                        "num_frames_saved": -1,
                        "elapsed_sec": 0.0,
                        "skipped_existing": True,
                    }
                )
                pose_paths.append(str(pose_rel.as_posix()))
                success_flags.append(True)
                continue

            t0 = perf_counter()
            res = extractor.extract_frame_dir(frame_dir=frame_dir, fps=args.assume_fps, max_frames=max_frames)
            elapsed = perf_counter() - t0

            np.savez_compressed(
                out_path,
                keypoints=res.keypoints,
                fps=np.array([res.fps], dtype=np.float32),
                frame_count=np.array([res.frame_count], dtype=np.int32),
                width=np.array([res.width], dtype=np.int32),
                height=np.array([res.height], dtype=np.int32),
                success=np.array([int(res.success)], dtype=np.int32),
                label_name=np.array([str(row.get("label_name", ""))]),
                video_path=np.array([str(row.get("video_path", ""))]),
                fps_source=np.array(["assumed_30"]),
            )

            logs.append(
                {
                    "split": split_name,
                    "video_path": str(row.get("video_path", "")),
                    "frame_dir": str(frame_dir.as_posix()),
                    "pose_path": str(pose_rel.as_posix()),
                    "success": bool(res.success),
                    "num_frames_saved": int(res.keypoints.shape[0]),
                    "elapsed_sec": float(elapsed),
                }
            )
            pose_paths.append(str(pose_rel.as_posix()))
            success_flags.append(bool(res.success))

        df["pose_path"] = pose_paths
        df["pose_success"] = success_flags
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
    extractor.close()


if __name__ == "__main__":
    main()
