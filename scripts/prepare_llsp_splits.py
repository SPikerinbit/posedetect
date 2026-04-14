#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def parse_group_id(name: str) -> str:
    m = re.match(r"(stu\d+)_", name)
    return m.group(1) if m else "unknown"


def load_one(action_dir: Path, action_name: str) -> pd.DataFrame:
    ann = pd.read_csv(action_dir / "annotations.csv")
    l_cols = [c for c in ann.columns if c.startswith("L")]
    rows = []
    for _, r in ann.iterrows():
        name = str(r["name"])
        video_path = action_dir / name
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        ts = pd.to_numeric(r[l_cols], errors="coerce").dropna().astype(float).values
        if len(ts) >= 2:
            motion_frames = float(np.max(ts) - np.min(ts))
        else:
            motion_frames = float(frame_count)
        duration_sec = motion_frames / fps if fps > 0 else 0.0
        count_raw = pd.to_numeric(r["count"], errors="coerce")
        if pd.isna(count_raw):
            continue
        count = float(count_raw)

        rows.append(
            {
                "video_path": str(video_path.as_posix()),
                "label_name": action_name,
                "group_id": parse_group_id(name),
                "count_label": count,
                "duration_label_sec": duration_sec,
                "fps": fps,
                "frame_count": frame_count,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLSP train/val/test splits with count+duration labels.")
    parser.add_argument("--llsp-root", type=Path, default=Path("dataset/LLSP"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits_llsp"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jj = load_one(args.llsp_root / "jump_jack_data", "JumpingJack")
    pu = load_one(args.llsp_root / "push_up_data", "PushUps")
    df = pd.concat([jj, pu], ignore_index=True)

    groups = df["group_id"].values
    sp1 = GroupShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=args.seed)
    train_idx, rest_idx = next(sp1.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    rest_df = df.iloc[rest_idx].reset_index(drop=True)

    val_share = args.val_ratio / (1.0 - args.train_ratio)
    sp2 = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=args.seed)
    val_idx, test_idx = next(sp2.split(rest_df, groups=rest_df["group_id"].values))
    val_df = rest_df.iloc[val_idx].reset_index(drop=True)
    test_df = rest_df.iloc[test_idx].reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(args.out_dir / f"{name}.csv", index=False, encoding="utf-8")

    summary = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "count": [len(train_df), len(val_df), len(test_df)],
        }
    )
    summary.to_csv(args.out_dir / "split_summary.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nAction dist:")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[{name}]")
        print(part["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
