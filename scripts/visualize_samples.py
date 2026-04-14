#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def grab_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random sample frames by split.")
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--out-file", type=Path, default=Path("outputs/figures/sample_grid.png"))
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv).sample(n=min(args.num, len(pd.read_csv(args.split_csv))), random_state=args.seed)
    n = len(df)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        cap = cv2.VideoCapture(str(row["video_path"]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx = max(total // 2, 0)
        img = grab_frame(Path(row["video_path"]), idx)
        plt.subplot(rows, cols, i)
        if img is not None:
            plt.imshow(img)
        plt.axis("off")
        plt.title(f"{row['label_name']}\n{Path(row['video_path']).name}")
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_file, dpi=180)
    plt.close()
    print(f"Saved: {args.out_file}")


if __name__ == "__main__":
    main()
