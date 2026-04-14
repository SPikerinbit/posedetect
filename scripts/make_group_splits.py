#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".mpeg", ".mpg"}


def parse_group_id(name: str) -> str:
    m = re.search(r"_g(\d+)_", name)
    return m.group(1) if m else "unknown"


def collect_samples(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for p in sorted(cls_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append(
                    {
                        "video_path": str(p.as_posix()),
                        "label_name": cls,
                        "group_id": parse_group_id(p.name),
                        "pose_path": f"{cls}/{p.stem}.npz",
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No video files found in {dataset_root}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create group-aware train/val/test split CSVs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits_group"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_samples(args.dataset_root)

    groups = df["group_id"].values
    splitter1 = GroupShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=args.seed)
    train_idx, rest_idx = next(splitter1.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    rest_df = df.iloc[rest_idx].reset_index(drop=True)

    rest_groups = rest_df["group_id"].values
    val_share = args.val_ratio / (1.0 - args.train_ratio)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=args.seed)
    val_idx, test_idx = next(splitter2.split(rest_df, groups=rest_groups))
    val_df = rest_df.iloc[val_idx].reset_index(drop=True)
    test_df = rest_df.iloc[test_idx].reset_index(drop=True)

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df.to_csv(args.out_dir / f"{split_name}.csv", index=False, encoding="utf-8")

    summary = pd.DataFrame(
        {"split": ["train", "val", "test"], "count": [len(train_df), len(val_df), len(test_df)]}
    )
    summary.to_csv(args.out_dir / "split_summary.csv", index=False, encoding="utf-8")

    print(summary.to_string(index=False))
    print("\nClass counts:")
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[{name}]")
        print(sub["label_name"].value_counts().to_string())
    print("\nGroup overlap check:")
    tr_g, va_g, te_g = set(train_df.group_id), set(val_df.group_id), set(test_df.group_id)
    print(f"train∩val={len(tr_g & va_g)}, train∩test={len(tr_g & te_g)}, val∩test={len(va_g & te_g)}")


if __name__ == "__main__":
    main()
