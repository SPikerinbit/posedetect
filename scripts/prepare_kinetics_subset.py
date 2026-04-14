#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "push up": "push_up",
    "jumping jacks": "jumping_jacks",
}


def normalize_row(row: pd.Series) -> pd.Series:
    start = int(row["time_start"])
    end = int(row["time_end"])
    yid = str(row["youtube_id"]).strip()
    label = str(row["label"]).strip().lower()
    split = str(row["split"]).strip().lower()
    label_dir = LABEL_MAP[label]
    clip_id = f"{yid}_{start:06d}_{end:06d}"
    url = f"https://www.youtube.com/watch?v={yid}"
    rel_path = f"{split}/{label_dir}/{clip_id}.mp4"
    return pd.Series(
        {
            "split": split,
            "label": label,
            "label_dir": label_dir,
            "youtube_id": yid,
            "time_start": start,
            "time_end": end,
            "clip_id": clip_id,
            "url": url,
            "rel_path": rel_path,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kinetics subset manifest by classes.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/kinetics_subset/annotations/k700_train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/kinetics_subset/annotations/k700_val.csv"))
    parser.add_argument("--out-manifest", type=Path, default=Path("data/kinetics_subset/manifests/k700_pushup_jumpingjacks_manifest.csv"))
    parser.add_argument("--max-per-class", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    df = pd.concat([train_df, val_df], ignore_index=True)
    df["label"] = df["label"].str.lower()

    target_labels = set(LABEL_MAP.keys())
    sub = df[df["label"].isin(target_labels)].copy()
    sub = sub.apply(normalize_row, axis=1)

    if args.max_per_class > 0:
        sub = (
            sub.sort_values(["label", "split", "youtube_id", "time_start"])
            .groupby("label", group_keys=False)
            .head(args.max_per_class)
            .reset_index(drop=True)
        )

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out_manifest, index=False, encoding="utf-8")

    stat = sub.groupby(["label", "split"]).size().reset_index(name="count")
    print(stat.to_string(index=False))
    print(f"\nTotal clips: {len(sub)}")
    print(f"Saved manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
