#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample manual-refine subset from split CSV.")
    parser.add_argument("--input-split-csv", type=Path, required=True)
    parser.add_argument("--manual-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-manual-csv", type=Path, required=True)
    parser.add_argument("--out-rest-csv", type=Path, required=True)
    args = parser.parse_args()

    if not (0.0 < args.manual_ratio < 1.0):
        raise ValueError("manual_ratio must be in (0, 1)")

    df = pd.read_csv(args.input_split_csv)
    need = {"video_path", "label_name"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing required columns: {sorted(miss)}")

    if "video_id" not in df.columns:
        df["video_id"] = df["video_path"].astype(str).map(lambda x: Path(x).stem)

    manual_df, rest_df = train_test_split(
        df,
        test_size=(1.0 - float(args.manual_ratio)),
        random_state=int(args.seed),
        stratify=df["label_name"],
    )

    manual_df = manual_df.copy()
    manual_df["source_type"] = "manual_candidate"

    args.out_manual_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_rest_csv.parent.mkdir(parents=True, exist_ok=True)

    manual_df.to_csv(args.out_manual_csv, index=False, encoding="utf-8")
    rest_df.to_csv(args.out_rest_csv, index=False, encoding="utf-8")

    summary = {
        "input": int(len(df)),
        "manual": int(len(manual_df)),
        "rest": int(len(rest_df)),
        "manual_ratio": float(len(manual_df) / max(len(df), 1)),
        "manual_class": manual_df["label_name"].value_counts().to_dict(),
        "rest_class": rest_df["label_name"].value_counts().to_dict(),
        "out_manual_csv": str(args.out_manual_csv),
        "out_rest_csv": str(args.out_rest_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
