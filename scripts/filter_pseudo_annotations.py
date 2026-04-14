#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter/clean pseudo annotation CSV for B-line training.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--keep-source-types", type=str, default="pseudo_instance,pseudo")
    parser.add_argument("--min-duration-sec", type=float, default=0.10)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required_cols = {"video_path", "start_time", "end_time", "label_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    before = len(df)

    keep_sources = {x.strip() for x in args.keep_source_types.split(",") if x.strip()}
    if "source_type" in df.columns and keep_sources:
        df = df[df["source_type"].isin(keep_sources)]

    if "confidence" in df.columns:
        df = df[df["confidence"].astype(float) >= float(args.min_confidence)]

    df = df[df["end_time"].astype(float) > df["start_time"].astype(float)]
    df = df[(df["end_time"].astype(float) - df["start_time"].astype(float)) >= float(args.min_duration_sec)]

    dedupe_cols = [c for c in ["video_path", "start_time", "end_time", "label_name", "source_type"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    summary = {
        "input_rows": int(before),
        "output_rows": int(len(df)),
        "dropped_rows": int(before - len(df)),
        "class_counts": df["label_name"].value_counts().to_dict() if len(df) > 0 else {},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
