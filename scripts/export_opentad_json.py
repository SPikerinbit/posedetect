#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export temporal annotations to OpenTAD-style JSON interface.")
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--subset", type=str, default="train", help="train|val|test")
    parser.add_argument("--keep-source-types", type=str, default="", help="comma-separated source_type filter")
    parser.add_argument("--version-name", type=str, default="penn_action_annotations_v1")
    args = parser.parse_args()

    df = pd.read_csv(args.annotation_csv)
    req = {"video_path", "start_time", "end_time", "label_name"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing required columns: {sorted(miss)}")

    keep_sources = {x.strip() for x in str(args.keep_source_types).split(",") if x.strip()}
    if keep_sources:
        if "source_type" not in df.columns:
            raise RuntimeError("keep-source-types provided but source_type column is missing in annotation csv")
        df = df[df["source_type"].astype(str).isin(keep_sources)].reset_index(drop=True)

    database = {}
    classes = sorted(df["label_name"].astype(str).unique().tolist())

    for video_path, g in df.groupby("video_path"):
        duration = float(g["end_time"].astype(float).max()) if len(g) > 0 else 0.0
        annotations = []
        for _, row in g.iterrows():
            s = float(row["start_time"])
            e = float(row["end_time"])
            if e <= s:
                continue
            annotations.append(
                {
                    "segment": [s, e],
                    "label": str(row["label_name"]),
                    "score": float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else 1.0,
                }
            )
        database[str(video_path)] = {
            "duration": duration,
            "subset": args.subset,
            "annotations": annotations,
        }

    out = {
        "version": str(args.version_name),
        "classes": classes,
        "database": database,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(database), "classes": classes, "output": str(args.output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
