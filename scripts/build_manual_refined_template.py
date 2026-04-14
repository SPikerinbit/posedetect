#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual refinement template from pseudo annotations and selected videos.")
    parser.add_argument("--manual-videos-csv", type=Path, required=True)
    parser.add_argument("--pseudo-annotation-csv", type=Path, required=True)
    parser.add_argument("--output-template-csv", type=Path, required=True)
    args = parser.parse_args()

    videos_df = pd.read_csv(args.manual_videos_csv)
    pseudo_df = pd.read_csv(args.pseudo_annotation_csv)

    if "video_path" not in videos_df.columns:
        raise RuntimeError("manual-videos-csv must contain video_path column")
    req = {"video_path", "start_time", "end_time", "label_name"}
    miss = req - set(pseudo_df.columns)
    if miss:
        raise RuntimeError(f"pseudo-annotation-csv missing columns: {sorted(miss)}")

    keep = set(videos_df["video_path"].astype(str).tolist())
    out_df = pseudo_df[pseudo_df["video_path"].astype(str).isin(keep)].copy()
    out_df["source_type"] = "manual_refined"
    out_df["review_status"] = "todo"
    out_df["review_note"] = ""

    args.output_template_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_template_csv, index=False, encoding="utf-8")

    summary = {
        "manual_videos": len(keep),
        "template_rows": int(len(out_df)),
        "output": str(args.output_template_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
