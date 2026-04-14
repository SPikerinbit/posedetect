#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED_LABELS = {"PushUps", "JumpingJack", "push_up", "jumping_jack"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manual-refined temporal annotations.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--strict-source-type", action="store_true")
    parser.add_argument("--expected-source-type", type=str, default="manual_refined")
    parser.add_argument("--output-report", type=Path, default=Path("outputs/reports/manual_annotation_validation.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"video_path", "start_time", "end_time", "label_name"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing required columns: {sorted(miss)}")

    issues = []
    if len(df) == 0:
        issues.append("empty_csv")

    bad_time = df[df["end_time"].astype(float) <= df["start_time"].astype(float)]
    if len(bad_time) > 0:
        issues.append(f"invalid_time_rows={len(bad_time)}")

    bad_label = df[~df["label_name"].astype(str).isin(ALLOWED_LABELS)]
    if len(bad_label) > 0:
        issues.append(f"invalid_label_rows={len(bad_label)}")

    dedupe_cols = [c for c in ["video_path", "start_time", "end_time", "label_name"] if c in df.columns]
    dup_rows = df.duplicated(subset=dedupe_cols).sum() if dedupe_cols else 0
    if dup_rows > 0:
        issues.append(f"duplicate_rows={int(dup_rows)}")

    if args.strict_source_type:
        if "source_type" not in df.columns:
            issues.append("source_type_missing")
        else:
            mismatch = (df["source_type"].astype(str) != str(args.expected_source_type)).sum()
            if mismatch > 0:
                issues.append(f"source_type_mismatch_rows={int(mismatch)}")

    report = {
        "input_csv": str(args.input_csv),
        "rows": int(len(df)),
        "classes": df["label_name"].value_counts().to_dict() if len(df) > 0 else {},
        "num_videos": int(df["video_path"].nunique()) if "video_path" in df.columns else 0,
        "issues": issues,
        "status": "ok" if len(issues) == 0 else "failed",
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
