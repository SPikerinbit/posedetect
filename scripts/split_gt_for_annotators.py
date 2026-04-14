#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ASSIGNEES = ["syq", "szh", "lrx"]


def _read_with_split(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"video_path", "start_time", "end_time", "label_name"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"{csv_path} missing required columns: {sorted(miss)}")
    out = df.copy()
    out["split"] = split_name
    out["source_file"] = str(csv_path)
    if "source_type" not in out.columns:
        out["source_type"] = "pseudo"
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    return out


def _split_videos_round_robin(video_df: pd.DataFrame) -> pd.DataFrame:
    # balance by annotation row count (greedy)
    video_df = video_df.sort_values(["ann_rows", "video_path"], ascending=[False, True]).reset_index(drop=True)
    loads = {a: 0 for a in ASSIGNEES}
    assigned = []
    for _, row in video_df.iterrows():
        a = min(ASSIGNEES, key=lambda x: (loads[x], x))
        loads[a] += int(row["ann_rows"])
        assigned.append(a)
    video_df["assignee"] = assigned
    return video_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Split pseudo GT into 3 annotator CSVs by video (syq/szh/lrx).")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("gt"))
    parser.add_argument("--sample-ratio", type=float, default=0.15, help="fraction per split to annotate; 0<r<=1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("sample-ratio must be in (0, 1]")

    df_train = _read_with_split(args.train_csv, "train")
    df_val = _read_with_split(args.val_csv, "val")
    df_test = _read_with_split(args.test_csv, "test")
    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # sample videos at split level (default 15%)
    sampled_parts = []
    for split_name, sdf in all_df.groupby("split", sort=False):
        vids = sorted(sdf["video_path"].astype(str).unique().tolist())
        n_total = len(vids)
        n_take = max(1, int(round(n_total * float(args.sample_ratio))))
        take = pd.Series(vids).sample(n=n_take, random_state=args.seed).tolist()
        sampled_parts.append(sdf[sdf["video_path"].isin(set(take))].copy())
    sampled = pd.concat(sampled_parts, ignore_index=True)

    # assign by video to avoid cross-person overlap on same video
    video_stats = (
        sampled.groupby(["split", "video_path"], as_index=False)
        .size()
        .rename(columns={"size": "ann_rows"})
    )
    video_assigned = _split_videos_round_robin(video_stats)

    merged = sampled.merge(video_assigned[["split", "video_path", "assignee"]], on=["split", "video_path"], how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    common_cols = [
        "assignee",
        "split",
        "video_path",
        "label_name",
        "start_time",
        "end_time",
        "confidence",
        "source_type",
        "source_file",
    ]
    extra_cols = [c for c in merged.columns if c not in common_cols]
    out_cols = common_cols + extra_cols

    for a in ASSIGNEES:
        sub = merged[merged["assignee"] == a].copy()
        sub = sub.sort_values(["split", "video_path", "start_time", "end_time"]).reset_index(drop=True)
        sub.to_csv(args.out_dir / f"{a}.csv", index=False, encoding="utf-8", columns=[c for c in out_cols if c in sub.columns])

    # mapping and summary
    mapping = video_assigned.sort_values(["assignee", "split", "video_path"]).reset_index(drop=True)
    mapping.to_csv(args.out_dir / "mapping_video_to_annotator.csv", index=False, encoding="utf-8")

    summary = {
        "sample_ratio": float(args.sample_ratio),
        "input_rows": int(len(all_df)),
        "sampled_rows": int(len(merged)),
        "input_videos_by_split": {
            k: int(v)
            for k, v in all_df.groupby("split")["video_path"].nunique().to_dict().items()
        },
        "sampled_videos_by_split": {
            k: int(v)
            for k, v in merged.groupby("split")["video_path"].nunique().to_dict().items()
        },
        "assignee_rows": {a: int((merged["assignee"] == a).sum()) for a in ASSIGNEES},
        "assignee_videos": {
            a: int(merged[merged["assignee"] == a]["video_path"].nunique()) for a in ASSIGNEES
        },
        "outputs": [
            str(args.out_dir / "syq.csv"),
            str(args.out_dir / "szh.csv"),
            str(args.out_dir / "lrx.csv"),
            str(args.out_dir / "mapping_video_to_annotator.csv"),
        ],
    }
    (args.out_dir / "assignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
