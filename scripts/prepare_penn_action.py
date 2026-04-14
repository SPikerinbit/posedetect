#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split


ACTION_TO_LABEL = {
    "pushup": "PushUps",
    "jumping_jacks": "JumpingJack",
}


def _extract_scalar_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def collect_penn_rows(penn_root: Path, keep_actions: set[str]) -> pd.DataFrame:
    labels_dir = penn_root / "labels"
    frames_dir = penn_root / "frames"
    rows: list[dict] = []

    for mat_path in sorted(labels_dir.glob("*.mat")):
        ann = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        action = str(ann["action"])
        if action not in keep_actions:
            continue

        label_name = ACTION_TO_LABEL[action]
        video_id = mat_path.stem
        frame_dir = frames_dir / video_id
        if not frame_dir.exists():
            continue

        train_flag = _extract_scalar_int(ann["train"])
        nframes = _extract_scalar_int(ann["nframes"])

        rows.append(
            {
                "video_id": video_id,
                "video_path": f"dataset/{label_name}/{video_id}.avi",
                "label_name": label_name,
                "label_raw": action,
                "train_flag": train_flag,
                "group_id": video_id,
                "pose_path": f"{label_name}/{video_id}.npz",
                "frame_dir": str(frame_dir.as_posix()),
                "nframes": nframes,
                "fps": 30.0,
                "fps_source": "assumed_30",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No samples found in {penn_root} for actions={sorted(keep_actions)}")
    return df


def split_official_train_val_test(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool = df[df["train_flag"] == 1].reset_index(drop=True)
    test_df = df[df["train_flag"] != 1].reset_index(drop=True)

    if train_pool.empty or test_df.empty:
        raise RuntimeError("Official Penn train/test split appears empty after filtering.")

    if not (0.0 < val_ratio < 0.5):
        raise ValueError("val_ratio must be in (0, 0.5)")

    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_pool["label_name"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


def write_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_root: Path) -> None:
    split_dir = out_root / "ucf_binary" / "penn"
    label_dir = out_root / "labels"
    split_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "video_id",
        "video_path",
        "label_name",
        "label_raw",
        "group_id",
        "pose_path",
        "frame_dir",
        "nframes",
        "fps",
        "fps_source",
    ]

    for name, sdf in (("train", train_df), ("val", val_df), ("test", test_df)):
        sdf[keep_cols].to_csv(split_dir / f"{name}.csv", index=False, encoding="utf-8")

    class_map = pd.DataFrame(
        [
            {"label_id": 0, "label_name": "background"},
            {"label_id": 1, "label_name": "PushUps"},
            {"label_id": 2, "label_name": "JumpingJack"},
        ]
    )
    class_map.to_csv(label_dir / "class_map.csv", index=False, encoding="utf-8")

    summary = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "train_class": train_df["label_name"].value_counts().to_dict(),
        "val_class": val_df["label_name"].value_counts().to_dict(),
        "test_class": test_df["label_name"].value_counts().to_dict(),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Penn Action splits for A/B pipeline.")
    parser.add_argument("--penn-root", type=Path, default=Path("/data1/shiyuqi/Penn_Action"))
    parser.add_argument("--out-root", type=Path, default=Path("data/penn_action/rebuild_dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--actions",
        type=str,
        default="pushup,jumping_jacks",
        help="comma-separated Penn raw action names",
    )
    args = parser.parse_args()

    keep_actions = {x.strip() for x in args.actions.split(",") if x.strip()}
    unsupported = keep_actions - set(ACTION_TO_LABEL.keys())
    if unsupported:
        raise ValueError(f"Unsupported actions: {sorted(unsupported)}")

    df = collect_penn_rows(args.penn_root, keep_actions=keep_actions)
    train_df, val_df, test_df = split_official_train_val_test(df, val_ratio=args.val_ratio, seed=args.seed)
    write_outputs(train_df, val_df, test_df, out_root=args.out_root)


if __name__ == "__main__":
    main()
