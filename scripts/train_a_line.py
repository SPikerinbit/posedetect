#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.a_line.pipeline import ALinePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train A-line feature engineering model.")
    parser.add_argument("--config", type=Path, default=Path("configs/a_line.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_variant = str(cfg.get("split_variant", "group"))
    split_root = Path(cfg.get("split_root", "data/rebuild_dataset"))
    split_dir = split_root / "ucf_binary" / split_variant

    pipeline = ALinePipeline(
        window_size=int(cfg.get("window_size", 48)),
        stride=int(cfg.get("stride", 8)),
        background_quantile=float(cfg.get("background_quantile", 0.2)),
        median_kernel=int(cfg.get("median_kernel", 9)),
        min_segment_frames=int(cfg.get("min_segment_frames", 10)),
        max_gap_frames=int(cfg.get("max_gap_frames", 6)),
        peak_min_distance_sec=float(cfg.get("peak_min_distance_sec", 0.35)),
        peak_prominence_ratio=float(cfg.get("peak_prominence_ratio", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )

    metrics = pipeline.fit(
        train_csv=split_dir / "train.csv",
        val_csv=split_dir / "val.csv",
        pose_root_ucf=Path(cfg.get("pose_root_ucf", "data/pose")),
        pose_root_llsp=Path(cfg.get("pose_root_llsp", "data/pose_llsp")),
    )

    model_out = Path(cfg.get("model_out", "outputs/checkpoints/a_line/a_line_model.joblib"))
    metrics_out = Path(cfg.get("metrics_out", "outputs/metrics/a_line_metrics.json"))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    pipeline.save(model_out)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
