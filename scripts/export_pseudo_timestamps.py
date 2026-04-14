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
    parser = argparse.ArgumentParser(description="Export pseudo timestamps from A-line model.")
    parser.add_argument("--config", type=Path, default=Path("configs/a_line.yaml"))
    parser.add_argument("--split-variant", type=str, default=None, help="random|group")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/pseudo_labels"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_variant = args.split_variant or str(cfg.get("split_variant", "group"))
    split_root = Path(cfg.get("split_root", "data/rebuild_dataset"))
    split_dir = split_root / "ucf_binary" / split_variant

    model_path = Path(cfg.get("model_out", "outputs/checkpoints/a_line/a_line_model.joblib"))
    pipeline = ALinePipeline.load(model_path)

    pose_root_ucf = Path(cfg.get("pose_root_ucf", "data/pose"))
    pose_root_llsp = Path(cfg.get("pose_root_llsp", "data/pose_llsp"))

    out_dir = args.out_root / split_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ["train", "val", "test"]:
        ret = pipeline.export_pseudo_annotations(
            split_csv=split_dir / f"{split}.csv",
            pose_root_ucf=pose_root_ucf,
            pose_root_llsp=pose_root_llsp,
            out_csv=out_dir / f"{split}_tal.csv",
            out_jsonl=out_dir / f"{split}_infer.jsonl",
        )
        summary[split] = ret

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
