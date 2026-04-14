#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.a_line.pipeline import ALinePipeline
from src.features.pose_preprocess import preprocess_keypoints


def infer_a_line(model_path: Path, pose_path: Path, video_path: str, out_json: Path) -> None:
    pipe = ALinePipeline.load(model_path)
    d = np.load(pose_path, allow_pickle=True)
    fps = float(d["fps"][0]) if "fps" in d else 25.0
    inf = pipe.infer_video(pose_path)
    schema = pipe.to_schema(video_path=video_path, inf=inf, fps=fps)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(schema.to_json(), encoding="utf-8")


def infer_b_minimal(checkpoint: Path, pose_path: Path, video_path: str, out_json: Path, seq_len: int = 128) -> None:
    import torch
    from src.b_line.minimal_detector import MinimalTemporalLocalizer

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MinimalTemporalLocalizer(input_dim=66, hidden_dim=192, num_classes=3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    d = np.load(pose_path, allow_pickle=True)
    fps = float(d["fps"][0]) if "fps" in d else 25.0
    x = preprocess_keypoints(d["keypoints"].astype(np.float32))[:, :, :2].reshape(-1, 66)
    t = x.shape[0]
    if t > seq_len:
        idxs = np.linspace(0, t - 1, seq_len).astype(np.int64)
        x = x[idxs]
    elif t < seq_len:
        x = np.pad(x, ((0, seq_len - t), (0, 0)), mode="edge")

    x_t = torch.from_numpy(x).float()
    schema = model.predict_schema(x_t, video_path=video_path, fps=fps)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(schema.to_json(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified inference to output segments + instances schema.")
    parser.add_argument("--line", choices=["a", "b_minimal"], default="a")
    parser.add_argument("--config", type=Path, default=Path("configs/a_line.yaml"))
    parser.add_argument("--pose-path", type=Path, required=True)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/predictions/infer_segments.json"))
    args = parser.parse_args()

    if args.line == "a":
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        model_path = Path(cfg.get("model_out", "outputs/checkpoints/a_line/a_line_model.joblib"))
        infer_a_line(model_path, args.pose_path, args.video_path, args.out_json)
    else:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        ckpt = Path(cfg.get("output_dir", "outputs/checkpoints/b_line")) / "best_minimal.pt"
        infer_b_minimal(ckpt, args.pose_path, args.video_path, args.out_json, seq_len=int(cfg.get("seq_len", 128)))

    print(str(args.out_json))


if __name__ == "__main__":
    main()
