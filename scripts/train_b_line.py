#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.b_line.minimal_detector import MinimalTemporalLocalizer, TemporalDataset


def train_minimal(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = TemporalDataset(
        annotation_csv=Path(cfg["train_annotation_csv"]),
        pose_root_ucf=Path(cfg.get("pose_root_ucf", "data/pose")),
        pose_root_llsp=Path(cfg.get("pose_root_llsp", "data/pose_llsp")),
        seq_len=int(cfg.get("seq_len", 128)),
    )
    ds_val = TemporalDataset(
        annotation_csv=Path(cfg["val_annotation_csv"]),
        pose_root_ucf=Path(cfg.get("pose_root_ucf", "data/pose")),
        pose_root_llsp=Path(cfg.get("pose_root_llsp", "data/pose_llsp")),
        seq_len=int(cfg.get("seq_len", 128)),
    )

    dl_train = DataLoader(ds_train, batch_size=int(cfg.get("batch_size", 8)), shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    dl_val = DataLoader(ds_val, batch_size=int(cfg.get("batch_size", 8)), shuffle=False, num_workers=int(cfg.get("num_workers", 0)))

    model = MinimalTemporalLocalizer(input_dim=66, hidden_dim=192, num_classes=3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))

    best_val = 0.0
    epochs = int(cfg.get("epochs", 8))
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, w, _, _ in dl_train:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction="none")
            loss = (loss * w.reshape(-1)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += float(loss.item())

        model.eval()
        tot = 0
        correct = 0
        with torch.no_grad():
            for x, y, _, _, _ in dl_val:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == y).sum().item())
                tot += int(y.numel())
        val_acc = float(correct / max(tot, 1))
        if val_acc >= best_val:
            best_val = val_acc
            out_dir = Path(cfg.get("output_dir", "outputs/checkpoints/b_line"))
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best_minimal.pt")

        print(json.dumps({"epoch": ep + 1, "train_loss": train_loss / max(len(dl_train), 1), "val_frame_acc": val_acc}, ensure_ascii=False))

    return {
        "backend": "minimal",
        "train_videos": len(ds_train),
        "val_videos": len(ds_val),
        "best_val_frame_acc": best_val,
        "checkpoint": str(Path(cfg.get("output_dir", "outputs/checkpoints/b_line")) / "best_minimal.pt"),
    }


def train_opentad(cfg: dict) -> dict:
    cmd = str(cfg.get("opentad_train_cmd", "")).strip()
    if not cmd:
        raise RuntimeError("opentad_train_cmd is empty")
    work_dir = Path(cfg.get("opentad_work_dir", "outputs/checkpoints/b_line/opentad"))
    work_dir.mkdir(parents=True, exist_ok=True)
    full_cmd = f"{cmd} --work-dir {work_dir.as_posix()}"
    proc = subprocess.run(full_cmd, shell=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"OpenTAD training failed with returncode={proc.returncode}")
    return {"backend": "opentad", "work_dir": str(work_dir), "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B-line temporal localization model.")
    parser.add_argument("--config", type=Path, default=Path("configs/b_line.yaml"))
    parser.add_argument("--backend", type=str, default=None, help="minimal|opentad")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backend = args.backend or str(cfg.get("backend", "minimal"))

    if backend == "minimal":
        metrics = train_minimal(cfg)
    elif backend == "opentad":
        metrics = train_opentad(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    metrics_out = Path(cfg.get("metrics_out", "outputs/metrics/b_line_metrics.json"))
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
