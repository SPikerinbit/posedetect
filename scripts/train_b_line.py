#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.b_line.full_detector import FullTemporalLocalizer, TemporalLocalizationDataset
from src.utils.seed import set_seed


def train_full(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = TemporalLocalizationDataset(
        annotation_csv=Path(cfg["train_annotation_csv"]),
        pose_root_ucf=Path(cfg.get("pose_root_ucf", "data/pose")),
        pose_root_llsp=Path(cfg.get("pose_root_llsp", "data/pose_llsp")),
        seq_len=int(cfg.get("seq_len", 256)),
    )
    ds_val = TemporalLocalizationDataset(
        annotation_csv=Path(cfg["val_annotation_csv"]),
        pose_root_ucf=Path(cfg.get("pose_root_ucf", "data/pose")),
        pose_root_llsp=Path(cfg.get("pose_root_llsp", "data/pose_llsp")),
        seq_len=int(cfg.get("seq_len", 256)),
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model_args = {
        "input_dim": int(cfg.get("input_dim", 66)),
        "hidden_dim": int(cfg.get("hidden_dim", 256)),
        "num_classes": int(cfg.get("num_classes", 3)),
        "num_transformer_layers": int(cfg.get("num_transformer_layers", 2)),
        "num_heads": int(cfg.get("num_heads", 8)),
    }
    model = FullTemporalLocalizer(**model_args).to(device)
    use_data_parallel = bool(cfg.get("use_data_parallel", True))
    if device.type == "cuda" and use_data_parallel and torch.cuda.device_count() > 1:
        gpu_ids = cfg.get("gpu_ids", None)
        if gpu_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, str):
            device_ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
        else:
            device_ids = [int(x) for x in gpu_ids]
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    best_val = 0.0
    epochs = int(cfg.get("epochs", 12))
    reg_loss_weight = float(cfg.get("reg_loss_weight", 1.0))
    checkpoint_name = str(cfg.get("checkpoint_name", "best_full.pt"))

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_cls = 0.0
        train_reg = 0.0
        n_train_steps = 0

        for x, y_cls, w_cls, y_lr, reg_mask, _, _ in dl_train:
            x = x.to(device)
            y_cls = y_cls.to(device)
            w_cls = w_cls.to(device)
            y_lr = y_lr.to(device)
            reg_mask = reg_mask.to(device)

            cls_logits, lr_pred = model(x)
            loss, losses = model.detection_loss(
                cls_logits=cls_logits,
                lr_pred=lr_pred,
                y_cls=y_cls,
                w_cls=w_cls,
                y_lr=y_lr,
                reg_mask=reg_mask,
                reg_loss_weight=reg_loss_weight,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += losses["total_loss"]
            train_cls += losses["cls_loss"]
            train_reg += losses["reg_loss"]
            n_train_steps += 1

        model.eval()
        val_loss = 0.0
        val_cls = 0.0
        val_reg = 0.0
        n_val_steps = 0
        tot = 0
        correct = 0

        with torch.no_grad():
            for x, y_cls, w_cls, y_lr, reg_mask, _, _ in dl_val:
                x = x.to(device)
                y_cls = y_cls.to(device)
                w_cls = w_cls.to(device)
                y_lr = y_lr.to(device)
                reg_mask = reg_mask.to(device)

                cls_logits, lr_pred = model(x)
                loss, losses = model.detection_loss(
                    cls_logits=cls_logits,
                    lr_pred=lr_pred,
                    y_cls=y_cls,
                    w_cls=w_cls,
                    y_lr=y_lr,
                    reg_mask=reg_mask,
                    reg_loss_weight=reg_loss_weight,
                )

                pred = torch.argmax(cls_logits, dim=-1)
                correct += int((pred == y_cls).sum().item())
                tot += int(y_cls.numel())

                val_loss += float(losses["total_loss"])
                val_cls += float(losses["cls_loss"])
                val_reg += float(losses["reg_loss"])
                n_val_steps += 1

        val_acc = float(correct / max(tot, 1))

        if val_acc >= best_val:
            best_val = val_acc
            out_dir = Path(cfg.get("output_dir", "outputs/checkpoints/b_line"))
            out_dir.mkdir(parents=True, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "model": state_dict,
                    "model_args": model_args,
                    "cfg": cfg,
                },
                out_dir / checkpoint_name,
            )

        print(
            json.dumps(
                {
                    "epoch": ep + 1,
                    "train_total_loss": train_loss / max(n_train_steps, 1),
                    "train_cls_loss": train_cls / max(n_train_steps, 1),
                    "train_reg_loss": train_reg / max(n_train_steps, 1),
                    "val_total_loss": val_loss / max(n_val_steps, 1),
                    "val_cls_loss": val_cls / max(n_val_steps, 1),
                    "val_reg_loss": val_reg / max(n_val_steps, 1),
                    "val_frame_acc": val_acc,
                },
                ensure_ascii=False,
            )
        )

    return {
        "backend": "full",
        "train_videos": len(ds_train),
        "val_videos": len(ds_val),
        "best_val_frame_acc": best_val,
        "checkpoint": str(Path(cfg.get("output_dir", "outputs/checkpoints/b_line")) / checkpoint_name),
    }


def train_opentad(cfg: dict) -> dict:
    cmd = str(cfg.get("opentad_train_cmd", "")).strip()
    if not cmd:
        raise RuntimeError("opentad_train_cmd is empty")
    work_dir = Path(cfg.get("opentad_work_dir", "outputs/checkpoints/b_line/opentad"))
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    cfg_sha1 = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()
    meta = {
        "mode": "opentad",
        "config_sha1": cfg_sha1,
        "train_annotation_csv": str(cfg.get("train_annotation_csv", "")),
        "val_annotation_csv": str(cfg.get("val_annotation_csv", "")),
        "test_annotation_csv": str(cfg.get("test_annotation_csv", "")),
        "opentad_train_cmd": cmd,
        "opentad_work_dir": str(work_dir),
    }
    (work_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    full_cmd = f"{cmd} --work-dir {work_dir.as_posix()}"
    proc = subprocess.run(full_cmd, shell=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"OpenTAD training failed with returncode={proc.returncode}")
    return {"backend": "opentad", "work_dir": str(work_dir), "status": "completed", "config_sha1": cfg_sha1}


def train_opentad_manual(cfg: dict) -> dict:
    manual_train = str(cfg.get("manual_train_annotation_csv", "")).strip()
    manual_val = str(cfg.get("manual_val_annotation_csv", "")).strip()
    if not manual_train:
        raise RuntimeError("manual_train_annotation_csv is empty")
    if not manual_val:
        raise RuntimeError("manual_val_annotation_csv is empty")
    if not Path(manual_train).exists():
        raise RuntimeError(f"manual_train_annotation_csv not found: {manual_train}")
    if not Path(manual_val).exists():
        raise RuntimeError(f"manual_val_annotation_csv not found: {manual_val}")

    manual_cfg = dict(cfg)
    manual_cfg["train_annotation_csv"] = manual_train
    manual_cfg["val_annotation_csv"] = manual_val
    if str(cfg.get("manual_test_annotation_csv", "")).strip():
        manual_cfg["test_annotation_csv"] = str(cfg.get("manual_test_annotation_csv", "")).strip()
    return train_opentad(manual_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B-line temporal localization model.")
    parser.add_argument("--config", type=Path, default=Path("configs/b_line.yaml"))
    parser.add_argument("--backend", type=str, default=None, help="full|opentad|opentad_manual")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backend = args.backend or str(cfg.get("backend", "full"))

    if backend == "full":
        metrics = train_full(cfg)
    elif backend == "opentad":
        metrics = train_opentad(cfg)
    elif backend == "opentad_manual":
        metrics = train_opentad_manual(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    metrics_out = Path(cfg.get("metrics_out", "outputs/metrics/b_line_metrics.json"))
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
