from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common.output_schema import InferenceSchema, Instance, Segment
from src.features.pose_preprocess import preprocess_keypoints

LABEL_TO_ID = {"background": 0, "push_up": 1, "jumping_jack": 2, "PushUps": 1, "JumpingJack": 2}
ID_TO_LABEL = {0: "background", 1: "push_up", 2: "jumping_jack"}


@dataclass
class VideoTarget:
    video_path: str
    intervals: list[tuple[float, float, int, float]]  # start_s, end_s, label_id, score


class TemporalLocalizationDataset(Dataset):
    def __init__(
        self,
        annotation_csv: Path,
        pose_root_ucf: Path,
        pose_root_llsp: Path,
        seq_len: int = 256,
    ):
        ann = pd.read_csv(annotation_csv)
        self.seq_len = seq_len
        self.pose_root_ucf = pose_root_ucf
        self.pose_root_llsp = pose_root_llsp
        self.samples: list[VideoTarget] = []

        grouped = ann.groupby("video_path")
        for video_path, g in grouped:
            intervals: list[tuple[float, float, int, float]] = []
            for _, row in g.iterrows():
                label = str(row.get("label_name", "background"))
                label_id = LABEL_TO_ID.get(label, 0)
                s = float(row.get("start_time", 0.0))
                e = float(row.get("end_time", 0.0))
                if e <= s:
                    continue
                conf = float(row.get("confidence", 1.0))
                intervals.append((s, e, label_id, conf))
            self.samples.append(VideoTarget(video_path=video_path, intervals=intervals))

    def __len__(self) -> int:
        return len(self.samples)

    def _pose_path(self, video_path: str) -> Path:
        rel = Path(video_path)
        pose_rel = Path(rel.parent.name) / f"{rel.stem}.npz"
        root = self.pose_root_llsp if "dataset/LLSP/" in str(video_path).replace("\\", "/") else self.pose_root_ucf
        return root / pose_rel

    @staticmethod
    def _to_fixed_length(x: np.ndarray, seq_len: int) -> np.ndarray:
        t = x.shape[0]
        if t == 0:
            return np.zeros((seq_len, x.shape[1]), dtype=x.dtype)
        if t == seq_len:
            return x
        if t > seq_len:
            idxs = np.linspace(0, t - 1, seq_len).astype(np.int64)
            return x[idxs]
        pad = seq_len - t
        return np.pad(x, ((0, pad), (0, 0)), mode="edge")

    @staticmethod
    def _to_fixed_length_1d(x: np.ndarray, seq_len: int, pad_value: float = 0.0) -> np.ndarray:
        t = x.shape[0]
        if t == 0:
            return np.full((seq_len,), pad_value, dtype=x.dtype)
        if t == seq_len:
            return x
        if t > seq_len:
            idxs = np.linspace(0, t - 1, seq_len).astype(np.int64)
            return x[idxs]
        pad = seq_len - t
        return np.pad(x, (0, pad), mode="constant", constant_values=pad_value)

    @staticmethod
    def _build_lr_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        lr = np.zeros((n, 2), dtype=np.float32)
        reg_mask = np.zeros((n,), dtype=np.float32)
        i = 0
        while i < n:
            lab = int(y[i])
            j = i + 1
            while j < n and int(y[j]) == lab:
                j += 1
            if lab != 0:
                for k in range(i, j):
                    lr[k, 0] = float(k - i)
                    lr[k, 1] = float(j - 1 - k)
                reg_mask[i:j] = 1.0
            i = j
        return lr, reg_mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        p = self._pose_path(sample.video_path)
        d = np.load(p, allow_pickle=True)
        fps = float(d["fps"][0]) if "fps" in d else 25.0
        keypoints = preprocess_keypoints(d["keypoints"].astype(np.float32))[:, :, :2]
        t = keypoints.shape[0]
        x = keypoints.reshape(t, -1)

        y = np.zeros((t,), dtype=np.int64)
        w = np.ones((t,), dtype=np.float32)
        for s_sec, e_sec, label_id, conf in sample.intervals:
            s = max(0, min(t - 1, int(s_sec * fps)))
            e = max(s + 1, min(t, int(e_sec * fps)))
            y[s:e] = label_id
            w[s:e] = np.maximum(w[s:e], conf)

        x_fix = self._to_fixed_length(x, self.seq_len)
        y_fix = self._to_fixed_length_1d(y, self.seq_len, pad_value=0).astype(np.int64)
        w_fix = self._to_fixed_length_1d(w, self.seq_len, pad_value=1.0).astype(np.float32)
        lr_fix, reg_mask = self._build_lr_targets(y_fix)

        return (
            torch.from_numpy(x_fix).float(),
            torch.from_numpy(y_fix).long(),
            torch.from_numpy(w_fix).float(),
            torch.from_numpy(lr_fix).float(),
            torch.from_numpy(reg_mask).float(),
            sample.video_path,
            fps,
        )


class _TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FullTemporalLocalizer(nn.Module):
    def __init__(
        self,
        input_dim: int = 66,
        hidden_dim: int = 256,
        num_classes: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv_stack = nn.Sequential(
            _TemporalConvBlock(hidden_dim, dilation=1),
            _TemporalConvBlock(hidden_dim, dilation=2),
            _TemporalConvBlock(hidden_dim, dilation=4),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        h = h.transpose(1, 2)
        h = self.conv_stack(h)
        h = h.transpose(1, 2)
        h = self.transformer(h)
        cls_logits = self.cls_head(h)
        lr = F.softplus(self.reg_head(h))
        return cls_logits, lr

    @staticmethod
    def detection_loss(
        cls_logits: torch.Tensor,
        lr_pred: torch.Tensor,
        y_cls: torch.Tensor,
        w_cls: torch.Tensor,
        y_lr: torch.Tensor,
        reg_mask: torch.Tensor,
        reg_loss_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = F.cross_entropy(cls_logits.reshape(-1, cls_logits.size(-1)), y_cls.reshape(-1), reduction="none")
        cls_loss = (cls_loss * w_cls.reshape(-1)).mean()

        reg_mask = reg_mask > 0
        if reg_mask.any():
            reg_loss = F.smooth_l1_loss(lr_pred[reg_mask], y_lr[reg_mask], reduction="mean")
        else:
            reg_loss = cls_loss.new_tensor(0.0)

        loss = cls_loss + reg_loss_weight * reg_loss
        return loss, {
            "cls_loss": float(cls_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "total_loss": float(loss.item()),
        }

    @staticmethod
    def _nms_1d(segs: list[tuple[int, int, int, float]], iou_thr: float) -> list[tuple[int, int, int, float]]:
        if not segs:
            return []
        segs = sorted(segs, key=lambda x: x[3], reverse=True)
        keep: list[tuple[int, int, int, float]] = []
        for cand in segs:
            _, s1, e1, _ = cand
            ok = True
            for kept in keep:
                _, s2, e2, _ = kept
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = max(e1, e2) - min(s1, s2)
                iou = inter / max(union, 1)
                if iou > iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(cand)
        return keep

    @torch.no_grad()
    def predict_schema(
        self,
        x: torch.Tensor,
        video_path: str,
        fps: float,
        score_thr: float = 0.35,
        nms_iou_thr: float = 0.5,
        min_segment_frames: int = 4,
        max_segments: int = 200,
    ) -> InferenceSchema:
        self.eval()
        cls_logits, lr = self(x.unsqueeze(0))
        probs = torch.softmax(cls_logits.squeeze(0), dim=-1).cpu().numpy()
        lr = lr.squeeze(0).cpu().numpy()

        labels = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)

        candidates: dict[int, list[tuple[int, int, int, float]]] = {1: [], 2: []}
        for i in range(len(labels)):
            lab = int(labels[i])
            if lab == 0:
                continue
            score = float(scores[i])
            if score < score_thr:
                continue
            left = int(round(float(lr[i, 0])))
            right = int(round(float(lr[i, 1])))
            s = max(0, i - left)
            e = min(len(labels), i + right + 1)
            if e - s < min_segment_frames:
                continue
            candidates[lab].append((lab, s, e, score))

        merged: list[tuple[int, int, int, float]] = []
        for lab in [1, 2]:
            kept = self._nms_1d(candidates[lab], iou_thr=nms_iou_thr)
            merged.extend(kept)

        merged = sorted(merged, key=lambda x: x[3], reverse=True)[:max_segments]
        merged = sorted(merged, key=lambda x: x[1])

        segments: list[Segment] = []
        instances: list[Instance] = []
        for rep_id, (lab, s, e, sc) in enumerate(merged, start=1):
            label = ID_TO_LABEL[lab]
            start = s / max(float(fps), 1e-6)
            end = e / max(float(fps), 1e-6)
            dur = end - start
            segments.append(Segment(label=label, start=start, end=end, duration=dur, score=float(sc), source="b_line_full"))
            instances.append(Instance(label=label, start=start, end=end, duration=dur, rep_id=rep_id, score=float(sc)))

        count_by_label = {
            "push_up": sum(1 for x in instances if x.label == "push_up"),
            "jumping_jack": sum(1 for x in instances if x.label == "jumping_jack"),
        }
        duration_by_label = {
            "push_up": float(sum(x.duration for x in segments if x.label == "push_up")),
            "jumping_jack": float(sum(x.duration for x in segments if x.label == "jumping_jack")),
        }
        summary = {
            "count_by_label": count_by_label,
            "total_duration_by_label": duration_by_label,
            "video_duration": float(len(labels) / max(float(fps), 1e-6)),
            "num_segments": len(segments),
        }
        return InferenceSchema(video_path=video_path, segments=segments, instances=instances, summary=summary)
