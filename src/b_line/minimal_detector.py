from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.common.output_schema import InferenceSchema, Instance, Segment
from src.features.pose_preprocess import preprocess_keypoints

LABEL_TO_ID = {"background": 0, "push_up": 1, "jumping_jack": 2, "PushUps": 1, "JumpingJack": 2}
ID_TO_LABEL = {0: "background", 1: "push_up", 2: "jumping_jack"}


@dataclass
class VideoTarget:
    video_path: str
    intervals: list[tuple[int, int, int, float]]  # start_f, end_f, label_id, score


class TemporalDataset(Dataset):
    def __init__(
        self,
        annotation_csv: Path,
        pose_root_ucf: Path,
        pose_root_llsp: Path,
        seq_len: int = 128,
    ):
        ann = pd.read_csv(annotation_csv)
        self.seq_len = seq_len
        self.pose_root_ucf = pose_root_ucf
        self.pose_root_llsp = pose_root_llsp
        self.samples: list[VideoTarget] = []

        grouped = ann.groupby("video_path")
        for video_path, g in grouped:
            intervals = []
            for _, row in g.iterrows():
                label = str(row["label_name"])
                label_id = LABEL_TO_ID.get(label, 0)
                intervals.append(
                    (
                        float(row["start_time"]),
                        float(row["end_time"]),
                        label_id,
                        float(row.get("confidence", 1.0)),
                    )
                )
            self.samples.append(VideoTarget(video_path=video_path, intervals=intervals))

    def __len__(self) -> int:
        return len(self.samples)

    def _pose_path(self, video_path: str) -> Path:
        rel = Path(video_path)
        pose_rel = Path(rel.parent.name) / f"{rel.stem}.npz"
        root = self.pose_root_llsp if "dataset/LLSP/" in str(video_path).replace("\\", "/") else self.pose_root_ucf
        return root / pose_rel

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
            w[s:e] = max(w[s:e].mean(), conf)

        if t > self.seq_len:
            idxs = np.linspace(0, t - 1, self.seq_len).astype(np.int64)
            x = x[idxs]
            y = y[idxs]
            w = w[idxs]
        elif t < self.seq_len:
            pad = self.seq_len - t
            x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
            y = np.pad(y, (0, pad), mode="edge")
            w = np.pad(w, (0, pad), mode="edge")

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).long(),
            torch.from_numpy(w).float(),
            sample.video_path,
            fps,
        )


class MinimalTemporalLocalizer(nn.Module):
    def __init__(self, input_dim: int = 66, hidden_dim: int = 192, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.cls_head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h, _ = self.temporal(h)
        logits = self.cls_head(h)
        return logits

    @torch.no_grad()
    def predict_schema(self, x: torch.Tensor, video_path: str, fps: float) -> InferenceSchema:
        self.eval()
        logits = self(x.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)

        segments: list[Segment] = []
        instances: list[Instance] = []

        i = 0
        while i < len(labels):
            lab = int(labels[i])
            j = i + 1
            while j < len(labels) and int(labels[j]) == lab:
                j += 1
            if lab != 0 and (j - i) >= 2:
                label = ID_TO_LABEL[lab]
                start = i / fps
                end = j / fps
                score = float(np.mean(scores[i:j]))
                segments.append(Segment(label=label, start=start, end=end, duration=end - start, score=score, source="b_line_minimal"))
                instances.append(Instance(label=label, start=start, end=end, duration=end - start, rep_id=len(instances) + 1, score=score))
            i = j

        count_by_label = {"push_up": sum(1 for x in instances if x.label == "push_up"), "jumping_jack": sum(1 for x in instances if x.label == "jumping_jack")}
        duration_by_label = {
            "push_up": float(sum(x.duration for x in segments if x.label == "push_up")),
            "jumping_jack": float(sum(x.duration for x in segments if x.label == "jumping_jack")),
        }
        summary = {
            "count_by_label": count_by_label,
            "total_duration_by_label": duration_by_label,
            "video_duration": len(labels) / max(fps, 1e-6),
        }
        return InferenceSchema(video_path=video_path, segments=segments, instances=instances, summary=summary)
