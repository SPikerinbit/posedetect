from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.pose_preprocess import preprocess_keypoints


def _resample_seq(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((target_len, x.shape[1], x.shape[2]), dtype=np.float32)
    idx = np.linspace(0, x.shape[0] - 1, target_len).astype(np.int64)
    return x[idx]


class PoseSequenceDataset(Dataset):
    def __init__(self, split_csv: Path, pose_root: Path, seq_len: int = 96):
        self.df = pd.read_csv(split_csv)
        self.pose_root = pose_root
        self.seq_len = seq_len
        labels = sorted(self.df["label_name"].unique().tolist())
        self.label2id = {k: i for i, k in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_pose = Path(row["pose_path"])
        pose_path = self.pose_root / rel_pose
        data = np.load(pose_path, allow_pickle=True)
        keypoints = data["keypoints"].astype(np.float32)
        keypoints = preprocess_keypoints(keypoints)[:, :, :2]  # [T,33,2]
        seq = _resample_seq(keypoints, self.seq_len)
        seq = seq.reshape(self.seq_len, -1)  # [T,66]

        x = torch.from_numpy(seq).float()
        y = torch.tensor(self.label2id[row["label_name"]], dtype=torch.long)
        meta = {
            "video_path": row["video_path"],
            "label_name": row["label_name"],
            "group_id": row.get("group_id", ""),
        }
        return x, y, meta

    def save_label_map(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.label2id, ensure_ascii=False, indent=2), encoding="utf-8")
