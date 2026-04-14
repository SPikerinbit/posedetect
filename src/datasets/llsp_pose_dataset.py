from __future__ import annotations

from pathlib import Path

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


class LLSPPoseDataset(Dataset):
    def __init__(self, split_csv: Path, pose_root: Path, seq_len: int = 96, preload: bool = True):
        self.df = pd.read_csv(split_csv)
        self.pose_root = pose_root
        self.seq_len = seq_len
        self.preload = preload
        labels = sorted(self.df["label_name"].unique().tolist())
        self.label2id = {k: i for i, k in enumerate(labels)}
        self.cache = []
        if self.preload:
            for i in range(len(self.df)):
                self.cache.append(self._load_item(i))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self.preload:
            return self.cache[idx]
        return self._load_item(idx)

    def _load_item(self, idx: int):
        row = self.df.iloc[idx]
        pose_path = self.pose_root / Path(row["pose_path"])
        data = np.load(pose_path, allow_pickle=True)
        keypoints = preprocess_keypoints(data["keypoints"].astype(np.float32))
        seq = _resample_seq(keypoints[:, :, :2], self.seq_len).reshape(self.seq_len, -1)
        x = torch.from_numpy(seq).float()
        y_cls = torch.tensor(self.label2id[row["label_name"]], dtype=torch.long)
        y_count = torch.tensor(float(row.get("count_label", 0.0)), dtype=torch.float32)
        y_dur = torch.tensor(float(row.get("duration_label_sec", 0.0)), dtype=torch.float32)
        return x, y_cls, y_count, y_dur
