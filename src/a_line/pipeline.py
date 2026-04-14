from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import json
import numpy as np
import pandas as pd

from src.common.output_schema import InferenceSchema, Instance, Segment
from src.features.pose_preprocess import preprocess_keypoints
from src.features.window_features import build_video_windows, extract_window_feature_vector

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


LABEL_TO_ID = {"background": 0, "PushUps": 1, "JumpingJack": 2, "push_up": 1, "jumping_jack": 2}
ID_TO_LABEL = {0: "background", 1: "push_up", 2: "jumping_jack"}


@dataclass
class InferenceResult:
    frame_labels: np.ndarray
    frame_scores: np.ndarray
    segments: list[tuple[int, int, int, float]]  # label_id, start_f, end_f, score
    instances: list[tuple[int, int, int, float]]  # label_id, start_f, end_f, score


class ALinePipeline:
    def __init__(
        self,
        window_size: int = 48,
        stride: int = 8,
        background_quantile: float = 0.2,
        median_kernel: int = 9,
        min_segment_frames: int = 10,
        max_gap_frames: int = 6,
        peak_min_distance_sec: float = 0.35,
        peak_prominence_ratio: float = 0.15,
        seed: int = 42,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ):
        self.window_size = window_size
        self.stride = stride
        self.background_quantile = background_quantile
        self.median_kernel = median_kernel
        self.min_segment_frames = min_segment_frames
        self.max_gap_frames = max_gap_frames
        self.peak_min_distance_sec = peak_min_distance_sec
        self.peak_prominence_ratio = peak_prominence_ratio
        self.seed = seed
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.model: Any = None
        self.feature_names: list[str] = []

    def _make_model(self):
        if XGBClassifier is not None:
            xgb_kwargs = {}
            if self.use_gpu:
                xgb_kwargs["device"] = f"cuda:{self.gpu_id}"
                # xgboost>=2 uses histogram tree method with CUDA device.
                xgb_kwargs["tree_method"] = "hist"
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=self.seed,
                n_jobs=4,
                **xgb_kwargs,
            )
        return RandomForestClassifier(n_estimators=400, random_state=self.seed, n_jobs=4)

    def _pose_root_from_video_path(self, video_path: str, pose_root_ucf: Path, pose_root_llsp: Path) -> Path:
        return pose_root_llsp if "dataset/LLSP/" in str(video_path).replace("\\", "/") else pose_root_ucf

    def _build_split_xy(self, split_csv: Path, pose_root_ucf: Path, pose_root_llsp: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(split_csv)
        feats = []
        labels = []
        for _, row in df.iterrows():
            pose_root = self._pose_root_from_video_path(str(row["video_path"]), pose_root_ucf, pose_root_llsp)
            pose_path = pose_root / Path(str(row["pose_path"]))
            if not pose_path.exists():
                continue
            d = np.load(pose_path, allow_pickle=True)
            keypoints = d["keypoints"].astype(np.float32)
            fps = float(d["fps"][0]) if "fps" in d else 25.0
            action_id = LABEL_TO_ID.get(str(row["label_name"]), 0)
            if action_id == 0:
                continue
            batch = build_video_windows(
                keypoints=keypoints,
                fps=fps,
                window_size=self.window_size,
                stride=self.stride,
                action_label=action_id,
                background_quantile=self.background_quantile,
            )
            if batch.features.size == 0:
                continue
            feats.append(batch.features)
            labels.append(batch.labels)
        if not feats:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.vstack(feats), np.concatenate(labels)

    def fit(self, train_csv: Path, val_csv: Path, pose_root_ucf: Path, pose_root_llsp: Path) -> dict[str, float]:
        x_train, y_train = self._build_split_xy(train_csv, pose_root_ucf, pose_root_llsp)
        x_val, y_val = self._build_split_xy(val_csv, pose_root_ucf, pose_root_llsp)
        if x_train.shape[0] == 0:
            raise RuntimeError("No training samples built from split.")
        self.model = self._make_model()
        self.model.fit(x_train, y_train)
        pred_val = self.model.predict(x_val) if x_val.shape[0] > 0 else np.array([], dtype=np.int64)
        # Compatibility: some xgboost builds may return probability matrix from predict.
        if isinstance(pred_val, np.ndarray) and pred_val.ndim == 2:
            pred_val = np.argmax(pred_val, axis=1).astype(np.int64)
        metrics = {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "val_acc": float(accuracy_score(y_val, pred_val)) if pred_val.size else 0.0,
            "val_f1_macro": float(f1_score(y_val, pred_val, average="macro")) if pred_val.size else 0.0,
        }
        return metrics

    def _window_features_for_video(self, keypoints: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq = preprocess_keypoints(keypoints)
        t = seq.shape[0]
        feats = []
        starts = []
        ends = []
        for s in range(0, max(1, t - self.window_size + 1), self.stride):
            e = min(t, s + self.window_size)
            if e - s < self.window_size:
                s = max(0, t - self.window_size)
                e = t
            vec, names = extract_window_feature_vector(seq[s:e], fps=fps)
            if not self.feature_names:
                self.feature_names = names
            feats.append(vec)
            starts.append(s)
            ends.append(e)
            if e >= t:
                break
        return np.stack(feats).astype(np.float32), np.array(starts), np.array(ends)

    def _majority_filter(self, arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return arr
        k = k if k % 2 == 1 else k + 1
        pad = k // 2
        x = np.pad(arr, (pad, pad), mode="edge")
        out = np.empty_like(arr)
        for i in range(len(arr)):
            win = x[i : i + k]
            vals, cnts = np.unique(win, return_counts=True)
            out[i] = vals[np.argmax(cnts)]
        return out

    def _frames_from_windows(self, probs: np.ndarray, starts: np.ndarray, ends: np.ndarray, total_frames: int) -> tuple[np.ndarray, np.ndarray]:
        c = probs.shape[1]
        frame_probs = np.zeros((total_frames, c), dtype=np.float32)
        frame_cnt = np.zeros((total_frames, 1), dtype=np.float32)
        for i in range(len(starts)):
            s, e = int(starts[i]), int(ends[i])
            frame_probs[s:e] += probs[i]
            frame_cnt[s:e] += 1.0
        frame_cnt = np.clip(frame_cnt, 1.0, None)
        frame_probs = frame_probs / frame_cnt
        frame_labels = np.argmax(frame_probs, axis=1).astype(np.int64)
        frame_scores = np.max(frame_probs, axis=1)
        frame_labels = self._majority_filter(frame_labels, self.median_kernel)
        return frame_labels, frame_scores

    def _extract_segments(self, frame_labels: np.ndarray, frame_scores: np.ndarray) -> list[tuple[int, int, int, float]]:
        segs: list[tuple[int, int, int, float]] = []
        n = len(frame_labels)
        i = 0
        while i < n:
            lab = int(frame_labels[i])
            j = i + 1
            while j < n and int(frame_labels[j]) == lab:
                j += 1
            if lab != 0 and (j - i) >= self.min_segment_frames:
                segs.append((lab, i, j, float(np.mean(frame_scores[i:j]))))
            i = j

        if not segs:
            return segs
        merged = [segs[0]]
        for lab, s, e, sc in segs[1:]:
            plab, ps, pe, psc = merged[-1]
            if lab == plab and s - pe <= self.max_gap_frames:
                merged[-1] = (plab, ps, e, float((psc + sc) * 0.5))
            else:
                merged.append((lab, s, e, sc))
        return merged

    def _find_peaks(self, x: np.ndarray, min_distance: int, min_prominence: float) -> list[int]:
        peaks = []
        last = -10**9
        for i in range(1, len(x) - 1):
            if x[i] <= x[i - 1] or x[i] <= x[i + 1]:
                continue
            if i - last < min_distance:
                continue
            local_base = 0.5 * (max(x[max(0, i - min_distance):i], default=x[i]) + max(x[i + 1 : min(len(x), i + min_distance + 1)], default=x[i]))
            if x[i] - local_base < min_prominence:
                continue
            peaks.append(i)
            last = i
        return peaks

    def _count_instances(self, seg: tuple[int, int, int, float], keypoints: np.ndarray, fps: float) -> list[tuple[int, int, int, float]]:
        lab, s, e, sc = seg
        seq = preprocess_keypoints(keypoints[s:e])
        if seq.shape[0] < 5:
            return []
        if lab == 1:
            elbow_l = self._joint_angle(seq[:, 11, :3], seq[:, 13, :3], seq[:, 15, :3])
            elbow_r = self._joint_angle(seq[:, 12, :3], seq[:, 14, :3], seq[:, 16, :3])
            signal = -(0.5 * (elbow_l + elbow_r))
        else:
            signal = np.linalg.norm(seq[:, 27, :2] - seq[:, 28, :2], axis=1)

        min_d = max(3, int(self.peak_min_distance_sec * max(fps, 1.0)))
        prom = max(1e-4, self.peak_prominence_ratio * float(np.max(signal) - np.min(signal)))
        p1 = self._find_peaks(signal, min_distance=min_d, min_prominence=prom)
        p2 = self._find_peaks(-signal, min_distance=min_d, min_prominence=prom)
        peaks = p1 if len(p1) >= len(p2) else p2
        if not peaks:
            return []

        inst = []
        for i, p in enumerate(peaks):
            left = peaks[i - 1] if i > 0 else max(0, p - min_d)
            right = peaks[i + 1] if i + 1 < len(peaks) else min(len(signal) - 1, p + min_d)
            a = int((left + p) * 0.5)
            b = int((p + right) * 0.5)
            inst.append((lab, s + a, s + b + 1, sc))
        return inst

    def _joint_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ba = a - b
        bc = c - b
        den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
        den = np.clip(den, 1e-6, None)
        cosv = np.sum(ba * bc, axis=1) / den
        cosv = np.clip(cosv, -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    def infer_video(self, pose_path: Path) -> InferenceResult:
        if self.model is None:
            raise RuntimeError("Model not loaded or trained.")
        d = np.load(pose_path, allow_pickle=True)
        keypoints = d["keypoints"].astype(np.float32)
        fps = float(d["fps"][0]) if "fps" in d else 25.0
        feats, starts, ends = self._window_features_for_video(keypoints, fps)
        probs = self.model.predict_proba(feats)
        total_frames = keypoints.shape[0]
        frame_labels, frame_scores = self._frames_from_windows(probs, starts, ends, total_frames)
        segments = self._extract_segments(frame_labels, frame_scores)
        instances: list[tuple[int, int, int, float]] = []
        for seg in segments:
            instances.extend(self._count_instances(seg, keypoints, fps))
        return InferenceResult(frame_labels=frame_labels, frame_scores=frame_scores, segments=segments, instances=instances)

    def to_schema(self, video_path: str, inf: InferenceResult, fps: float) -> InferenceSchema:
        segments = []
        instances = []
        count_by_label = {"push_up": 0, "jumping_jack": 0}
        duration_by_label = {"push_up": 0.0, "jumping_jack": 0.0}

        for lab, s, e, sc in inf.segments:
            label = ID_TO_LABEL[lab]
            dur = (e - s) / max(fps, 1e-6)
            segments.append(Segment(label=label, start=s / fps, end=e / fps, duration=dur, score=float(sc), source="a_line"))
            duration_by_label[label] += dur

        for i, (lab, s, e, sc) in enumerate(inf.instances, start=1):
            label = ID_TO_LABEL[lab]
            dur = (e - s) / max(fps, 1e-6)
            instances.append(Instance(label=label, start=s / fps, end=e / fps, duration=dur, rep_id=i, score=float(sc)))
            count_by_label[label] += 1

        video_duration = len(inf.frame_labels) / max(fps, 1e-6)
        summary = {
            "count_by_label": count_by_label,
            "total_duration_by_label": duration_by_label,
            "video_duration": video_duration,
        }
        return InferenceSchema(video_path=video_path, segments=segments, instances=instances, summary=summary)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window_size": self.window_size,
            "stride": self.stride,
            "background_quantile": self.background_quantile,
            "median_kernel": self.median_kernel,
            "min_segment_frames": self.min_segment_frames,
            "max_gap_frames": self.max_gap_frames,
            "peak_min_distance_sec": self.peak_min_distance_sec,
            "peak_prominence_ratio": self.peak_prominence_ratio,
            "seed": self.seed,
            "use_gpu": self.use_gpu,
            "gpu_id": self.gpu_id,
            "model": self.model,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "ALinePipeline":
        payload = joblib.load(path)
        obj = cls(
            window_size=payload["window_size"],
            stride=payload["stride"],
            background_quantile=payload["background_quantile"],
            median_kernel=payload["median_kernel"],
            min_segment_frames=payload["min_segment_frames"],
            max_gap_frames=payload["max_gap_frames"],
            peak_min_distance_sec=payload["peak_min_distance_sec"],
            peak_prominence_ratio=payload["peak_prominence_ratio"],
            seed=payload["seed"],
            use_gpu=bool(payload.get("use_gpu", False)),
            gpu_id=int(payload.get("gpu_id", 0)),
        )
        obj.model = payload["model"]
        obj.feature_names = payload.get("feature_names", [])
        return obj

    def export_pseudo_annotations(
        self,
        split_csv: Path,
        pose_root_ucf: Path,
        pose_root_llsp: Path,
        out_csv: Path,
        out_jsonl: Path,
    ) -> dict[str, int]:
        df = pd.read_csv(split_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        jsonl = []
        for _, row in df.iterrows():
            video_path = str(row["video_path"])
            pose_root = self._pose_root_from_video_path(video_path, pose_root_ucf, pose_root_llsp)
            pose_path = pose_root / Path(str(row["pose_path"]))
            if not pose_path.exists():
                continue
            d = np.load(pose_path, allow_pickle=True)
            fps = float(d["fps"][0]) if "fps" in d else 25.0
            inf = self.infer_video(pose_path)
            schema = self.to_schema(video_path, inf, fps=fps)
            jsonl.append(schema.to_dict())
            for seg in schema.segments:
                rows.append(
                    {
                        "video_path": video_path,
                        "start_time": seg.start,
                        "end_time": seg.end,
                        "label_name": seg.label,
                        "label_id": LABEL_TO_ID[seg.label],
                        "source_type": "pseudo",
                        "confidence": seg.score,
                    }
                )
            for inst in schema.instances:
                rows.append(
                    {
                        "video_path": video_path,
                        "start_time": inst.start,
                        "end_time": inst.end,
                        "label_name": inst.label,
                        "label_id": LABEL_TO_ID[inst.label],
                        "source_type": "pseudo_instance",
                        "confidence": inst.score,
                    }
                )

        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
        with out_jsonl.open("w", encoding="utf-8") as f:
            for item in jsonl:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return {"videos": len(jsonl), "annotations": len(rows)}
