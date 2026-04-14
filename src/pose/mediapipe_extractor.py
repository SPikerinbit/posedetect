from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseExtractionResult:
    keypoints: np.ndarray  # [T, 33, 4]
    fps: float
    frame_count: int
    width: int
    height: int
    success: bool


class MediaPipePoseExtractor:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract_video(self, video_path: Path, max_frames: int | None = None) -> PoseExtractionResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return PoseExtractionResult(
                keypoints=np.zeros((0, 33, 4), dtype=np.float32),
                fps=0.0,
                frame_count=0,
                width=0,
                height=0,
                success=False,
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        all_keypoints = []
        read_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            kps = np.full((33, 4), np.nan, dtype=np.float32)
            if result.pose_landmarks is not None:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    kps[i, 0] = lm.x
                    kps[i, 1] = lm.y
                    kps[i, 2] = lm.z
                    kps[i, 3] = lm.visibility
            all_keypoints.append(kps)
            read_frames += 1
            if max_frames is not None and read_frames >= max_frames:
                break

        cap.release()
        self.pose.reset()

        keypoints = np.array(all_keypoints, dtype=np.float32)
        return PoseExtractionResult(
            keypoints=keypoints,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            success=keypoints.shape[0] > 0,
        )
