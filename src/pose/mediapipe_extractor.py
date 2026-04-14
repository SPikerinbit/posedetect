from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


@dataclass
class PoseExtractionResult:
    keypoints: np.ndarray  # [T, 33, 4]
    fps: float
    frame_count: int
    width: int
    height: int
    success: bool


class MediaPipePoseExtractor:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        model_path: Path | None = None,
        use_gpu_delegate: bool = False,
    ):
        # model_complexity is kept for backward-compatible CLI signature.
        if model_path is None:
            model_path = Path("models/pose_landmarker_lite.task")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose Landmarker task file not found: {model_path}. "
                "Download from MediaPipe model hub, e.g. pose_landmarker_lite.task."
            )

        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if bool(use_gpu_delegate)
            else mp_python.BaseOptions.Delegate.CPU
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path), delegate=delegate),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=float(min_detection_confidence),
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

    @staticmethod
    def _extract_keypoints(result) -> np.ndarray:
        kps = np.full((33, 4), np.nan, dtype=np.float32)
        if not result.pose_landmarks:
            return kps
        pts = result.pose_landmarks[0]
        n = min(33, len(pts))
        for i in range(n):
            lm = pts[i]
            kps[i, 0] = float(getattr(lm, "x", np.nan))
            kps[i, 1] = float(getattr(lm, "y", np.nan))
            kps[i, 2] = float(getattr(lm, "z", np.nan))
            vis = getattr(lm, "visibility", None)
            prs = getattr(lm, "presence", None)
            if vis is not None:
                kps[i, 3] = float(vis)
            elif prs is not None:
                kps[i, 3] = float(prs)
            else:
                kps[i, 3] = 1.0
        return kps

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
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_img)
            kps = self._extract_keypoints(result)
            all_keypoints.append(kps)
            read_frames += 1
            if max_frames is not None and read_frames >= max_frames:
                break

        cap.release()
        keypoints = np.array(all_keypoints, dtype=np.float32)
        return PoseExtractionResult(
            keypoints=keypoints,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            success=keypoints.shape[0] > 0,
        )

    def extract_frame_dir(self, frame_dir: Path, fps: float = 30.0, max_frames: int | None = None) -> PoseExtractionResult:
        image_paths = sorted(frame_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = sorted(frame_dir.glob("*.png"))
        if not image_paths:
            return PoseExtractionResult(
                keypoints=np.zeros((0, 33, 4), dtype=np.float32),
                fps=float(fps),
                frame_count=0,
                width=0,
                height=0,
                success=False,
            )

        all_keypoints = []
        width = 0
        height = 0
        read_frames = 0

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            if width <= 0 or height <= 0:
                height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_img)
            kps = self._extract_keypoints(result)
            all_keypoints.append(kps)

            read_frames += 1
            if max_frames is not None and read_frames >= max_frames:
                break

        keypoints = np.array(all_keypoints, dtype=np.float32)
        return PoseExtractionResult(
            keypoints=keypoints,
            fps=float(fps),
            frame_count=int(len(image_paths)),
            width=int(width),
            height=int(height),
            success=keypoints.shape[0] > 0,
        )
