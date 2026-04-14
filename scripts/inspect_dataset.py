#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".mpeg", ".mpg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class VideoMeta:
    path: str
    class_name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    readable: bool


def _safe_quantiles(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * q)
    return float(sorted_values[idx])


def _describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p25": _safe_quantiles(values, 0.25),
        "p75": _safe_quantiles(values, 0.75),
    }


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    all_files = [p for p in dataset_root.rglob("*") if p.is_file()]
    extension_counts = Counter(p.suffix.lower() for p in all_files)

    # Assume class-by-folder unless split folders exist.
    split_candidates = {"train", "val", "valid", "validation", "test"}
    top_level_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    top_level_names = {p.name.lower() for p in top_level_dirs}
    has_splits = len(top_level_names & split_candidates) > 0

    class_counts: dict[str, int] = defaultdict(int)
    label_source = "folder_name"
    media_type = "unknown"

    video_metas: list[VideoMeta] = []
    unreadable_files: list[str] = []
    empty_files: list[str] = []

    for path in all_files:
        suffix = path.suffix.lower()
        if path.stat().st_size == 0:
            empty_files.append(str(path))

        if has_splits:
            parts = path.relative_to(dataset_root).parts
            if len(parts) >= 2:
                class_name = parts[1]
                class_counts[class_name] += 1
        else:
            class_name = path.parent.name
            class_counts[class_name] += 1

        if suffix in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(path))
            readable = cap.isOpened()
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if readable else 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
            cap.release()

            if (not readable) or frame_count <= 0 or width <= 0 or height <= 0:
                unreadable_files.append(str(path))

            duration = frame_count / fps if fps and frame_count > 0 else 0.0
            video_metas.append(
                VideoMeta(
                    path=str(path),
                    class_name=class_name,
                    fps=fps,
                    frame_count=frame_count,
                    width=width,
                    height=height,
                    duration_sec=duration,
                    readable=readable,
                )
            )

    # Determine media type
    ext_set = {ext for ext, cnt in extension_counts.items() if cnt > 0}
    if ext_set & VIDEO_EXTS and not (ext_set & IMAGE_EXTS):
        media_type = "video"
    elif ext_set & IMAGE_EXTS and not (ext_set & VIDEO_EXTS):
        media_type = "image"
    elif (ext_set & IMAGE_EXTS) and (ext_set & VIDEO_EXTS):
        media_type = "mixed"

    frame_counts = [m.frame_count for m in video_metas if m.frame_count > 0]
    fps_values = [m.fps for m in video_metas if m.fps > 0]
    durations = [m.duration_sec for m in video_metas if m.duration_sec > 0]
    widths = [m.width for m in video_metas if m.width > 0]
    heights = [m.height for m in video_metas if m.height > 0]

    split_detail = {"has_train_val_test": has_splits}
    if has_splits:
        split_detail["detected_split_dirs"] = sorted(top_level_names & split_candidates)
    else:
        split_detail["detected_split_dirs"] = []

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "total_files": len(all_files),
        "classes": dict(sorted(class_counts.items())),
        "num_classes": len(class_counts),
        "extension_counts": dict(sorted(extension_counts.items())),
        "media_type": media_type,
        "label_source": label_source,
        "splits": split_detail,
        "file_integrity": {
            "empty_files_count": len(empty_files),
            "empty_files": empty_files[:50],
            "unreadable_files_count": len(unreadable_files),
            "unreadable_files": unreadable_files[:50],
        },
        "video_stats": {
            "count": len(video_metas),
            "fps": _describe(fps_values),
            "frame_count": _describe([float(v) for v in frame_counts]),
            "duration_sec": _describe(durations),
            "width": _describe([float(v) for v in widths]),
            "height": _describe([float(v) for v in heights]),
            "avg_resolution": {
                "width": float(mean(widths)) if widths else None,
                "height": float(mean(heights)) if heights else None,
            },
        },
        "video_samples": [asdict(v) for v in video_metas[:20]],
    }
    return summary


def to_markdown(summary: dict[str, Any]) -> str:
    cls_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in summary["classes"].items()
    )
    ext_lines = "\n".join(
        f"- `{ext or '[no_ext]'}`: {count}"
        for ext, count in summary["extension_counts"].items()
    )
    vs = summary["video_stats"]

    md = f"""# Dataset Summary

## 1. Basic Info
- Dataset root: `{summary["dataset_root"]}`
- Total files: {summary["total_files"]}
- Num classes: {summary["num_classes"]}
- Media type: `{summary["media_type"]}`
- Label source: `{summary["label_source"]}`
- Has train/val/test split dirs: `{summary["splits"]["has_train_val_test"]}`
- Detected split dirs: {summary["splits"]["detected_split_dirs"]}

## 2. Class Distribution
{cls_lines or "- 未检测到类别"}

## 3. File Extensions
{ext_lines or "- 未检测到文件"}

## 4. Integrity Check
- Empty files: {summary["file_integrity"]["empty_files_count"]}
- Unreadable video files: {summary["file_integrity"]["unreadable_files_count"]}

## 5. Video Stats
- Video count: {vs["count"]}
- FPS: {json.dumps(vs["fps"], ensure_ascii=False)}
- Frame count: {json.dumps(vs["frame_count"], ensure_ascii=False)}
- Duration (sec): {json.dumps(vs["duration_sec"], ensure_ascii=False)}
- Width: {json.dumps(vs["width"], ensure_ascii=False)}
- Height: {json.dumps(vs["height"], ensure_ascii=False)}
- Avg resolution: {json.dumps(vs["avg_resolution"], ensure_ascii=False)}
"""
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset structure and quality.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/dataset_summary.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/dataset_summary.md"))
    args = parser.parse_args()

    summary = inspect_dataset(args.dataset_root)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)

    args.report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report_md.write_text(to_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.report_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
