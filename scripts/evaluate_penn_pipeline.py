#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_label(x: str) -> str:
    m = {
        "PushUps": "push_up",
        "JumpingJack": "jumping_jack",
        "pushup": "push_up",
        "jumping_jacks": "jumping_jack",
        "push_up": "push_up",
        "jumping_jack": "jumping_jack",
    }
    return m.get(str(x), str(x))


def segment_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1e-8)


def greedy_match(gt: list[tuple[float, float]], pred: list[tuple[float, float, float]], thr: float) -> tuple[int, int, int]:
    used = np.zeros((len(gt),), dtype=np.int32)
    tp = 0
    fp = 0

    pred_sorted = sorted(pred, key=lambda x: x[2], reverse=True)
    for ps, pe, _ in pred_sorted:
        best_iou = 0.0
        best_j = -1
        for j, (gs, ge) in enumerate(gt):
            if used[j] == 1:
                continue
            iou = segment_iou((ps, pe), (gs, ge))
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= thr:
            used[best_j] = 1
            tp += 1
        else:
            fp += 1
    fn = int((used == 0).sum())
    return tp, fp, fn


def load_predictions(pred_input: Path) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    # video -> label -> [(start, end, score)]
    out: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))

    if pred_input.is_dir():
        files = sorted(pred_input.glob("*.json"))
        for p in files:
            d = json.loads(p.read_text(encoding="utf-8"))
            video_path = str(d.get("video_path", p.stem))
            for seg in d.get("segments", []):
                lab = normalize_label(str(seg.get("label", "")))
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                sc = float(seg.get("score", 1.0))
                if e > s:
                    out[video_path][lab].append((s, e, sc))
        return out

    # jsonl mode
    with pred_input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            video_path = str(d.get("video_path", ""))
            for seg in d.get("segments", []):
                lab = normalize_label(str(seg.get("label", "")))
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                sc = float(seg.get("score", 1.0))
                if e > s:
                    out[video_path][lab].append((s, e, sc))
    return out


def load_ground_truth(gt_csv: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    out: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    df = pd.read_csv(gt_csv)
    req = {"video_path", "start_time", "end_time", "label_name"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing required columns in gt csv: {sorted(miss)}")

    for _, row in df.iterrows():
        s = float(row["start_time"])
        e = float(row["end_time"])
        if e <= s:
            continue
        v = str(row["video_path"])
        lab = normalize_label(str(row["label_name"]))
        out[v][lab].append((s, e))
    return out


def evaluate(
    preds: dict[str, dict[str, list[tuple[float, float, float]]]],
    gt: dict[str, dict[str, list[tuple[float, float]]]],
    iou_thresholds: list[float],
) -> dict[str, Any]:
    labels = sorted({l for vv in gt.values() for l in vv.keys()} | {l for vv in preds.values() for l in vv.keys()})

    iou_metrics = {}
    for thr in iou_thresholds:
        tp = fp = fn = 0
        for video in sorted(set(gt.keys()) | set(preds.keys())):
            for lab in labels:
                g = gt.get(video, {}).get(lab, [])
                p = preds.get(video, {}).get(lab, [])
                tpi, fpi, fni = greedy_match(g, p, thr)
                tp += tpi
                fp += fpi
                fn += fni
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou_metrics[f"iou@{thr:.1f}"] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    count_err: dict[str, list[float]] = defaultdict(list)
    dur_err: dict[str, list[float]] = defaultdict(list)
    for video in sorted(set(gt.keys()) | set(preds.keys())):
        for lab in labels:
            g = gt.get(video, {}).get(lab, [])
            p = preds.get(video, {}).get(lab, [])
            count_err[lab].append(abs(len(g) - len(p)))
            g_dur = sum(e - s for s, e in g)
            p_dur = sum(e - s for s, e, _ in p)
            dur_err[lab].append(abs(g_dur - p_dur))

    count_mae = {lab: float(np.mean(vals)) if vals else 0.0 for lab, vals in count_err.items()}
    dur_mae = {lab: float(np.mean(vals)) if vals else 0.0 for lab, vals in dur_err.items()}

    return {
        "num_videos_gt": len(gt),
        "num_videos_pred": len(preds),
        "labels": labels,
        "segment_metrics": iou_metrics,
        "count_mae_by_label": count_mae,
        "duration_mae_by_label": dur_mae,
        "count_mae_overall": float(np.mean(list(count_mae.values()))) if count_mae else 0.0,
        "duration_mae_overall": float(np.mean(list(dur_mae.values()))) if dur_mae else 0.0,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Penn Pipeline Evaluation",
        "",
        f"- GT videos: {report['num_videos_gt']}",
        f"- Pred videos: {report['num_videos_pred']}",
        f"- Labels: {', '.join(report['labels'])}",
        "",
        "## Segment Metrics",
    ]
    for k, v in report["segment_metrics"].items():
        lines.append(f"- {k}: P={v['precision']:.4f}, R={v['recall']:.4f}, F1={v['f1']:.4f} (TP={v['tp']}, FP={v['fp']}, FN={v['fn']})")

    lines.extend(["", "## Count MAE", f"- Overall: {report['count_mae_overall']:.4f}"])
    for k, v in report["count_mae_by_label"].items():
        lines.append(f"- {k}: {v:.4f}")

    lines.extend(["", "## Duration MAE (sec)", f"- Overall: {report['duration_mae_overall']:.4f}"])
    for k, v in report["duration_mae_by_label"].items():
        lines.append(f"- {k}: {v:.4f}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted segments against Penn-style GT CSV.")
    parser.add_argument("--pred-input", type=Path, required=True, help="JSONL file or directory of per-video JSON")
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/metrics/penn_eval.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/metrics/penn_eval.md"))
    parser.add_argument("--iou-thresholds", type=str, default="0.3,0.5")
    args = parser.parse_args()

    iou_thresholds = [float(x.strip()) for x in args.iou_thresholds.split(",") if x.strip()]
    preds = load_predictions(args.pred_input)
    gt = load_ground_truth(args.gt_csv)
    report = evaluate(preds, gt, iou_thresholds=iou_thresholds)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(to_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
