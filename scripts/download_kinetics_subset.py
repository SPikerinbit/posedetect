#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
import pandas as pd
from tqdm import tqdm


def iter_rows(df: pd.DataFrame) -> Iterable[dict]:
    for row in df.to_dict(orient="records"):
        yield row


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout[-4000:]


def download_clip(
    url: str,
    start: int,
    end: int,
    out_file: Path,
    temp_dir: Path,
    ffmpeg_bin: str,
    retries: int,
) -> tuple[bool, str]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_template = str((temp_dir / "%(id)s.%(ext)s").resolve())
    for i in range(retries + 1):
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--quiet",
            "--no-warnings",
            "--ignore-errors",
            "--no-playlist",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "--download-sections",
            f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mp4",
            "--ffmpeg-location",
            ffmpeg_bin,
            "-o",
            str(out_file.resolve()),
            url,
        ]
        code, out = run_cmd(cmd)
        if code == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return True, "ok"
        msg = f"attempt={i+1}, code={code}, out={out[-500:]}"
    return False, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Kinetics clips by manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/kinetics_subset/manifests/k700_pushup_jumpingjacks_manifest.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/kinetics_subset/videos"))
    parser.add_argument("--temp-dir", type=Path, default=Path("data/kinetics_subset/.tmp"))
    parser.add_argument("--log-csv", type=Path, default=Path("data/kinetics_subset/logs/download_log.csv"))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 means all clips")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    existing_status = {}
    if args.log_csv.exists():
        old = pd.read_csv(args.log_csv)
        if not old.empty and "clip_id" in old.columns and "status" in old.columns:
            for r in old.to_dict(orient="records"):
                existing_status[r["clip_id"]] = r["status"]

    with args.log_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "label",
                "split",
                "url",
                "time_start",
                "time_end",
                "output_path",
                "status",
                "message",
            ],
        )
        if args.log_csv.stat().st_size == 0:
            writer.writeheader()

        ok_count = 0
        fail_count = 0
        skip_count = 0
        for row in tqdm(iter_rows(df), total=len(df), desc="download"):
            clip_id = row["clip_id"]
            rel = Path(row["rel_path"])
            out_file = args.output_root / rel
            if out_file.exists() and out_file.stat().st_size > 0:
                skip_count += 1
                continue
            if existing_status.get(clip_id) == "ok":
                skip_count += 1
                continue

            ok, msg = download_clip(
                url=row["url"],
                start=int(row["time_start"]),
                end=int(row["time_end"]),
                out_file=out_file,
                temp_dir=args.temp_dir,
                ffmpeg_bin=ffmpeg_bin,
                retries=args.retries,
            )
            status = "ok" if ok else "fail"
            if ok:
                ok_count += 1
            else:
                fail_count += 1

            writer.writerow(
                {
                    "clip_id": clip_id,
                    "label": row["label"],
                    "split": row["split"],
                    "url": row["url"],
                    "time_start": int(row["time_start"]),
                    "time_end": int(row["time_end"]),
                    "output_path": str(out_file.as_posix()),
                    "status": status,
                    "message": msg,
                }
            )
            f.flush()

    print(
        f"Finished. ok={ok_count}, fail={fail_count}, skipped={skip_count}, total_manifest={len(df)}"
    )
    print(f"Log: {args.log_csv}")


if __name__ == "__main__":
    main()
