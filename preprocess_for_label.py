#!/usr/bin/env python3
"""
preprocess.py - 从帧序列合成视频并生成 Label Studio 任务 JSON
"""

import csv
import os
import subprocess
import json
from collections import defaultdict
from pathlib import Path

# ========== 配置 ==========
BASE_DIR = Path("./penn").resolve()          # Penn 数据根目录
INPUT_CSV = BASE_DIR / "input.csv"           # 输入 CSV
FRAMES_DIR = BASE_DIR / "datasets/frames"            # 帧序列根目录 (./penn/datasets/frames/{video_id}/)
TEMP_VIDEO_DIR = BASE_DIR / "temp_video"     # 临时视频输出目录
TASKS_JSONL = TEMP_VIDEO_DIR / "tasks.jsonl"       # 输出任务文件

FPS = 30                                     # Penn Action 帧率
FRAME_PATTERN = "%06d.jpg"                   # 帧文件名格式 (如 000001.jpg)
LABEL_STUDIO_MEDIA_PREFIX = "/data/local/media"  # Label Studio 本地文件服务前缀
# 你的 DOCUMENT_ROOT 应指向 BASE_DIR，因此视频 URL 为 {PREFIX}/temp_video/{video_id}.mp4

# ========== 辅助函数 ==========
def extract_video_id(video_path: str) -> str:
    """从 video_path (如 'dataset/JumpingJack/1061.avi') 提取视频 ID (如 '1061')"""
    # 去除扩展名，取最后一部分（不含 .avi）
    stem = Path(video_path).stem
    return stem

def video_exists(video_id: str) -> bool:
    """检查临时视频是否已存在"""
    return (TEMP_VIDEO_DIR / f"{video_id}.mp4").exists()

def create_video(video_id: str):
    """使用 ffmpeg 从帧序列合成视频"""
    frame_pattern = FRAMES_DIR / video_id / FRAME_PATTERN
    output_path = TEMP_VIDEO_DIR / f"{video_id}.mp4"
    
    # 检查帧序列目录是否存在
    if not (FRAMES_DIR / video_id).is_dir():
        raise FileNotFoundError(f"帧序列目录不存在: {FRAMES_DIR / video_id}")
    
    # 检查是否有任何帧文件（简单检查第一个帧）
    first_frame = FRAMES_DIR / video_id / FRAME_PATTERN.replace("*", "1").replace("%06d", "000001")
    if not first_frame.exists():
        # 尝试查找任何 jpg 文件
        any_jpg = list((FRAMES_DIR / video_id).glob("*.jpg"))
        if not any_jpg:
            raise FileNotFoundError(f"在 {FRAMES_DIR / video_id} 中没有找到 .jpg 文件")
        # 自动推断帧模式：假设数字位数固定，找到最小和最大编号
        # 为简化，仍使用 %06d 模式，但提示用户检查
        print(f"警告: {first_frame} 不存在，但存在其他 jpg，请确认帧命名格式为 %06d.jpg")
    
    # 构建 ffmpeg 命令
    cmd = [
        "ffmpeg", "-y",                     # -y 覆盖输出
        "-framerate", str(FPS),
        "-i", str(frame_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    print(f"正在合成视频: {video_id} -> {output_path}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"完成: {output_path}")

def build_task_jsonl():
    """读取 CSV，分组，生成 tasks.jsonl"""
    # 读取 CSV
    rows_by_video = defaultdict(list)
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row['video_path']
            rows_by_video[video_path].append(row)
    
    # 确保临时视频目录存在
    TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for video_path, rows in rows_by_video.items():
        video_id = extract_video_id(video_path)
        
        # 合成视频（如果不存在）
        if not video_exists(video_id):
            try:
                create_video(video_id)
            except Exception as e:
                print(f"跳过视频 {video_id} 的合成: {e}")
                continue
        
        # 构建预标注列表
        predictions = []
        for row in rows:
            start = float(row['start_time'])
            end = float(row['end_time'])
            label = row['label_name']
            # 跳过无效区间
            if start >= end:
                print(f"警告: {video_path} 中存在 start_time >= end_time 的行，已跳过")
                continue
            predictions.append({
                "id": f"{video_id}_{start}_{end}",
                "from_name": "videoLabels",
                "to_name": "video",
                "type": "timelinelabels",
                "value": {
                    "ranges" : [{"start": int(round(start*30,0)), "end": int(round(end*30,0))}],
                    "timelinelabels":[label]
                }
            })
        
        # 构建任务 data 字段
        # 视频 URL（Label Studio 本地文件服务）
        video_url = f"/data/local-files/?d=temp_video/{video_id}.mp4"
        # 从第一行提取公共元数据
        first_row = rows[0]
        task_data = {
            "video_url": video_url,
            "original_video_path": video_path,
            "split": first_row.get('split', ''),
            "assignee": first_row.get('assignee', ''),
            "source_file": first_row.get('source_file', ''),
            "label_id": first_row.get('label_id', ''),
        }
        
        task = {
            "data": task_data,
            "predictions": [{
                "result": predictions
            }]
        }
        tasks.append(task)
    
    # 写入 tasks.jsonl
    with open(TASKS_JSONL, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"已生成 {len(tasks)} 个任务，保存至 {TASKS_JSONL}")

if __name__ == "__main__":
    build_task_jsonl()