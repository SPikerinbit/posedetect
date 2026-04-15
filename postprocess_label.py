#!/usr/bin/env python3
"""
convert_export.py - 将 Label Studio 导出的 JSON 转换为 CSV
- 从临时视频文件中读取帧率（使用 ffprobe）
- 将 start/end（帧索引）转换为秒
- 边界截断到视频时长内
- 保留原始 CSV 中的 assignee/split/video_path/source_file，confidence 调整为 0.99
- 输出每个标注片段为一行
"""

import json
import csv
import subprocess
from pathlib import Path
from collections import defaultdict

# ========== 配置 ==========
BASE_DIR = Path("./penn").resolve()
INPUT_CSV = BASE_DIR / "input.csv"
EXPORT_JSON = BASE_DIR / "export.json"
TEMP_VIDEO_DIR = BASE_DIR / "temp_video"
OUTPUT_CSV = BASE_DIR / "labeled_output.csv"

LABEL_TO_ID = {
    "push_up": 1,
    "jumping_jack": 2
}

# ========== 视频元数据读取 ==========
def get_video_fps_and_frames(video_path: Path):
    """
    使用 ffprobe 获取视频的帧率 (fps) 和总帧数
    返回 (fps, total_frames)
    """
    # 获取帧率
    cmd_fps = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    fps_str = subprocess.check_output(cmd_fps, text=True).strip()
    # 帧率可能是分数形式如 "30000/1001"，需转换为浮点数
    if '/' in fps_str:
        num, den = map(float, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)
    
    # 获取总帧数
    cmd_frames = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    frames_str = subprocess.check_output(cmd_frames, text=True).strip()
    total_frames = int(frames_str) if frames_str else 0
    
    return fps, total_frames

def get_video_duration(video_path: Path, fps, total_frames):
    """根据帧率和总帧数计算时长（秒），也可用 ffprobe 直接获取，此处用帧数计算更可靠"""
    if total_frames > 0:
        return total_frames / fps
    # 备选：使用 ffprobe 获取时长
    cmd_dur = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    dur_str = subprocess.check_output(cmd_dur, text=True).strip()
    return float(dur_str) if dur_str else 0.0

# ========== 原始数据加载 ==========
def load_original_rows():
    """建立 video_path -> 元数据映射"""
    original = {}
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = row['video_path']
            if vp not in original:
                original[vp] = {
                    'assignee': row['assignee'],
                    'split': row['split'],
                    'video_path': vp,
                    'source_file': row['source_file'],
                }
    return original

# ========== 转换主函数 ==========
def convert():
    original_rows = load_original_rows()
    
    with open(EXPORT_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    output_rows = []
    
    for task in tasks:
        data = task.get('data', {})
        original_video_path = data.get('original_video_path', '')
        if original_video_path not in original_rows:
            print(f"警告: {original_video_path} 不在 input.csv 中，跳过")
            continue
        meta = original_rows[original_video_path]
        
        # 找到对应的临时视频文件
        video_id = Path(original_video_path).stem
        video_file = TEMP_VIDEO_DIR / f"{video_id}.mp4"
        if not video_file.exists():
            print(f"警告: 视频文件 {video_file} 不存在，跳过")
            continue
        
        # 获取视频帧率和总帧数
        try:
            fps, total_frames = get_video_fps_and_frames(video_file)
            duration = get_video_duration(video_file, fps, total_frames)
        except Exception as e:
            print(f"警告: 无法读取 {video_file} 元数据 ({e})，使用默认 FPS=30")
            fps = 30.0
            total_frames = 0
        
        # 遍历所有标注版本
        annotations = task.get('annotations', [])
        anno = annotations[-1]
        results = anno.get('result', [])
        for res in results:
            if res.get('type') != 'timelinelabels':
                continue
            value = res.get('value', {})
            start_frame = value.get('ranges', [])[0].get('start')
            end_frame = value.get('ranges', [])[0].get('end')
            labels = value.get('timelinelabels', [])
            if start_frame is None or end_frame is None or not labels:
                print(res)
                continue
            
            # 确保是整数
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            
            # 边界截断（基于总帧数）
            if total_frames > 0:
                start_frame = max(1, min(start_frame, total_frames - 1))
                end_frame = max(1, min(end_frame, total_frames))
            else:
                # 没有总帧数信息，只保证非负
                start_frame = max(1, start_frame)
                end_frame = max(1, end_frame)
            
            if start_frame >= end_frame:
                continue  # 无效区间
            
            # 帧索引转秒，开始时间向前舍入，结束时间向后舍入
            start_sec = (start_frame - 1) / fps
            end_sec = (end_frame + 1) / fps
            
            label_name = labels[0]
            label_id = LABEL_TO_ID.get(label_name, 0)
            
            output_rows.append({
                'assignee': meta['assignee'],
                'split': meta['split'],
                'video_path': meta['video_path'],
                'label_name': label_name,
                'start_time': start_sec,
                'end_time': end_sec,
                'confidence': '0.99', # 因为是人标的
                'source_type': 'manual_refined',
                'source_file': meta['source_file'],
                'label_id': label_id
            })
    
    # 写入 CSV
    fieldnames = ['assignee', 'split', 'video_path', 'label_name', 'start_time', 'end_time',
                  'confidence', 'source_type', 'source_file', 'label_id']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"转换完成，共 {len(output_rows)} 行，保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    convert()