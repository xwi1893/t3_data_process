"""
segment_merger.py
合并连续帧为场景片段
"""

from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGE_PARAMS


def merge_into_segments(
    frames: List[dict],
    max_gap_seconds: float = MERGE_PARAMS["max_gap_seconds"],
    min_segment_duration: float = MERGE_PARAMS["min_segment_duration"],
) -> List[dict]:
    """合并连续帧为候选场景片段

    连续性判断: 相邻帧时间戳差 <= max_gap_seconds (转换为纳秒)
    过滤: 时长 < min_segment_duration 的片段

    Args:
        frames: 同一 (driver_id, scene_type) 的帧列表，已按时间排序
        max_gap_seconds: 最大允许帧间隔(秒)
        min_segment_duration: 最短片段时长(秒)

    Returns:
        场景片段列表
    """
    if not frames:
        return []

    max_gap_ns = int(max_gap_seconds * 1e9)
    min_duration_ns = int(min_segment_duration * 1e9)

    segments = []
    current_run = [frames[0]]

    for i in range(1, len(frames)):
        prev_ts = frames[i - 1]['timestamp_ns']
        curr_ts = frames[i]['timestamp_ns']

        if curr_ts - prev_ts <= max_gap_ns:
            current_run.append(frames[i])
        else:
            # 间隔过大，结束当前片段
            segments.append(current_run)
            current_run = [frames[i]]

    # 最后一个片段
    if current_run:
        segments.append(current_run)

    # 构建片段结构并过滤过短片段
    result = []
    for run in segments:
        start_ts = run[0]['timestamp_ns']
        end_ts = run[-1]['timestamp_ns']
        duration_ns = end_ts - start_ts

        if duration_ns < min_duration_ns:
            continue

        first = run[0]
        segment = {
            'driver_id': first['driver_id'],
            'scene_type': first['scene_type'],
            'scene_name': first['scene_name_cn'],
            'scene_name_en': first['scene_name_en'],
            'dir_key': first['dir_key'],
            'cloud_path': first['cloud_path'],
            'pb_files': [f['pb_file'] for f in run],
            'start_timestamp_ns': start_ts,
            'end_timestamp_ns': end_ts,
            'duration_sec': duration_ns / 1e9,
            'frame_count': len(run),
            'confirmed': False,
            'confirm_reason': '',
        }
        result.append(segment)

    print(f"[merger] {len(segments)} 个原始片段 -> "
          f"{len(result)} 个有效片段 (过滤 <{min_segment_duration}s)")

    return result


def group_and_merge(
    frame_scene: dict,
    dir_key: str,
    driver_reverse_index: dict,
    date_split: dict,
) -> List[dict]:
    """完整的分组+合并流程

    对单个目录执行: 分组 -> 合并 -> 返回所有候选片段

    Args:
        frame_scene: {pb_filename: [labels]}
        dir_key: 目录key
        driver_reverse_index: 反向索引
        date_split: 目录key到云端路径

    Returns:
        所有候选场景片段列表
    """
    from grouper.frame_grouper import group_frames

    groups = group_frames(frame_scene, dir_key, driver_reverse_index, date_split)
    if not groups:
        return []

    all_segments = []
    for (driver_id, scene_type), frames in groups.items():
        segments = merge_into_segments(frames)

        # 分配 segment_id
        for idx, seg in enumerate(segments):
            seg['segment_id'] = idx

        all_segments.extend(segments)

    scene_names = {1: "路口停车", 2: "起步", 3: "跟车", 4: "跟停", 5: "变道"}
    type_counts = {}
    for seg in all_segments:
        st = seg['scene_type']
        type_counts[st] = type_counts.get(st, 0) + 1

    print(f"[merger] {dir_key}: 共 {len(all_segments)} 个候选片段")
    for st, count in sorted(type_counts.items()):
        print(f"  Scene {st} ({scene_names.get(st, '?')}): {count} 个")

    return all_segments
