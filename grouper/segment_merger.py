"""
segment_merger.py
合并连续帧为场景片段 + 时间窗口扩展
"""

from typing import Dict, List, Optional, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGE_PARAMS, WINDOW_EXTENSION
from data_loader.index_loader import extract_timestamp_from_pb_filename


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


def extend_time_window(
    segment: dict,
    frame_scene: dict,
    all_pb_files_sorted: List[str],
    max_extend_seconds: float = WINDOW_EXTENSION["max_extend_seconds"],
    stop_labels: Set[str] = WINDOW_EXTENSION["stop_labels"],
) -> dict:
    """使用 frame_scene.json 扩展场景片段的时间窗口

    从片段边界向前/后扩展，直到:
    - 达到最大扩展秒数
    - 遇到 stop_labels 中的标签 (如 error_data)
    - 到达文件列表边界

    Args:
        segment: 场景片段 dict
        frame_scene: {pb_filename: [labels]} 全部帧标签
        all_pb_files_sorted: 该目录下所有pb文件名(按时间排序)
        max_extend_seconds: 最大扩展秒数
        stop_labels: 停止扩展的标签集合

    Returns:
        更新后的 segment (pb_files 和 timestamp 已扩展)
    """
    if not all_pb_files_sorted:
        return segment

    max_extend_ns = int(max_extend_seconds * 1e9)
    pb_set = set(segment['pb_files'])
    start_ts = segment['start_timestamp_ns']
    end_ts = segment['end_timestamp_ns']

    # 构建文件名到排序索引的映射
    file_to_idx = {f: i for i, f in enumerate(all_pb_files_sorted)}

    # 找到当前片段在排序列表中的边界
    first_file = segment['pb_files'][0]
    last_file = segment['pb_files'][-1]
    first_idx = file_to_idx.get(first_file, 0)
    last_idx = file_to_idx.get(last_file, len(all_pb_files_sorted) - 1)

    # 向前扩展
    new_first_idx = first_idx
    for i in range(first_idx - 1, -1, -1):
        fname = all_pb_files_sorted[i]
        ts = extract_timestamp_from_pb_filename(fname)
        if start_ts - ts > max_extend_ns:
            break
        labels = frame_scene.get(fname, [])
        if any(l in stop_labels for l in labels):
            break
        pb_set.add(fname)
        new_first_idx = i

    # 向后扩展
    new_last_idx = last_idx
    for i in range(last_idx + 1, len(all_pb_files_sorted)):
        fname = all_pb_files_sorted[i]
        ts = extract_timestamp_from_pb_filename(fname)
        if ts - end_ts > max_extend_ns:
            break
        labels = frame_scene.get(fname, [])
        if any(l in stop_labels for l in labels):
            break
        pb_set.add(fname)
        new_last_idx = i

    # 重建有序的 pb_files 列表
    extended_files = [
        all_pb_files_sorted[i]
        for i in range(new_first_idx, new_last_idx + 1)
        if all_pb_files_sorted[i] in pb_set
    ]

    if len(extended_files) > len(segment['pb_files']):
        old_count = segment['frame_count']
        segment['pb_files'] = extended_files
        segment['start_timestamp_ns'] = extract_timestamp_from_pb_filename(
            extended_files[0]
        )
        segment['end_timestamp_ns'] = extract_timestamp_from_pb_filename(
            extended_files[-1]
        )
        segment['duration_sec'] = (
            (segment['end_timestamp_ns'] - segment['start_timestamp_ns']) / 1e9
        )
        segment['frame_count'] = len(extended_files)
        print(f"[merger] 时间窗口扩展: {old_count} -> {len(extended_files)} 帧, "
              f"时长 {segment['duration_sec']:.1f}s")

    return segment


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
