"""
segment_merger.py
识别连续场景片段 + 均匀采样 k 帧

旧版: 合并连续帧为多帧片段 (merge_into_segments)
新版: 识别连续场景 → 均匀采样 k 帧 (group_and_sample)
"""

from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGE_PARAMS, SAMPLE_PARAMS


def identify_continuous_runs(
    frames: List[dict],
    max_gap_seconds: float = MERGE_PARAMS["max_gap_seconds"],
) -> List[List[dict]]:
    """识别连续场景片段（不合并，只分段）

    连续性判断: 相邻帧时间戳差 <= max_gap_seconds (转换为纳秒)

    Args:
        frames: 同一 (driver_id, scene_type) 的帧列表，已按时间排序
        max_gap_seconds: 最大允许帧间隔(秒)

    Returns:
        连续帧列表的列表 [[f0, f1, ...], [f5, f6, ...], ...]
    """
    if not frames:
        return []

    max_gap_ns = int(max_gap_seconds * 1e9)

    runs = []
    current_run = [frames[0]]

    for i in range(1, len(frames)):
        prev_ts = frames[i - 1]['timestamp_ns']
        curr_ts = frames[i]['timestamp_ns']

        if curr_ts - prev_ts <= max_gap_ns:
            current_run.append(frames[i])
        else:
            runs.append(current_run)
            current_run = [frames[i]]

    if current_run:
        runs.append(current_run)

    return runs


def sample_k_frames(
    run: List[dict],
    k: int = SAMPLE_PARAMS["k_frames_default"],
) -> List[dict]:
    """从连续片段中均匀采样 k 帧

    采样策略: 等间隔，覆盖首、中、尾

    Args:
        run: 连续帧列表，已按时间排序
        k: 采样帧数

    Returns:
        采样条目列表，每个条目 = 单个帧的信息 + sample_id
    """
    n = len(run)
    if n <= k:
        indices = list(range(n))
    else:
        # 等间隔采样: 首尾 + 中间均匀分布
        indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        # 去重（极端情况下 round 可能产生重复）
        indices = sorted(set(indices))

    samples = []
    for idx in indices:
        frame = run[idx]
        sample = {
            'driver_id': frame['driver_id'],
            'scene_type': frame['scene_type'],
            'scene_name': frame['scene_name_cn'],
            'scene_name_en': frame['scene_name_en'],
            'dir_key': frame['dir_key'],
            'cloud_path': frame['cloud_path'],
            'pb_file': frame['pb_file'],
            'timestamp_ns': frame['timestamp_ns'],
            'labels': frame.get('labels', []),
            'confirmed': False,
            'confirm_reason': '',
        }
        samples.append(sample)

    return samples


def _get_k_for_scene(scene_type: int) -> int:
    """根据场景类型获取对应的采样帧数 k"""
    per_scene = SAMPLE_PARAMS.get("k_frames_per_scene", {})
    return per_scene.get(scene_type, SAMPLE_PARAMS["k_frames_default"])


def group_and_sample(
    frame_scene: dict,
    dir_key: str,
    driver_reverse_index: dict,
    date_split: dict,
) -> List[dict]:
    """完整的分组 + 连续片段识别 + k 帧采样

    对单个目录执行: 分组 → 识别连续片段 → 按场景类型均匀采样 k 帧
    k 值由 config.SAMPLE_PARAMS["k_frames_per_scene"] 按场景类型决定，
    未配置的场景使用 SAMPLE_PARAMS["k_frames_default"]

    Args:
        frame_scene: {pb_filename: [labels]}
        dir_key: 目录key
        driver_reverse_index: 反向索引
        date_split: 目录key到云端路径

    Returns:
        所有采样条目列表
    """
    from grouper.frame_grouper import group_frames

    groups = group_frames(frame_scene, dir_key, driver_reverse_index, date_split)
    if not groups:
        return []

    all_samples = []
    for (driver_id, scene_type), frames in groups.items():
        k = _get_k_for_scene(scene_type)
        runs = identify_continuous_runs(frames)

        for run in runs:
            samples = sample_k_frames(run, k=k)
            all_samples.extend(samples)

    # 分配 sample_id
    for idx, sample in enumerate(all_samples):
        sample['sample_id'] = idx

    scene_names = {1: "路口停车", 2: "起步", 3: "跟车", 4: "跟停", 5: "变道"}
    type_counts = {}
    for s in all_samples:
        st = s['scene_type']
        type_counts[st] = type_counts.get(st, 0) + 1

    print(f"[sampler] {dir_key}: 共 {len(all_samples)} 个采样条目")
    for st, count in sorted(type_counts.items()):
        k_used = _get_k_for_scene(st)
        print(f"  Scene {st} ({scene_names.get(st, '?')}): {count} 个 (k={k_used})")

    return all_samples

