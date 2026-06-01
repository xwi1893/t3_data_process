"""
detector.py
场景检测调度器: 根据候选场景类型分发到对应检测函数
"""

from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.scene_detectors import (
    extract_segment_features,
    confirm_following_stop,
    confirm_intersection_stop,
    confirm_empty_start,
    confirm_following_vehicle,
    confirm_lane_change,
)


# 场景类型 -> 确认函数映射
_CONFIRM_FUNCS = {
    1: confirm_intersection_stop,
    2: confirm_empty_start,
    3: confirm_following_vehicle,
    4: confirm_following_stop,
    5: confirm_lane_change,
}

_SCENE_NAMES = {
    1: "路口停车",
    2: "起步",
    3: "跟车",
    4: "跟停",
    5: "变道",
}


def confirm_scene(
    segment: dict,
    frame_data: Dict[str, dict],
) -> dict:
    """对候选片段运行检测算法，返回确认结果

    Args:
        segment: 候选场景片段 (含 scene_type, pb_files 等)
        frame_data: {pb_filename: normalized_frame}

    Returns:
        更新后的 segment，增加 confirmed, confirm_reason, detection 字段
    """
    scene_type = segment['scene_type']
    scene_name = _SCENE_NAMES.get(scene_type, f"未知({scene_type})")

    confirm_func = _CONFIRM_FUNCS.get(scene_type)
    if confirm_func is None:
        segment['confirmed'] = False
        segment['confirm_reason'] = f"无对应的确认算法: scene_type={scene_type}"
        return segment

    # 提取特征
    features = extract_segment_features(frame_data, segment['pb_files'])
    if features is None:
        segment['confirmed'] = False
        segment['confirm_reason'] = "特征提取失败 (有效帧不足)"
        return segment

    # 运行确认算法
    result = confirm_func(features)
    segment['confirmed'] = result['confirmed']
    segment['confirm_reason'] = result.get('reason', '')
    segment['detection'] = {
        'method': confirm_func.__name__,
        'metrics': result.get('metrics', {}),
    }
    segment['features'] = features  # 保存特征供后续使用

    status = "通过" if result['confirmed'] else "拒绝"
    print(f"[detector] {scene_name} #{segment.get('segment_id', '?')}: "
          f"{status} - {result.get('reason', '')}")

    return segment


def confirm_all_segments(
    segments: List[dict],
    all_frame_data: Dict[str, Dict[str, dict]],
) -> List[dict]:
    """对所有候选片段进行二次确认

    Args:
        segments: 所有候选片段列表
        all_frame_data: {dir_key: {pb_filename: normalized_frame}}

    Returns:
        更新后的片段列表 (每个片段增加 confirmed 等字段)
    """
    confirmed_count = 0
    rejected_count = 0

    for segment in segments:
        dir_key = segment['dir_key']
        dir_frames = all_frame_data.get(dir_key, {})

        if not dir_frames:
            segment['confirmed'] = False
            segment['confirm_reason'] = f"无帧数据: {dir_key}"
            rejected_count += 1
            continue

        confirm_scene(segment, dir_frames)

        if segment['confirmed']:
            confirmed_count += 1
        else:
            rejected_count += 1

    print(f"\n[detector] 二次确认完成: "
          f"{confirmed_count} 通过, {rejected_count} 拒绝, "
          f"共 {len(segments)} 个片段")

    return segments
