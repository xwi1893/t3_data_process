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
    sample: dict,
    frame_data: Dict[str, dict],
) -> dict:
    """对单个采样帧运行检测算法

    支持单帧输入: frame_data 仅含 1 个 pb 文件，
    时序信息从该 pb 的内部数据（历史轨迹 + 未来轨迹）获取。

    Args:
        sample: 采样条目 (含 scene_type, pb_file 等)
        frame_data: {pb_filename: normalized_frame}

    Returns:
        更新后的 sample，增加 confirmed, confirm_reason, detection 字段
    """
    scene_type = sample['scene_type']
    scene_name = _SCENE_NAMES.get(scene_type, f"未知({scene_type})")

    pb_path = os.path.join(sample.get('cloud_path', ''), sample.get('pb_file', ''))

    confirm_func = _CONFIRM_FUNCS.get(scene_type)
    if confirm_func is None:
        sample['confirmed'] = False
        sample['confirm_reason'] = f"无对应的确认算法: scene_type={scene_type}"
        print(f"[detector] {scene_name} #{sample.get('sample_id', '?')}: 拒绝 - {sample['confirm_reason']} | pb: {pb_path}")
        return sample

    # pb_files: 兼容新版(单帧)和旧版(多帧)
    pb_files = sample.get('pb_files') or [sample['pb_file']]

    # 提取特征
    features = extract_segment_features(frame_data, pb_files)
    if features is None:
        sample['confirmed'] = False
        sample['confirm_reason'] = "特征提取失败 (有效帧不足)"
        print(f"[detector] {scene_name} #{sample.get('sample_id', '?')}: 拒绝 - {sample['confirm_reason']} | pb: {pb_path}")
        return sample

    # 运行确认算法
    result = confirm_func(features)
    sample['confirmed'] = result['confirmed']
    sample['confirm_reason'] = result.get('reason', '')
    sample['detection'] = {
        'method': confirm_func.__name__,
        'metrics': result.get('metrics', {}),
    }
    sample['features'] = features  # 保存特征供后续使用

    status = "通过" if result['confirmed'] else "拒绝"
    sample_id = sample.get('sample_id', sample.get('segment_id', '?'))
    print(f"[detector] {scene_name} #{sample_id}: "
          f"{status} - {result.get('reason', '')}"
          f"{' | pb: ' + pb_path if not result['confirmed'] else ''}")

    return sample


def confirm_all_samples(
    samples: List[dict],
    all_frame_data: Dict[str, Dict[str, dict]],
) -> List[dict]:
    """对所有采样条目进行二次确认

    Args:
        samples: 所有采样条目列表
        all_frame_data: {dir_key: {pb_filename: normalized_frame}}

    Returns:
        更新后的采样条目列表
    """
    confirmed_count = 0
    rejected_count = 0

    for sample in samples:
        dir_key = sample['dir_key']
        dir_frames = all_frame_data.get(dir_key, {})

        if not dir_frames:
            sample['confirmed'] = False
            sample['confirm_reason'] = f"无帧数据: {dir_key}"
            rejected_count += 1
            continue

        confirm_scene(sample, dir_frames)

        if sample['confirmed']:
            confirmed_count += 1
        else:
            rejected_count += 1

    print(f"\n[detector] 二次确认完成: "
          f"{confirmed_count} 通过, {rejected_count} 拒绝, "
          f"共 {len(samples)} 个采样")

    return samples
