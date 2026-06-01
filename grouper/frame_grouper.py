"""
frame_grouper.py
按 (driver_id, scene_type) 对帧进行分组
"""

from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.index_loader import (
    extract_timestamp_from_pb_filename,
    lookup_driver_id,
    parse_directory_key,
)
from grouper.scene_classifier import classify_frame_scene_all


def group_frames(
    frame_scene: dict,
    dir_key: str,
    driver_reverse_index: dict,
    date_split: dict,
) -> Dict[Tuple[str, int], List[dict]]:
    """对单个目录的帧按 (driver_id, scene_type) 分组

    Args:
        frame_scene: {pb_filename: [labels]}
        dir_key: 目录key (如 '2026_02_27_22_36_43_dlp-LDP41B96XSD165867')
        driver_reverse_index: (date, vehicle_id) -> driver_id 反向索引
        date_split: {dir_key: cloud_path}

    Returns:
        {(driver_id, scene_type): [frame_info, ...]}
        其中 frame_info = {
            'pb_file': str,
            'timestamp_ns': int,
            'scene_type': int,
            'scene_name_cn': str,
            'scene_name_en': str,
            'labels': list,
            'driver_id': str,
            'dir_key': str,
            'cloud_path': str,
        }
    """
    # 查找驾驶员ID
    driver_id = lookup_driver_id(dir_key, driver_reverse_index)
    if driver_id is None:
        print(f"[grouper] 警告: 未找到 {dir_key} 对应的驾驶员ID，跳过")
        return {}

    cloud_path = date_split.get(dir_key, "")

    # 标签分类
    classified = classify_frame_scene_all(frame_scene)
    if not classified:
        print(f"[grouper] {dir_key}: 无有效场景帧")
        return {}

    # 按 (driver_id, scene_type) 分组
    groups: Dict[Tuple[str, int], List[dict]] = {}

    for pb_file, (scene_type, name_cn, name_en) in classified.items():
        ts_ns = extract_timestamp_from_pb_filename(pb_file)

        frame_info = {
            'pb_file': pb_file,
            'timestamp_ns': ts_ns,
            'scene_type': scene_type,
            'scene_name_cn': name_cn,
            'scene_name_en': name_en,
            'labels': frame_scene.get(pb_file, []),
            'driver_id': driver_id,
            'dir_key': dir_key,
            'cloud_path': cloud_path,
        }

        key = (driver_id, scene_type)
        if key not in groups:
            groups[key] = []
        groups[key].append(frame_info)

    # 每组内按时间戳排序
    for key in groups:
        groups[key].sort(key=lambda f: f['timestamp_ns'])

    print(f"[grouper] {dir_key} (driver={driver_id}): "
          f"{len(classified)} 帧分为 {len(groups)} 组")

    return groups
