"""
index_loader.py
加载并关联所有索引文件:
- driver_split.json: 驾驶员 → 日期 → 车辆 映射
- date_split.json: 目录key → 云端路径
- frame_scene.json: pb文件名 → 场景标签数组
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional


def load_json(path: str) -> dict:
    """通用 JSON 加载"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_driver_split(path: str) -> dict:
    """加载 driver_split.json
    返回: {driver_id: {date: {vehicle_id: {}}}}
    """
    data = load_json(path)
    print(f"[index_loader] 加载 driver_split: {len(data)} 位驾驶员")
    return data


def build_driver_reverse_index(driver_split: dict) -> Dict[Tuple[str, str], str]:
    """构建反向索引: (date_str, vehicle_id) -> driver_id

    遍历 {driver_id: {date: {vehicle_id: {}}}}
    输出 {(date, vehicle_id): driver_id}

    date 格式在 driver_split 中为 YYYY-MM-DD
    """
    reverse_idx = {}
    for driver_id, dates in driver_split.items():
        for date_str, vehicles in dates.items():
            for vehicle_id in vehicles.keys():
                key = (date_str, vehicle_id)
                if key in reverse_idx:
                    print(f"[index_loader] 警告: ({date_str}, {vehicle_id}) "
                          f"映射到多个驾驶员: {reverse_idx[key]}, {driver_id}")
                reverse_idx[key] = driver_id
    print(f"[index_loader] 反向索引构建完成: {len(reverse_idx)} 条记录")
    return reverse_idx


def load_date_split(path: str) -> Dict[str, str]:
    """加载 date_split.json
    返回: {dir_key: cloud_path}
    """
    data = load_json(path)
    print(f"[index_loader] 加载 date_split: {len(data)} 条记录")
    return data


def load_data_batch(path: str) -> List[str]:
    """加载 data_batch.json
    返回: batch 目录路径列表
    """
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"data_batch.json 应为列表格式: {path}")
    print(f"[index_loader] 加载 data_batch: {len(data)} 个 batch 目录")
    return data


def load_merged_date_split(batch_dirs: List[str]) -> Dict[str, str]:
    """加载并合并所有 batch 的 date_split.json

    Args:
        batch_dirs: batch 目录路径列表

    Returns:
        合并后的 {dir_key: cloud_path}
    """
    merged = {}
    for batch_dir in batch_dirs:
        ds_path = os.path.join(batch_dir, "date_split.json")
        if not os.path.exists(ds_path):
            print(f"[index_loader] 警告: date_split.json 不存在: {ds_path}")
            continue
        ds = load_json(ds_path)
        merged.update(ds)
    print(f"[index_loader] 合并 date_split: {len(merged)} 条记录 (来自 {len(batch_dirs)} 个 batch)")
    return merged


def load_frame_scene(path: str) -> Dict[str, List[str]]:
    """加载单个 frame_scene.json
    返回: {pb_filename: [labels]}
    """
    data = load_json(path)
    print(f"[index_loader] 加载 frame_scene: {len(data)} 帧")
    return data


def parse_directory_key(dir_key: str) -> Tuple[str, str]:
    """从目录key中提取日期和车辆ID

    输入: '2026_02_27_22_36_43_dlp-LDP41B96XSD165867'
    输出: ('2026-02-27', 'LDP41B96XSD165867')
    """
    # 匹配: YYYY_MM_DD_HH_MM_SS_dlp-VEHICLE_ID
    m = re.match(
        r'^(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}_dlp-(.+)$',
        dir_key
    )
    if not m:
        raise ValueError(f"无法解析目录key: {dir_key}")

    date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    vehicle_id = m.group(4)
    return date_str, vehicle_id


def lookup_driver_id(
    dir_key: str,
    driver_reverse_index: Dict[Tuple[str, str], str]
) -> Optional[str]:
    """根据目录key查找驾驶员ID

    从 dir_key 中提取 (date, vehicle_id)，
    在反向索引中查找对应的 driver_id
    """
    try:
        date_str, vehicle_id = parse_directory_key(dir_key)
    except ValueError:
        return None

    return driver_reverse_index.get((date_str, vehicle_id))


def extract_timestamp_from_pb_filename(pb_filename: str) -> int:
    """从 pb 文件名提取纳秒时间戳

    输入: '1771887926405091968.pb'
    输出: 1771887926405091968
    """
    name = pb_filename
    if name.endswith('.pb'):
        name = name[:-3]
    return int(name)
