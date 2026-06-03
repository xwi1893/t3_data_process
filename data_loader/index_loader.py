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
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def _load_single_date_split(ds_path: str) -> Tuple[str, dict]:
    """加载单个 date_split.json (多进程 worker)

    Returns:
        (path, data) 元组; 加载失败时 data 为空 dict
    """
    if not os.path.exists(ds_path):
        return ds_path, {}
    try:
        with open(ds_path, 'r', encoding='utf-8') as f:
            return ds_path, json.load(f)
    except Exception:
        return ds_path, {}


def load_merged_date_split(
    batch_dirs: List[str], use_cache: bool = True, max_workers: int = 0,
) -> Dict[str, str]:
    """加载并合并所有 batch 的 date_split.json

    支持 pickle 缓存: 若 batch_dirs 未变化，直接加载缓存跳过 JSON 解析
    支持多进程并行加载: max_workers > 1 时并行读取 date_split.json

    Args:
        batch_dirs: batch 目录路径列表
        use_cache: 是否启用缓存
        max_workers: 并行加载 worker 数 (0=串行)

    Returns:
        合并后的 {dir_key: cloud_path}
    """
    # 缓存逻辑: 根据 batch_dirs 内容生成 hash 作为缓存 key
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
    cache_key = hashlib.md5("|".join(sorted(batch_dirs)).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"date_split_{cache_key}.pkl")

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                merged = pickle.load(f)
            print(f"[index_loader] 从缓存加载 date_split: {len(merged)} 条记录")
            return merged
        except Exception as e:
            print(f"[index_loader] 缓存加载失败，重新解析: {e}")

    # 收集待加载的 date_split.json 路径
    ds_paths = []
    for batch_dir in batch_dirs:
        ds_path = os.path.join(batch_dir, "date_split.json")
        ds_paths.append(ds_path)

    # 并行或串行加载
    results = []
    if max_workers > 1 and len(ds_paths) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load_single_date_split, p): p for p in ds_paths}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for p in ds_paths:
            results.append(_load_single_date_split(p))

    # 合并 (主进程)
    merged = {}
    skipped = 0
    for ds_path, ds in results:
        if not ds:
            skipped += 1
            continue
        merged.update(ds)

    total = len(results)
    if skipped:
        print(f"[index_loader] 警告: {skipped}/{total} 个 date_split.json 加载失败")
    print(f"[index_loader] 合并 date_split: {len(merged)} 条记录 (来自 {total - skipped}/{total} 个 batch)")

    # 写入缓存
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[index_loader] date_split 缓存已保存: {cache_path}")
        except Exception as e:
            print(f"[index_loader] 缓存保存失败: {e}")

    return merged


def load_frame_scene(path: str) -> Dict[str, List[str]]:
    """加载单个 frame_scene.json
    返回: {pb_filename: [labels]}
    """
    data = load_json(path)
    return data


def check_label_prescreen(
    dir_path: str,
    target_labels: set,
) -> bool:
    """通过 label_count.json 快速预筛目录是否包含目标标签

    若 label_count.json 不存在则返回 True (保守策略，不跳过)

    Args:
        dir_path: dir_key 目录路径
        target_labels: 目标标签集合 (如 {"brake2stop", "at_intersection", ...})

    Returns:
        True: 可能包含目标标签，需要加载 frame_scene.json
        False: 确定不含目标标签，可跳过
    """
    lc_path = os.path.join(dir_path, "label_count.json")
    if not os.path.exists(lc_path):
        return True  # 无预筛文件，保守加载

    try:
        with open(lc_path, 'r', encoding='utf-8') as f:
            label_count = json.load(f)
        # label_count 格式: {"label_name": count, ...}
        for label in target_labels:
            if label_count.get(label, 0) > 0:
                return True
        return False
    except Exception:
        return True  # 解析失败，保守加载


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
