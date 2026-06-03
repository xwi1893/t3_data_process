"""
feature_writer.py
特征输出
"""

import json
import os
from typing import Dict, List
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCENE_TYPE_NAMES


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def write_features(
    samples: List[dict],
    output_dir: str,
) -> str:
    """将所有采样条目特征写入 JSON 文件

    Args:
        samples: 包含特征的采样条目列表
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    for s in samples:
        if not s.get('confirmed', False):
            continue

        scene_type = s.get('scene_type', 0)
        scene_en = SCENE_TYPE_NAMES.get(scene_type, f"scene{scene_type}")

        # pb_files: 兼容新版(单帧)和旧版(多帧)
        pb_files = s.get('pb_files') or [s.get('pb_file', '')]

        feature_entry = {
            "driver_id": s.get('driver_id', ''),
            "scene_type": scene_type,
            "scene_name": scene_en,
            "sample_id": s.get('sample_id', s.get('segment_id', 0)),
            "directory_key": s.get('dir_key', ''),
            "pb_file": s.get('pb_file', ''),
            "pb_files": pb_files,
            "timestamp_ns": s.get('timestamp_ns', 0),
        }

        # 通用特征
        if 'general_features' in s:
            feature_entry["general"] = s['general_features']

        # 场景专用特征
        if 'scene_features' in s:
            feature_entry[scene_en] = s['scene_features']

        # 检测指标
        if 'detection' in s:
            feature_entry["detection_metrics"] = s['detection'].get('metrics', {})

        # 合规指标
        if 'compliance' in s:
            feature_entry["compliance_metrics"] = s['compliance'].get('metrics', {})

        all_features.append(feature_entry)

    output_path = os.path.join(output_dir, "scene_features.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"[output] 特征文件已写入: {output_path}")
    print(f"  共 {len(all_features)} 个已确认采样条目的特征")

    return output_path
