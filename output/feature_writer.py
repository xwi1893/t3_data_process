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
    segments: List[dict],
    output_dir: str,
) -> str:
    """将所有片段特征写入 JSON 文件

    Args:
        segments: 包含特征的片段列表
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    for seg in segments:
        if not seg.get('confirmed', False):
            continue

        scene_type = seg.get('scene_type', 0)
        scene_en = SCENE_TYPE_NAMES.get(scene_type, f"scene{scene_type}")

        feature_entry = {
            "driver_id": seg.get('driver_id', ''),
            "scene_type": scene_type,
            "scene_name": scene_en,
            "segment_id": seg.get('segment_id', 0),
            "directory_key": seg.get('dir_key', ''),
            "pb_files": seg.get('pb_files', []),
            "duration_sec": seg.get('duration_sec', 0),
            "frame_count": seg.get('frame_count', 0),
        }

        # 通用特征
        if 'general_features' in seg:
            feature_entry["general"] = seg['general_features']

        # 场景专用特征
        if 'scene_features' in seg:
            feature_entry[scene_en] = seg['scene_features']

        # 检测指标
        if 'detection' in seg:
            feature_entry["detection_metrics"] = seg['detection'].get('metrics', {})

        # 合规指标
        if 'compliance' in seg:
            feature_entry["compliance_metrics"] = seg['compliance'].get('metrics', {})

        all_features.append(feature_entry)

    output_path = os.path.join(output_dir, "scene_features.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"[output] 特征文件已写入: {output_path}")
    print(f"  共 {len(all_features)} 个已确认片段的特征")

    return output_path
