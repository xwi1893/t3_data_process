"""
training_manifest.py
训练数据 JSON 清单生成
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCENE_TYPE_NAMES


def build_manifest(segments: List[dict]) -> dict:
    """从已处理片段构建训练清单

    Args:
        segments: 所有已处理片段 (含 confirmed, compliance, features)

    Returns:
        训练清单字典
    """
    # 统计
    drivers = set()
    total_segments = 0

    manifest = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_drivers": 0,
            "total_segments": 0,
            "confirmed_segments": 0,
            "compliant_segments": 0,
        },
        "drivers": {},
    }

    for seg in segments:
        driver_id = seg.get('driver_id', 'unknown')
        scene_type = seg.get('scene_type', 0)
        scene_en = SCENE_TYPE_NAMES.get(scene_type, f"scene{scene_type}")

        confirmed = seg.get('confirmed', False)
        compliant = seg.get('compliance', {}).get('passed', False)

        drivers.add(driver_id)

        if driver_id not in manifest["drivers"]:
            manifest["drivers"][driver_id] = {}

        if scene_en not in manifest["drivers"][driver_id]:
            manifest["drivers"][driver_id][scene_en] = []

        entry = {
            "segment_id": seg.get('segment_id', 0),
            "directory_key": seg.get('dir_key', ''),
            "cloud_path": seg.get('cloud_path', ''),
            "pb_files": seg.get('pb_files', []),
            "frame_range_ns": [
                seg.get('start_timestamp_ns', 0),
                seg.get('end_timestamp_ns', 0),
            ],
            "duration_sec": round(seg.get('duration_sec', 0), 2),
            "frame_count": seg.get('frame_count', 0),
            "confirmed": confirmed,
            "compliance_passed": compliant,
        }

        # 检测指标
        if 'detection' in seg:
            entry["detection"] = {
                "method": seg['detection'].get('method', ''),
                "metrics": seg['detection'].get('metrics', {}),
            }

        # 合规指标
        if 'compliance' in seg:
            entry["compliance"] = seg['compliance']

        manifest["drivers"][driver_id][scene_en].append(entry)

        if confirmed:
            manifest["metadata"]["confirmed_segments"] += 1
        if compliant:
            manifest["metadata"]["compliant_segments"] += 1
        total_segments += 1

    manifest["metadata"]["total_drivers"] = len(drivers)
    manifest["metadata"]["total_segments"] = total_segments

    return manifest


def write_manifest(
    segments: List[dict],
    output_path: str,
) -> str:
    """生成并写入训练清单 JSON

    Args:
        segments: 所有已处理片段
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    manifest = build_manifest(segments)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[output] 训练清单已写入: {output_path}")
    print(f"  驾驶员数: {manifest['metadata']['total_drivers']}")
    print(f"  总片段数: {manifest['metadata']['total_segments']}")
    print(f"  已确认: {manifest['metadata']['confirmed_segments']}")
    print(f"  合规: {manifest['metadata']['compliant_segments']}")

    return output_path
