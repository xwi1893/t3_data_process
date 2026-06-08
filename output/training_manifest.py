"""
training_manifest.py
训练数据 JSON 清单生成
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCENE_TYPE_NAMES


def build_manifest(samples: List[dict]) -> dict:
    """从已处理采样条目构建训练清单

    Args:
        samples: 所有已处理采样条目 (含 confirmed, compliance, features)

    Returns:
        训练清单字典
    """
    drivers = set()
    total_samples = 0

    manifest = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_drivers": 0,
            "total_samples": 0,
            "confirmed_samples": 0,
            "compliant_samples": 0,
        },
        "drivers": {},
    }

    for s in samples:
        driver_id = s.get('driver_id', 'unknown')
        scene_type = s.get('scene_type', 0)
        scene_en = SCENE_TYPE_NAMES.get(scene_type, f"scene{scene_type}")

        confirmed = s.get('confirmed', False)
        compliant = s.get('compliance', {}).get('passed', False)

        drivers.add(driver_id)

        if driver_id not in manifest["drivers"]:
            manifest["drivers"][driver_id] = {}

        if scene_en not in manifest["drivers"][driver_id]:
            manifest["drivers"][driver_id][scene_en] = []

        # pb_files: 兼容新版(单帧)和旧版(多帧)
        pb_files = s.get('pb_files') or [s.get('pb_file', '')]

        entry = {
            "sample_id": s.get('sample_id', s.get('segment_id', 0)),
            "directory_key": s.get('dir_key', ''),
            "cloud_path": s.get('cloud_path', ''),
            "pb_file": s.get('pb_file', ''),
            "pb_files": pb_files,
            "timestamp_ns": s.get('timestamp_ns', 0),
            "confirmed": confirmed,
            "compliance_passed": compliant,
        }

        # 检测指标
        if 'detection' in s:
            entry["detection"] = {
                "method": s['detection'].get('method', ''),
                "metrics": s['detection'].get('metrics', {}),
            }

        # 合规指标
        if 'compliance' in s:
            entry["compliance"] = s['compliance']

        manifest["drivers"][driver_id][scene_en].append(entry)

        if confirmed:
            manifest["metadata"]["confirmed_samples"] += 1
        if compliant:
            manifest["metadata"]["compliant_samples"] += 1
        total_samples += 1

    manifest["metadata"]["total_drivers"] = len(drivers)
    manifest["metadata"]["total_samples"] = total_samples

    return manifest


def write_manifest(
    samples: List[dict],
    output_path: str,
) -> str:
    """流式生成并写入训练清单 JSON

    按驾驶员/场景逐条写入，避免一次性构建大字典

    Args:
        samples: 所有已处理采样条目
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    drivers = set()
    total_samples = 0
    confirmed_samples = 0
    compliant_samples = 0

    # 按 (driver_id, scene_en) 分桶
    driver_scene_entries: Dict[str, Dict[str, list]] = {}

    for s in samples:
        driver_id = s.get('driver_id', 'unknown')
        scene_type = s.get('scene_type', 0)
        scene_en = SCENE_TYPE_NAMES.get(scene_type, f"scene{scene_type}")

        confirmed = s.get('confirmed', False)
        compliant = s.get('compliance', {}).get('passed', False)

        drivers.add(driver_id)
        total_samples += 1
        if confirmed:
            confirmed_samples += 1
        if compliant:
            compliant_samples += 1

        pb_files = s.get('pb_files') or [s.get('pb_file', '')]

        entry = {
            "sample_id": s.get('sample_id', s.get('segment_id', 0)),
            "directory_key": s.get('dir_key', ''),
            "cloud_path": s.get('cloud_path', ''),
            "pb_file": s.get('pb_file', ''),
            "pb_files": pb_files,
            "timestamp_ns": s.get('timestamp_ns', 0),
            "confirmed": confirmed,
            "compliance_passed": compliant,
        }

        if 'detection' in s:
            entry["detection"] = {
                "method": s['detection'].get('method', ''),
                "metrics": s['detection'].get('metrics', {}),
            }
        if 'compliance' in s:
            entry["compliance"] = s['compliance']

        if driver_id not in driver_scene_entries:
            driver_scene_entries[driver_id] = {}
        if scene_en not in driver_scene_entries[driver_id]:
            driver_scene_entries[driver_id][scene_en] = []
        driver_scene_entries[driver_id][scene_en].append(entry)

    # 流式写入
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n')

        # metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_drivers": len(drivers),
            "total_samples": total_samples,
            "confirmed_samples": confirmed_samples,
            "compliant_samples": compliant_samples,
        }
        f.write('  "metadata": ')
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
        f.write(',\n')

        # drivers
        f.write('  "drivers": {\n')
        driver_list = sorted(driver_scene_entries.keys())
        for di, drv in enumerate(driver_list):
            scenes = driver_scene_entries[drv]
            f.write(f'    {json.dumps(drv)}: {{\n')
            scene_keys = sorted(scenes.keys())
            for si, scn in enumerate(scene_keys):
                f.write(f'      {json.dumps(scn)}: ')
                f.write(json.dumps(scenes[scn], ensure_ascii=False, indent=2))
                if si < len(scene_keys) - 1:
                    f.write(',')
                f.write('\n')
            f.write('    }')
            if di < len(driver_list) - 1:
                f.write(',')
            f.write('\n')
        f.write('  }\n')
        f.write('}\n')

    print(f"[output] 训练清单已写入: {output_path}")
    print(f"  驾驶员数: {len(drivers)}")
    print(f"  总采样数: {total_samples}")
    print(f"  已确认: {confirmed_samples}")
    print(f"  合规: {compliant_samples}")

    return output_path
