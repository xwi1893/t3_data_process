"""
extract_features_from_manifest.py
从 training_manifest.json 中读取确认合规的样本，
加载 pb 文件并提取特征，输出 scene_features.json

用法:
    python extract_features_from_manifest.py
    python extract_features_from_manifest.py --manifest ./output_data/training_manifest.json
    python extract_features_from_manifest.py --output-dir ./output_data
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import SCENE_TYPE_NAMES, PERFORMANCE_CFG
from data_loader.pb_loader import load_frames_by_files
from detection.scene_detectors import extract_segment_features
from feature_extraction.general_features import extract_general_features
from feature_extraction.scene_extractors import extract_scene_features
from output.feature_writer import NumpyEncoder


def load_manifest(manifest_path: str) -> dict:
    """加载 training_manifest.json"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_compliant_samples(manifest: dict) -> list:
    """从 manifest 中收集所有确认且合规的样本条目

    Returns:
        样本列表, 每个元素包含 driver_id, scene_type, scene_name 及原始样本数据
    """
    samples = []
    drivers = manifest.get('drivers', {})

    for driver_id, scene_dict in drivers.items():
        for scene_name, sample_list in scene_dict.items():
            # 从 scene_name 推断 scene_type
            scene_type = None
            for st, sn in SCENE_TYPE_NAMES.items():
                if sn == scene_name:
                    scene_type = st
                    break
            if scene_type is None:
                print(f"[警告] 未知场景名: {scene_name}, 跳过")
                continue

            for s in sample_list:
                if not s.get('confirmed', False):
                    continue
                if not s.get('compliance_passed', False):
                    continue

                samples.append({
                    'driver_id': driver_id,
                    'scene_type': scene_type,
                    'scene_name': scene_name,
                    'sample': s,
                })

    return samples


def extract_features_for_sample(
    sample_entry: dict,
) -> dict:
    """为单个合规样本加载 pb 并提取特征

    Args:
        sample_entry: collect_compliant_samples 返回的元素

    Returns:
        特征条目 (与 scene_features.json 格式一致), 或 None (加载/提取失败)
    """
    driver_id = sample_entry['driver_id']
    scene_type = sample_entry['scene_type']
    scene_name = sample_entry['scene_name']
    s = sample_entry['sample']

    cloud_path = s.get('cloud_path', '')
    pb_file = s.get('pb_file', '')
    pb_files_list = s.get('pb_files') or [pb_file]

    # 构造完整路径
    full_pb_path = os.path.join(cloud_path, pb_file) if cloud_path else pb_file
    full_pb_files = [
        os.path.join(cloud_path, pf) if cloud_path else pf
        for pf in pb_files_list
    ]

    # 检查 pb 目录是否存在
    if not cloud_path or not os.path.isdir(cloud_path):
        print(f"  [跳过] pb 目录不存在: {cloud_path}")
        return None

    # 加载 pb 帧数据
    frame_data = load_frames_by_files(cloud_path, pb_files_list)
    if not frame_data:
        print(f"  [跳过] 无有效帧数据: {full_pb_path}")
        return None

    # 提取特征向量 (复用 scene_detectors 的 extract_segment_features)
    features = extract_segment_features(frame_data, pb_files_list)
    if features is None:
        print(f"  [跳过] 特征提取失败: {full_pb_path}")
        return None

    # 提取通用特征和场景专用特征
    general_feats = extract_general_features(features, features)
    scene_feats = extract_scene_features(features, scene_type)

    # 构造输出条目
    entry = {
        "driver_id": driver_id,
        "scene_type": scene_type,
        "scene_name": scene_name,
        "sample_id": s.get('sample_id', 0),
        "directory_key": s.get('directory_key', ''),
        "pb_file": full_pb_path,
        "pb_files": full_pb_files,
        "timestamp_ns": s.get('timestamp_ns', 0),
        "general": general_feats,
    }

    # 场景专用特征 (以场景英文名作为 key)
    if scene_feats and "error" not in scene_feats:
        entry[scene_name] = scene_feats

    # 检测指标 (从 manifest 中保留)
    detection = s.get('detection', {})
    entry["detection_metrics"] = detection.get('metrics', {})

    # 合规指标
    compliance = s.get('compliance', {})
    entry["compliance_metrics"] = compliance.get('metrics', {})

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="从 training_manifest.json 提取确认合规样本的特征"
    )
    parser.add_argument(
        "--manifest", type=str, default="",
        help="training_manifest.json 路径 (默认: output_dir/training_manifest.json)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="输出目录 (默认: 与 manifest 同目录)"
    )
    parser.add_argument(
        "--output-file", type=str, default="scene_features.json",
        help="输出文件名 (默认: scene_features.json)"
    )
    args = parser.parse_args()

    # 确定路径
    if args.manifest:
        manifest_path = args.manifest
    else:
        output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "output_data")
        manifest_path = os.path.join(output_dir, "training_manifest.json")

    if not os.path.exists(manifest_path):
        print(f"[错误] manifest 文件不存在: {manifest_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(manifest_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_file)

    # 加载 manifest
    print("=" * 60)
    print("从 training_manifest.json 提取特征")
    print("=" * 60)

    manifest = load_manifest(manifest_path)
    metadata = manifest.get('metadata', {})
    print(f"  manifest: {manifest_path}")
    print(f"  总样本: {metadata.get('total_samples', '?')}")
    print(f"  已确认: {metadata.get('confirmed_samples', '?')}")
    print(f"  合规: {metadata.get('compliant_samples', '?')}")

    # 收集合规样本
    compliant_samples = collect_compliant_samples(manifest)
    print(f"  待提取特征: {len(compliant_samples)} 个")

    if not compliant_samples:
        print("\n[提示] 没有确认且合规的样本")
        # 写入空数组
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print(f"  已写入空文件: {output_path}")
        return

    # 逐个提取特征
    print(f"\n[提取特征] 共 {len(compliant_samples)} 个样本")
    feature_entries = []
    success_count = 0
    fail_count = 0

    for i, entry in enumerate(compliant_samples):
        s = entry['sample']
        pb_path = os.path.join(s.get('cloud_path', ''), s.get('pb_file', ''))
        print(f"  [{i+1}/{len(compliant_samples)}] "
              f"{entry['driver_id']}/{entry['scene_name']} "
              f"#{s.get('sample_id', '?')}")

        result = extract_features_for_sample(entry)
        if result is not None:
            feature_entries.append(result)
            success_count += 1
        else:
            fail_count += 1

    # 写入输出
    print(f"\n[输出] {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_entries, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n{'=' * 60}")
    print(f"特征提取完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  输出: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
