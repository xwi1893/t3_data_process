"""
main.py
场景清洗与提取 - CLI 入口与流水线编排

用法:
    python main.py --config config_override.json
    python main.py --frame-scene-dir /path/to/frame_scenes \
                   --driver-split /path/to/driver_split.json \
                   --date-split /path/to/date_split.json \
                   --pb-dir /path/to/pb_data \
                   --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PATHS, MERGE_PARAMS
from data_loader.index_loader import (
    load_driver_split, build_driver_reverse_index,
    load_date_split, load_frame_scene,
)
from data_loader.pb_loader import load_frames_by_files
from grouper.segment_merger import group_and_merge
from detection.detector import confirm_scene
from compliance.checker import check_compliance
from feature_extraction.base_extractor import extract_all_features
from feature_extraction.general_features import extract_general_features
from feature_extraction.scene_extractors import extract_scene_features
from output.training_manifest import write_manifest
from output.feature_writer import write_features


def load_config_override(config_path: str) -> dict:
    """加载配置覆盖文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pipeline(
    frame_scene_dir: str = "",
    driver_split_file: str = "",
    date_split_file: str = "",
    pb_data_dir: str = "",
    output_dir: str = "",
) -> dict:
    """执行完整流水线

    Stage 1: 索引加载
    Stage 2: 标签初步筛选
    Stage 3: 分组与合并
    Stage 4: 帧数据加载
    Stage 5: 算法二次确认
    Stage 6: 合规检查
    Stage 7: 特征提取 + 输出
    """
    print("=" * 60)
    print("场景清洗与提取流水线")
    print("=" * 60)
    start_time = datetime.now()

    # ===== Stage 1: 索引加载 =====
    print("\n[Stage 1] 索引加载")
    driver_split = load_driver_split(driver_split_file)
    driver_reverse_index = build_driver_reverse_index(driver_split)
    date_split = load_date_split(date_split_file)

    # ===== Stage 2-3: 标签筛选 + 分组合并 =====
    print("\n[Stage 2-3] 标签筛选 + 分组合并")

    # 发现 frame_scene 文件
    # 根据实际目录结构调整此逻辑
    all_segments = []
    frame_scene_files = []

    if frame_scene_dir and os.path.isdir(frame_scene_dir):
        for fname in os.listdir(frame_scene_dir):
            if fname.endswith('.json'):
                frame_scene_files.append(os.path.join(frame_scene_dir, fname))

    if not frame_scene_files:
        print("[警告] 未找到 frame_scene 文件，请检查路径配置")
        return {"error": "无 frame_scene 文件"}

    print(f"  发现 {len(frame_scene_files)} 个 frame_scene 文件")

    for fs_path in frame_scene_files:
        print(f"\n  处理: {os.path.basename(fs_path)}")
        frame_scene = load_frame_scene(fs_path)

        # 用文件名作为 dir_key (实际需根据目录结构调整)
        dir_key = os.path.splitext(os.path.basename(fs_path))[0]

        segments = group_and_merge(
            frame_scene, dir_key, driver_reverse_index, date_split
        )
        all_segments.extend(segments)

    print(f"\n  总计 {len(all_segments)} 个候选片段")

    if not all_segments:
        print("[结束] 无候选片段")
        return {"total_segments": 0}

    # ===== Stage 4: 帧数据加载 + Stage 5: 检测确认 + Stage 6: 合规 =====
    print("\n[Stage 4-6] 帧数据加载 + 检测确认 + 合规检查")

    for seg in all_segments:
        dir_key = seg['dir_key']

        # 加载 pb 帧数据
        # 实际使用时需要构造正确的 pb 目录路径
        pb_dir = os.path.join(pb_data_dir, dir_key) if pb_data_dir else ""

        if pb_dir and os.path.isdir(pb_dir):
            frame_data = load_frames_by_files(pb_dir, seg['pb_files'])
        else:
            print(f"  [跳过] pb 目录不存在: {pb_dir}")
            seg['confirmed'] = False
            seg['confirm_reason'] = f"pb 目录不存在: {pb_dir}"
            continue

        if not frame_data:
            seg['confirmed'] = False
            seg['confirm_reason'] = "无有效帧数据"
            continue

        # Stage 5: 算法二次确认
        confirm_scene(seg, frame_data)

        if not seg.get('confirmed', False):
            continue

        # Stage 6: 合规检查
        check_compliance(seg, frame_data)

        # ===== Stage 7: 特征提取 =====
        # 直接特征 + 间接特征
        features = extract_all_features(frame_data, seg['pb_files'])
        if features is not None:
            # 通用特征
            seg['general_features'] = extract_general_features(features, features)
            # 场景专用特征
            seg['scene_features'] = extract_scene_features(
                features, seg['scene_type']
            )
            seg['features'] = features

    # ===== 输出 =====
    print("\n[Stage 7] 输出")
    if not output_dir:
        output_dir = os.path.join(PROJECT_ROOT, "output_data")

    os.makedirs(output_dir, exist_ok=True)

    # 训练清单
    manifest_path = os.path.join(output_dir, "training_manifest.json")
    write_manifest(all_segments, manifest_path)

    # 特征文件
    feature_path = write_features(all_segments, output_dir)

    # 汇总
    elapsed = (datetime.now() - start_time).total_seconds()
    confirmed = sum(1 for s in all_segments if s.get('confirmed', False))
    compliant = sum(1 for s in all_segments
                    if s.get('compliance', {}).get('passed', False))

    summary = {
        "total_candidates": len(all_segments),
        "confirmed": confirmed,
        "compliant": compliant,
        "elapsed_seconds": round(elapsed, 1),
        "manifest_path": manifest_path,
        "feature_path": feature_path,
    }

    print(f"\n{'=' * 60}")
    print(f"流水线完成 ({elapsed:.1f}s)")
    print(f"  候选: {len(all_segments)}")
    print(f"  已确认: {confirmed}")
    print(f"  合规: {compliant}")
    print(f"{'=' * 60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="场景清洗与提取")
    parser.add_argument("--config", type=str, default="",
                        help="配置覆盖文件路径 (JSON)")
    parser.add_argument("--frame-scene-dir", type=str, default="",
                        help="frame_scene.json 所在目录")
    parser.add_argument("--driver-split", type=str, default="",
                        help="driver_split.json 路径")
    parser.add_argument("--date-split", type=str, default="",
                        help="date_split.json 路径")
    parser.add_argument("--pb-dir", type=str, default="",
                        help="pb 数据文件基础目录")
    parser.add_argument("--output-dir", type=str, default="",
                        help="输出目录")

    args = parser.parse_args()

    # 合并配置
    paths = {
        "frame_scene_base_dir": args.frame_scene_dir or DATA_PATHS["frame_scene_base_dir"],
        "driver_split_file": args.driver_split or DATA_PATHS["driver_split_file"],
        "date_split_file": args.date_split or DATA_PATHS["date_split_file"],
        "pb_data_base_dir": args.pb_dir or DATA_PATHS["pb_data_base_dir"],
        "output_dir": args.output_dir or DATA_PATHS["output_dir"],
    }

    # 加载配置覆盖
    if args.config:
        override = load_config_override(args.config)
        paths.update(override.get("data_paths", {}))

    # 验证路径
    required = ["driver_split_file", "date_split_file", "frame_scene_base_dir"]
    missing = [k for k in required if not paths.get(k)]
    if missing:
        print(f"错误: 缺少必要路径配置: {', '.join(missing)}")
        print("请通过命令行参数或 config.py 中的 DATA_PATHS 配置")
        sys.exit(1)

    summary = run_pipeline(
        frame_scene_dir=paths["frame_scene_base_dir"],
        driver_split_file=paths["driver_split_file"],
        date_split_file=paths["date_split_file"],
        pb_data_dir=paths["pb_data_base_dir"],
        output_dir=paths["output_dir"],
    )

    return summary


if __name__ == "__main__":
    main()
