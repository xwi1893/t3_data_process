"""
main.py
场景清洗与提取 - CLI 入口与流水线编排

用法:
    python main.py                   # 运行全部阶段
    python main.py --phase 1         # 仅标签筛选 + k帧采样
    python main.py --phase 2         # 仅检测确认 + 特征提取
    python main.py --phase 1 2       # 依次运行两阶段
    python main.py --config override.json --output-dir /path
"""

import argparse
import json
import os
import sys
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PATHS, PERFORMANCE_CFG
from data_loader.index_loader import (
    load_driver_split, build_driver_reverse_index,
    load_data_batch, load_merged_date_split,
    load_frame_scene, lookup_driver_id,
)
from data_loader.pb_loader import load_frames_by_files
from grouper.segment_merger import group_and_sample
from detection.detector import confirm_scene
from compliance.checker import check_compliance
from feature_extraction.general_features import extract_general_features
from feature_extraction.scene_extractors import extract_scene_features
from output.streaming_writer import StreamingJsonArrayWriter
from output.training_manifest import write_manifest
from output.feature_writer import write_features


# ============================================================
# 工具函数
# ============================================================

def load_config_override(config_path: str) -> dict:
    """加载配置覆盖文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# Phase 1: 标签筛选 + 连续场景分段 + k 帧采样
# ============================================================

def run_phase1(
    data_batch_file: str = "",
    driver_split_file: str = "",
    output_dir: str = "",
) -> dict:
    """Phase 1: 标签筛选 + k 帧采样

    Stage 1: 驾驶员索引加载
    Stage 2: Batch 发现
    Stage 3: date_split 加载与合并
    Stage 4: 标签筛选 + 连续场景分段 + k 帧采样

    输出: {output_dir}/phase1_samples.json
    """
    print("=" * 60)
    print("Phase 1: 标签筛选 + k 帧采样")
    print("=" * 60)
    start_time = datetime.now()

    # ===== Stage 1: 驾驶员索引加载 =====
    print("\n[Stage 1] 驾驶员索引加载")
    driver_split = load_driver_split(driver_split_file)
    driver_reverse_index = build_driver_reverse_index(driver_split)

    # ===== Stage 2: Batch 发现 =====
    print("\n[Stage 2] Batch 发现")
    batch_dirs = load_data_batch(data_batch_file)
    if not batch_dirs:
        print("[警告] 未找到 batch 目录")
        return {"error": "无 batch 目录"}
    print(f"  发现 {len(batch_dirs)} 个 batch 目录")

    # ===== Stage 3: date_split 加载与合并 =====
    print("\n[Stage 3] date_split 加载与合并")
    use_cache = PERFORMANCE_CFG.get("use_date_split_cache", True)
    batch_workers = PERFORMANCE_CFG.get("batch_workers", 0)
    date_split = load_merged_date_split(batch_dirs, use_cache=use_cache,
                                         max_workers=batch_workers)
    if not date_split:
        print("[警告] date_split 为空")
        return {"error": "无 date_split 数据"}

    # ===== Stage 4: 标签筛选 + 连续场景分段 + k 帧采样 =====
    print("\n[Stage 4] 标签筛选 + 连续场景分段 + k 帧采样")
    total_dirs = 0
    skipped_no_driver = 0
    skipped_no_scene = 0
    total_samples = 0

    # 保存中间结果
    if not output_dir:
        output_dir = os.path.join(PROJECT_ROOT, "output_data")
    os.makedirs(output_dir, exist_ok=True)
    phase1_path = os.path.join(output_dir, "phase1_samples.json")

    with StreamingJsonArrayWriter(phase1_path) as writer:
        for batch_dir in batch_dirs:
            date_dir = os.path.join(batch_dir, "date")
            if not os.path.isdir(date_dir):
                print(f"  [跳过] date 目录不存在: {date_dir}")
                continue

            for dir_key in os.listdir(date_dir):
                dir_path = os.path.join(date_dir, dir_key)
                if not os.path.isdir(dir_path):
                    continue
                total_dirs += 1

                # 验证 driver_id
                driver_id = lookup_driver_id(dir_key, driver_reverse_index)
                if driver_id is None:
                    skipped_no_driver += 1
                    continue

                # 加载 frame_scene.json
                fs_path = os.path.join(dir_path, "frame_scene.json")
                if not os.path.exists(fs_path):
                    skipped_no_scene += 1
                    continue

                frame_scene = load_frame_scene(fs_path)
                samples = group_and_sample(
                    frame_scene, dir_key, driver_reverse_index, date_split
                )

                # 流式写入每个采样条目
                for s in samples:
                    entry = {k: v for k, v in s.items()
                             if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    writer.append(entry)
                    total_samples += 1

    print(f"\n  扫描目录: {total_dirs}")
    print(f"  跳过(无 driver_id): {skipped_no_driver}")
    print(f"  跳过(无 frame_scene): {skipped_no_scene}")
    print(f"  总计 {total_samples} 个采样条目")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Phase 1 完成 ({elapsed:.1f}s)")
    print(f"  采样: {total_samples}")
    print(f"  已保存: {phase1_path}")
    print(f"{'=' * 60}")

    return {
        "total_samples": total_samples,
        "phase1_path": phase1_path,
        "elapsed_seconds": round(elapsed, 1),
    }


# ============================================================
# Phase 2: 检测确认 + 合规检查 + 特征提取 + 输出
# ============================================================

def run_phase2(output_dir: str = "") -> dict:
    """Phase 2: 检测确认 + 特征提取

    加载 Phase 1 中间结果，执行:
    Stage 5: 单帧数据加载 + 检测确认 + 合规检查 + 特征提取
    Stage 6: 输出 (training_manifest + features)
    """
    print("=" * 60)
    print("Phase 2: 检测确认 + 特征提取")
    print("=" * 60)
    start_time = datetime.now()

    if not output_dir:
        output_dir = os.path.join(PROJECT_ROOT, "output_data")

    # 加载 Phase 1 中间结果
    phase1_path = os.path.join(output_dir, "phase1_samples.json")
    if not os.path.exists(phase1_path):
        print(f"[错误] Phase 1 输出不存在: {phase1_path}")
        print("请先运行 Phase 1")
        return {"error": "无 Phase 1 输出"}

    with open(phase1_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f"  加载 Phase 1 采样: {len(all_samples)} 个")

    if not all_samples:
        return {"total_samples": 0}

    # ===== Stage 5: 单帧数据加载 + 检测确认 + 合规检查 + 特征提取 =====
    print("\n[Stage 5] 单帧数据加载 + 检测确认 + 合规检查 + 特征提取")

    for sample in all_samples:
        pb_dir = sample.get('cloud_path', '')

        if not pb_dir or not os.path.isdir(pb_dir):
            print(f"  [跳过] pb 目录不存在: {pb_dir}")
            sample['confirmed'] = False
            sample['confirm_reason'] = f"pb 目录不存在: {pb_dir}"
            continue

        pb_file = sample['pb_file']
        full_pb_path = os.path.join(pb_dir, pb_file)
        frame_data = load_frames_by_files(pb_dir, [pb_file])

        if not frame_data:
            print(f"  [拒绝] 无有效帧数据: {full_pb_path}")
            sample['confirmed'] = False
            sample['confirm_reason'] = "无有效帧数据"
            continue

        # 算法二次确认
        confirm_scene(sample, frame_data)

        if not sample.get('confirmed', False):
            continue

        # 合规检查
        check_compliance(sample, frame_data)

        # 复用 confirm_scene 已提取的特征
        features = sample.get('features')
        if features is not None:
            sample['general_features'] = extract_general_features(features, features)
            sample['scene_features'] = extract_scene_features(
                features, sample['scene_type']
            )

    # ===== Stage 6: 输出 =====
    print("\n[Stage 6] 输出")
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "training_manifest.json")
    write_manifest(all_samples, manifest_path)

    feature_path = write_features(all_samples, output_dir)

    # 汇总
    elapsed = (datetime.now() - start_time).total_seconds()
    confirmed = sum(1 for s in all_samples if s.get('confirmed', False))
    compliant = sum(1 for s in all_samples
                    if s.get('compliance', {}).get('passed', False))

    summary = {
        "total_samples": len(all_samples),
        "confirmed": confirmed,
        "compliant": compliant,
        "elapsed_seconds": round(elapsed, 1),
        "manifest_path": manifest_path,
        "feature_path": feature_path,
    }

    print(f"\n{'=' * 60}")
    print(f"Phase 2 完成 ({elapsed:.1f}s)")
    print(f"  采样: {len(all_samples)}")
    print(f"  已确认: {confirmed}")
    print(f"  合规: {compliant}")
    print(f"{'=' * 60}")

    return summary


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="场景清洗与提取")
    parser.add_argument("--phase", type=int, nargs='+', default=[1, 2],
                        choices=[1, 2],
                        help="运行阶段: 1(筛选采样) 2(检测提取), 默认全部")
    parser.add_argument("--config", type=str, default="",
                        help="配置覆盖文件路径 (JSON)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="输出目录")

    args = parser.parse_args()

    # 合并配置
    paths = {
        "driver_split_file": DATA_PATHS["driver_split_file"],
        "data_batch_file": DATA_PATHS["data_batch_file"],
        "output_dir": args.output_dir or DATA_PATHS["output_dir"],
    }

    # 加载配置覆盖
    if args.config:
        override = load_config_override(args.config)
        paths.update(override.get("data_paths", {}))

    # 验证路径 (Phase 2 单独运行时不需要索引文件)
    if 1 in args.phase:
        required = ["driver_split_file", "data_batch_file"]
        missing = [k for k in required if not paths.get(k)]
        if missing:
            print(f"错误: 缺少必要路径配置: {', '.join(missing)}")
            print("请通过命令行参数或 config.py 中的 DATA_PATHS 配置")
            sys.exit(1)

    output_dir = paths["output_dir"]

    if 1 in args.phase:
        run_phase1(
            data_batch_file=paths["data_batch_file"],
            driver_split_file=paths["driver_split_file"],
            output_dir=output_dir,
        )

    if 2 in args.phase:
        run_phase2(output_dir=output_dir)


if __name__ == "__main__":
    main()
