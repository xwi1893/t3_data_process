"""
main.py
场景清洗与提取 - CLI 入口与流水线编排

用法:
    python main.py
    python main.py --config config_override.json
    python main.py --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PATHS, PERFORMANCE_CFG, SCENE_LABEL_MAP, COMPOUND_LABEL_RULES
from data_loader.index_loader import (
    load_driver_split, build_driver_reverse_index,
    load_data_batch, load_merged_date_split,
    load_frame_scene, lookup_driver_id, check_label_prescreen,
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


def _get_tqdm():
    """获取 tqdm 进度条，若未安装则回退到无操作"""
    if not PERFORMANCE_CFG.get("progress_bar", True):
        return lambda x, **kw: x
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return lambda x, **kw: x


def _get_target_labels() -> set:
    """获取所有目标标签集合 (用于预筛)"""
    labels = set(SCENE_LABEL_MAP.keys())
    for rule in COMPOUND_LABEL_RULES:
        labels.update(rule["required_labels"])
    return labels


def load_config_override(config_path: str) -> dict:
    """加载配置覆盖文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _process_single_batch(
    batch_dir: str,
    driver_reverse_index: dict,
    date_split: dict,
    target_labels: set,
    use_prescreen: bool,
) -> list:
    """处理单个 batch 的 dir_key 目录，返回候选片段列表

    可作为 ProcessPoolExecutor 的 worker 函数

    Args:
        batch_dir: batch 目录路径
        driver_reverse_index: (date, vehicle_id) -> driver_id
        date_split: {dir_key: cloud_path}
        target_labels: 目标标签集合 (用于预筛)
        use_prescreen: 是否启用 label_count.json 预筛

    Returns:
        候选片段列表
    """
    segments = []
    date_dir = os.path.join(batch_dir, "date")
    if not os.path.isdir(date_dir):
        return segments

    for dir_key in os.listdir(date_dir):
        dir_path = os.path.join(date_dir, dir_key)
        if not os.path.isdir(dir_path):
            continue

        # 验证 driver_id
        driver_id = lookup_driver_id(dir_key, driver_reverse_index)
        if driver_id is None:
            continue

        # 预筛: 通过 label_count.json 快速判断是否包含目标标签
        if use_prescreen and not check_label_prescreen(dir_path, target_labels):
            continue

        # 加载 frame_scene.json
        fs_path = os.path.join(dir_path, "frame_scene.json")
        if not os.path.exists(fs_path):
            continue

        frame_scene = load_frame_scene(fs_path)
        batch_segments = group_and_merge(
            frame_scene, dir_key, driver_reverse_index, date_split
        )
        segments.extend(batch_segments)

    return segments


def run_pipeline(
    data_batch_file: str = "",
    driver_split_file: str = "",
    output_dir: str = "",
) -> dict:
    """执行完整流水线

    Stage 1: 驾驶员索引加载
    Stage 2: Batch 发现
    Stage 3: date_split 加载与合并
    Stage 4: 标签筛选 + 分组合并
    Stage 5: 帧数据加载 + 检测确认 + 合规检查
    Stage 6: 特征提取 + 输出
    """
    print("=" * 60)
    print("场景清洗与提取流水线")
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

    # ===== Stage 4: 标签筛选 + 分组合并 =====
    print("\n[Stage 4] 标签筛选 + 分组合并")
    all_segments = []
    target_labels = _get_target_labels()
    use_prescreen = PERFORMANCE_CFG.get("use_label_prescreen", True)
    batch_workers = PERFORMANCE_CFG.get("batch_workers", 0)
    tqdm_bar = _get_tqdm()

    if batch_workers > 1 and len(batch_dirs) > 1:
        # === 多进程并行 ===
        print(f"  多进程模式: {batch_workers} 个 worker")
        with ProcessPoolExecutor(max_workers=batch_workers) as pool:
            futures = {}
            for batch_dir in batch_dirs:
                fut = pool.submit(
                    _process_single_batch,
                    batch_dir, driver_reverse_index, date_split,
                    target_labels, use_prescreen,
                )
                futures[fut] = batch_dir

            for fut in tqdm_bar(as_completed(futures), total=len(futures),
                                desc="  batch 处理"):
                try:
                    segments = fut.result()
                    all_segments.extend(segments)
                except Exception as e:
                    batch_dir = futures[fut]
                    print(f"\n  [错误] batch {batch_dir}: {e}")
    else:
        # === 单进程串行 ===
        for batch_dir in tqdm_bar(batch_dirs, desc="  batch 处理"):
            segments = _process_single_batch(
                batch_dir, driver_reverse_index, date_split,
                target_labels, use_prescreen,
            )
            all_segments.extend(segments)

    print(f"\n  总计 {len(all_segments)} 个候选片段")

    if not all_segments:
        print("[结束] 无候选片段")
        return {"total_segments": 0}

    # ===== Stage 5: 帧数据加载 + 检测确认 + 合规检查 =====
    print("\n[Stage 5] 帧数据加载 + 检测确认 + 合规检查")
    pb_load_workers = PERFORMANCE_CFG.get("pb_load_workers", 0)

    for seg in tqdm_bar(all_segments, desc="  片段处理"):
        # 从 segment 的 cloud_path 推导 pb 目录
        pb_dir = seg.get('cloud_path', '')

        if not pb_dir or not os.path.isdir(pb_dir):
            print(f"  [跳过] pb 目录不存在: {pb_dir}")
            seg['confirmed'] = False
            seg['confirm_reason'] = f"pb 目录不存在: {pb_dir}"
            continue

        frame_data = load_frames_by_files(pb_dir, seg['pb_files'],
                                           max_workers=pb_load_workers)

        if not frame_data:
            seg['confirmed'] = False
            seg['confirm_reason'] = "无有效帧数据"
            continue

        # 算法二次确认
        confirm_scene(seg, frame_data)

        if not seg.get('confirmed', False):
            continue

        # 合规检查
        check_compliance(seg, frame_data)

        # ===== 特征提取 =====
        features = extract_all_features(frame_data, seg['pb_files'])
        if features is not None:
            seg['general_features'] = extract_general_features(features, features)
            seg['scene_features'] = extract_scene_features(
                features, seg['scene_type']
            )
            seg['features'] = features

    # ===== 输出 =====
    print("\n[Stage 6] 输出")
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

    # 验证路径
    required = ["driver_split_file", "data_batch_file"]
    missing = [k for k in required if not paths.get(k)]
    if missing:
        print(f"错误: 缺少必要路径配置: {', '.join(missing)}")
        print("请通过命令行参数或 config.py 中的 DATA_PATHS 配置")
        sys.exit(1)

    summary = run_pipeline(
        data_batch_file=paths["data_batch_file"],
        driver_split_file=paths["driver_split_file"],
        output_dir=paths["output_dir"],
    )

    return summary


if __name__ == "__main__":
    main()
