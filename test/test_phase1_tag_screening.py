"""
test_phase1_tag_screening.py
阶段1测试: 标签初筛 + 分组采样 (云平台 batch 目录结构)

支持两种模式:
  - 小批量测试: --max-batches N --max-dirs M (限制范围，快速验证)
  - 全量测试: 不加限制参数，自动使用 PERFORMANCE_CFG 中的多进程/预筛/缓存/tqdm

测试内容:
  1. 索引加载 (driver_split, data_batch, date_split)
  2. 驾驶员反向索引构建
  3. 标签分类 (7标签 -> 5场景)
  4. 帧分组 + 连续片段识别 + k 帧采样 (支持多进程并行)

用法:
  # 全量测试 (使用 PERFORMANCE_CFG 优化)
  python test/test_phase1_tag_screening.py

  # 小批量测试 (快速验证)
  python test/test_phase1_tag_screening.py --max-batches 2 --max-dirs 5
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PATHS, PERFORMANCE_CFG, SCENE_LABEL_MAP, COMPOUND_LABEL_RULES


def _get_tqdm():
    """获取 tqdm 进度条"""
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


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label: str, value, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{label}: {value}")


def test_load_indexes(data_batch_path, driver_split_path, max_batches):
    """测试1: 索引文件加载"""
    print_header("测试1: 索引文件加载")

    from data_loader.index_loader import (
        load_driver_split, build_driver_reverse_index,
        load_data_batch, load_merged_date_split,
    )

    # driver_split
    driver_split = load_driver_split(driver_split_path)
    print_result("驾驶员数量", len(driver_split))

    # 反向索引
    driver_reverse_index = build_driver_reverse_index(driver_split)
    print_result("反向索引条目", len(driver_reverse_index))

    # 抽样展示
    sample_keys = list(driver_reverse_index.keys())[:3]
    for k in sample_keys:
        print(f"    ({k[0]}, {k[1]}) -> {driver_reverse_index[k]}")

    # data_batch
    all_batch_dirs = load_data_batch(data_batch_path)
    print_result("data_batch 总 batch 数", len(all_batch_dirs))

    # 截取 (max_batches=0 表示全量)
    if max_batches > 0:
        batch_dirs = all_batch_dirs[:max_batches]
    else:
        batch_dirs = all_batch_dirs
    print_result("本次测试 batch 数", len(batch_dirs))
    if len(batch_dirs) <= 5:
        for bd in batch_dirs:
            print(f"    {bd}")

    # 合并 date_split (支持缓存 + 并行加载)
    use_cache = PERFORMANCE_CFG.get("use_date_split_cache", True)
    batch_workers = PERFORMANCE_CFG.get("batch_workers", 0)
    date_split = load_merged_date_split(batch_dirs, use_cache=use_cache,
                                         max_workers=batch_workers)
    print_result("date_split 条目", len(date_split))

    return driver_reverse_index, batch_dirs, date_split


def test_classify_labels(sample_frame_scenes):
    """测试2: 标签分类"""
    print_header("测试2: 标签分类 (7标签 -> 5场景)")

    from grouper.scene_classifier import classify_frame, classify_frame_scene_all
    from config import SCENE_LABEL_MAP, COMPOUND_LABEL_RULES

    # 展示映射表
    print("\n  标签映射规则:")
    print("    [复合规则] (优先匹配):")
    for rule in COMPOUND_LABEL_RULES:
        req = ', '.join(rule['required_labels'])
        stype, cn, en = rule['result']
        print(f"      {{{req}}} -> Scene {stype} ({cn}, {en})")
    print("    [单标签规则]:")
    for label, (stype, cn, en) in SCENE_LABEL_MAP.items():
        print(f"      {label} -> Scene {stype} ({cn}, {en})")

    # 单元级测试
    print("\n  单元测试:")
    test_cases = [
        (["at_intersection", "brake2stop"], "Scene 1"),
        (["brake2stop", "at_intersection"], "Scene 1"),  # 顺序无关
        (["at_intersection"], "None"),  # 单独 at_intersection 不匹配
        (["static2move"], "Scene 2"),
        (["longi_interaction_follow_front_large_vehicle"], "Scene 3"),
        (["longi_interaction_follow_front_small_vehicle"], "Scene 3"),
        (["brake2stop"], "Scene 4"),
        (["left_lane_change_effi"], "Scene 5"),
        (["right_lane_change_effi"], "Scene 5"),
        (["error_data"], "None"),
        ([], "None"),
        (["unknown_label"], "None"),
    ]

    all_pass = True
    for labels, expected_prefix in test_cases:
        result = classify_frame(labels)
        if result is None:
            actual = "None"
            is_ok = expected_prefix == "None"
        else:
            actual = f"Scene {result[0]}"
            is_ok = actual == expected_prefix
        mark = "PASS" if is_ok else "FAIL"
        if not is_ok:
            all_pass = False
        print(f"    [{mark}] {labels} -> {actual}")

    print(f"\n  标签分类单元测试: {'全部通过' if all_pass else '有失败项'}")

    # 全量分类 (对采样的 frame_scene 做分类统计)
    total_frames = 0
    total_classified = 0
    for dir_key, frame_scene in sample_frame_scenes.items():
        classified = classify_frame_scene_all(frame_scene)
        total_frames += len(frame_scene)
        total_classified += len(classified)

    print_result("采样目录有效场景帧", f"{total_classified} / {total_frames}")

    return total_classified > 0


def _process_dir_for_test(
    batch_dir, dir_key, driver_reverse_index, date_split,
    target_labels, use_prescreen,
):
    """处理单个 (batch, dir_key) 目录，返回 (segments, stat_str)

    目录级并行 worker: 支持多进程调用，返回值均可序列化
    """
    from data_loader.index_loader import (
        load_frame_scene, lookup_driver_id, check_label_prescreen,
    )
    from grouper.segment_merger import group_and_sample

    batch_name = os.path.basename(batch_dir)
    dir_path = os.path.join(batch_dir, "date", dir_key)

    # 验证 driver_id
    driver_id = lookup_driver_id(dir_key, driver_reverse_index)
    if driver_id is None:
        return [], "no_driver"

    # 预筛
    if use_prescreen and not check_label_prescreen(dir_path, target_labels):
        return [], "prescreen"

    # 加载 frame_scene.json
    fs_path = os.path.join(dir_path, "frame_scene.json")
    if not os.path.exists(fs_path):
        return [], "no_scene"

    frame_scene = load_frame_scene(fs_path)
    samples = group_and_sample(
        frame_scene, dir_key, driver_reverse_index, date_split
    )
    for s in samples:
        s['batch'] = batch_name
    return samples, "processed"


def _scan_date_dirs(date_dir):
    """扫描 date 目录下的子目录名称列表 (多进程 worker)"""
    dir_keys = []
    try:
        with os.scandir(date_dir) as it:
            for entry in it:
                if entry.is_dir():
                    dir_keys.append(entry.name)
    except OSError:
        pass
    return dir_keys


def test_group_and_sample(batch_dirs, driver_reverse_index, date_split, max_dirs):
    """测试3: 分组 + 连续片段识别 + k 帧采样 (目录级多进程并行 + tqdm 进度条)"""
    print_header("测试3: 分组 + k 帧采样")

    target_labels = _get_target_labels()
    use_prescreen = PERFORMANCE_CFG.get("use_label_prescreen", True)
    batch_workers = PERFORMANCE_CFG.get("batch_workers", 0)
    tqdm_bar = _get_tqdm()

    # 收集所有待处理的 (batch_dir, dir_key) 工作项
    # 并行扫描: 将每个 batch 的目录扫描分发给 worker 进程
    work_items = []
    scan_jobs = []
    for batch_dir in batch_dirs:
        date_dir = os.path.join(batch_dir, "date")
        if not os.path.isdir(date_dir):
            continue
        scan_jobs.append((batch_dir, date_dir))

    if batch_workers > 1 and len(scan_jobs) > 1:
        # === 并行扫描 (目录级) ===
        with ProcessPoolExecutor(max_workers=batch_workers) as pool:
            futures = {
                pool.submit(_scan_date_dirs, dd): bd
                for bd, dd in scan_jobs
            }
            for fut in tqdm_bar(as_completed(futures), total=len(futures),
                                desc="  目录扫描"):
                bd = futures[fut]
                try:
                    dir_keys = fut.result()
                except Exception:
                    continue
                if max_dirs > 0:
                    dir_keys = dir_keys[:max_dirs]
                for dk in dir_keys:
                    work_items.append((bd, dk))
    else:
        # === 串行扫描 ===
        for bd, dd in scan_jobs:
            dir_keys = _scan_date_dirs(dd)
            if max_dirs > 0:
                dir_keys = dir_keys[:max_dirs]
            for dk in dir_keys:
                work_items.append((bd, dk))

    print(f"  工作项: {len(work_items)} 个目录")

    all_samples = []
    merged_stats = {
        "total_dirs": len(work_items),
        "skipped_no_driver": 0, "skipped_prescreen": 0,
        "skipped_no_scene": 0, "processed": 0,
    }
    stat_map = {
        "no_driver": "skipped_no_driver",
        "prescreen": "skipped_prescreen",
        "no_scene": "skipped_no_scene",
        "processed": "processed",
    }

    if batch_workers > 1 and len(work_items) > 1:
        # === 多进程并行 (目录级) ===
        print(f"  多进程模式: {batch_workers} 个 worker")
        with ProcessPoolExecutor(max_workers=batch_workers) as pool:
            futures = {
                pool.submit(
                    _process_dir_for_test,
                    bd, dk, driver_reverse_index, date_split,
                    target_labels, use_prescreen,
                ): (bd, dk)
                for bd, dk in work_items
            }
            for fut in tqdm_bar(as_completed(futures), total=len(futures),
                                desc="  目录处理"):
                try:
                    samples, stat = fut.result()
                    all_samples.extend(samples)
                    key = stat_map.get(stat)
                    if key:
                        merged_stats[key] += 1
                except Exception as e:
                    bd, dk = futures[fut]
                    print(f"\n  [错误] {os.path.basename(bd)}/{dk}: {e}")
    else:
        # === 单进程串行 ===
        for bd, dk in tqdm_bar(work_items, desc="  目录处理"):
            samples, stat = _process_dir_for_test(
                bd, dk, driver_reverse_index, date_split,
                target_labels, use_prescreen,
            )
            all_samples.extend(samples)
            key = stat_map.get(stat)
            if key:
                merged_stats[key] += 1

    # 汇总
    scene_names = {1: "路口停车", 2: "起步", 3: "跟车", 4: "跟停", 5: "变道"}

    print(f"\n  {'='*50}")
    print(f"  扫描目录: {merged_stats['total_dirs']}")
    print(f"  跳过(无 driver_id): {merged_stats['skipped_no_driver']}")
    if use_prescreen:
        print(f"  跳过(预筛无标签): {merged_stats['skipped_prescreen']}")
    print(f"  跳过(无 frame_scene): {merged_stats['skipped_no_scene']}")
    print(f"  实际处理: {merged_stats['processed']}")
    print(f"  候选采样: {len(all_samples)}")

    if all_samples:
        type_counts = defaultdict(int)
        for s in all_samples:
            type_counts[s['scene_type']] += 1
        print(f"\n  按场景类型统计:")
        for st, count in sorted(type_counts.items()):
            print(f"    Scene {st} ({scene_names.get(st, '?')}): {count} 个")

        # 采样明细 (仅展示前 20 个)
        show_limit = min(20, len(all_samples))
        print(f"\n  采样明细 (前 {show_limit}/{len(all_samples)}):")
        for s in all_samples[:show_limit]:
            name = scene_names.get(s['scene_type'], f"scene{s['scene_type']}")
            print(f"\n    [{s.get('batch', '?')}] {s['dir_key']}")
            print(f"      Scene {s['scene_type']} ({name}), "
                  f"driver={s['driver_id']}")
            print(f"      pb_file: {s['pb_file']}")
            print(f"      cloud_path: {s['cloud_path'][:60]}...")

    return all_samples, merged_stats


def main():
    parser = argparse.ArgumentParser(
        description="阶段1测试: 标签初筛 + 分组采样 (云平台 batch 结构)"
    )
    parser.add_argument("--max-batches", type=int, default=0,
                        help="最多处理的 batch 数 (0=全量, 默认 0)")
    parser.add_argument("--max-dirs", type=int, default=0,
                        help="每个 batch 最多处理的 date 目录数 (0=全量, 默认 0)")
    args = parser.parse_args()

    # 确定路径
    data_batch_path = DATA_PATHS["data_batch_file"]
    driver_split_path = DATA_PATHS["driver_split_file"]

    start_time = datetime.now()
    mode = "全量" if args.max_batches == 0 and args.max_dirs == 0 else "小批量"
    print(f"阶段1测试 ({mode}模式)")
    print(f"  data_batch:   {data_batch_path}")
    print(f"  driver_split: {driver_split_path}")
    print(f"  max_batches:  {args.max_batches or '全量'}")
    print(f"  max_dirs:     {args.max_dirs or '全量'}")
    perf = PERFORMANCE_CFG
    print(f"  性能配置: batch_workers={perf['batch_workers']}, "
          f"prescreen={perf['use_label_prescreen']}, "
          f"cache={perf['use_date_split_cache']}, "
          f"tqdm={perf['progress_bar']}")

    # 验证文件存在
    for path, name in [
        (data_batch_path, "data_batch.json"),
        (driver_split_path, "driver_split.json"),
    ]:
        if not os.path.exists(path):
            print(f"\n错误: {name} 不存在: {path}")
            print("请修改 config.py 中的 DATA_PATHS")
            sys.exit(1)

    # 测试1: 索引加载
    driver_reverse_index, batch_dirs, date_split = \
        test_load_indexes(data_batch_path, driver_split_path, args.max_batches)

    if not batch_dirs:
        print("\n[结束] 无 batch 目录")
        sys.exit(0)

    # 预采样: 加载少量 frame_scene 用于标签分类测试
    print_header("预采样 frame_scene (标签分类用)")
    sample_frame_scenes = {}
    from data_loader.index_loader import load_frame_scene

    for batch_dir in batch_dirs[:1]:  # 只用第一个 batch 采样
        date_dir = os.path.join(batch_dir, "date")
        if not os.path.isdir(date_dir):
            continue
        for dir_key in os.listdir(date_dir)[:3]:
            fs_path = os.path.join(date_dir, dir_key, "frame_scene.json")
            if os.path.exists(fs_path):
                sample_frame_scenes[dir_key] = load_frame_scene(fs_path)
    print(f"  采样 {len(sample_frame_scenes)} 个目录")

    # 测试2: 标签分类
    has_valid = test_classify_labels(sample_frame_scenes)
    if not has_valid:
        print("\n[警告] 采样的 frame_scene 中无有效场景标签")

    # 测试3: 分组 + k 帧采样
    all_samples, stats = test_group_and_sample(
        batch_dirs, driver_reverse_index, date_split, args.max_dirs
    )

    # 输出采样结果供阶段 2 使用
    output_dir = os.path.join(PROJECT_ROOT, "test", "test_output")
    os.makedirs(output_dir, exist_ok=True)
    phase1_output = os.path.join(output_dir, "phase1_samples.json")

    from output.streaming_writer import StreamingJsonArrayWriter
    with StreamingJsonArrayWriter(phase1_output) as writer:
        for s in all_samples:
            entry = {k: v for k, v in s.items()
                     if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            writer.append(entry)

    # 总结
    elapsed = (datetime.now() - start_time).total_seconds()
    print_header("阶段1测试完成")
    print(f"  模式:       {mode}")
    print(f"  耗时:       {elapsed:.1f}s")
    print(f"  处理 batch: {len(batch_dirs)} 个")
    print(f"  处理目录:   {stats['processed']} 个")
    print(f"  候选采样:   {len(all_samples)} 个")
    print(f"  采样已保存: {phase1_output} ({len(all_samples)} 个)")
    print("  阶段2 测试请运行:")
    print("    python test/test_phase2_detection.py")


if __name__ == "__main__":
    main()
