"""
test_phase2_detection.py
阶段2测试: 算法确认 + 合规检查 + 特征提取

测试内容 (均使用真实数据):
  1. pb 帧加载与归一化
  2. 帧数据工具函数 (前车检测、横向位置、停止线)

端到端测试 (需要阶段1输出):
  3. 加载阶段1采样 → 检测确认 → 合规检查 → 特征提取 → 输出

用法:
  # 先运行阶段1, 再运行阶段2
  python test/test_phase1_tag_screening.py
  python test/test_phase2_detection.py

  # 指定阶段1输出路径
  python test/test_phase2_detection.py --phase1-output /path/to/phase1_samples.json

  # 限制端到端测试的采样数量
  python test/test_phase2_detection.py --max-samples 5
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_REF_DIR = os.path.join(PROJECT_ROOT, "reference")
DEFAULT_PHASE1_OUTPUT = os.path.join(PROJECT_ROOT, "test", "test_output", "phase1_samples.json")


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label: str, value, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{label}: {value}")


# ============================================================
# 测试1: pb 帧加载
# ============================================================

def test_pb_loading(pb_path: str):
    """测试 pb 文件加载与归一化"""
    print_header("测试1: pb 帧加载与归一化")

    from data_loader.pb_loader import load_pb_frame, normalize_frame

    print(f"\n  加载: {os.path.basename(pb_path)}")

    raw = load_pb_frame(pb_path)
    print_result("原始字段数", len(raw))
    top_keys = list(raw.keys())[:8]
    print(f"  顶层字段: {top_keys}")

    normed = normalize_frame(raw)
    print_result("归一化字段", list(normed.keys())[:8])

    ego = normed.get('data_ego_curr_status', {})
    print(f"\n  自车状态:")
    print(f"    速度 v = {ego.get('v', 0):.2f} m/s")
    print(f"    航向 yaw = {ego.get('yaw', 0):.4f}")
    print(f"    角速度 yaw_rate = {ego.get('yaw_rate', 0):.4f}")

    agents = normed.get('data_agent', [])
    print(f"\n  周围目标: {len(agents)} 个")
    if agents:
        cls_counts = {}
        for a in agents:
            c = a.get('cls', -1)
            cls_counts[c] = cls_counts.get(c, 0) + 1
        print(f"    类别分布: {cls_counts}")
        for a in agents:
            if a.get('cls') == 2:
                print(f"    车辆示例: x={a.get('x', 0):.1f}, y={a.get('y', 0):.1f}, "
                      f"vx={a.get('vx', 0):.2f}, vy={a.get('vy', 0):.2f}")
                break

    lanes = normed.get('data_laneline', [])
    print(f"\n  车道线: {len(lanes)} 条")
    for i, lane in enumerate(lanes[:4]):
        pts = lane.get('pts_fixed_num', [])
        lt = lane.get('laneline_type', lane.get('type', 0))
        print(f"    [{i}] type={lt}, 采样点数={len(pts)}")

    roaditems = normed.get('data_roaditems', {})
    stoplines = roaditems.get('stoplines', [])
    print(f"\n  道路元素: 停止线 {len(stoplines)} 条")

    return raw, normed


# ============================================================
# 测试2: 帧工具函数
# ============================================================

def test_frame_utils(normed_frame: dict):
    """测试帧数据工具函数"""
    print_header("测试2: 帧数据工具函数")

    from utils.frame_utils import (
        get_lead_vehicle, calculate_lateral_position,
        check_has_stopline, check_has_traffic_light,
        get_lane_boundaries, smooth_lead_distance,
    )

    agents = normed_frame.get('data_agent', [])
    lanes = normed_frame.get('data_laneline', [])
    ego = normed_frame.get('data_ego_curr_status', {})

    left, right, width = get_lane_boundaries(lanes)
    print(f"\n  车道边界:")
    print(f"    左边界: {left}")
    print(f"    右边界: {right}")
    print(f"    车道宽度: {width:.2f}m")

    lead, dist, speed = get_lead_vehicle(agents, lanes, ego.get('yaw', 0))
    print(f"\n  前车检测:")
    if lead:
        print(f"    存在前车: 距离={dist:.1f}m, 速度={speed:.2f}m/s")
        print(f"    前车位置: x={lead.get('x', 0):.1f}, y={lead.get('y', 0):.1f}")
    else:
        print(f"    无前车")

    lat_pos, curvature = calculate_lateral_position(lanes)
    print(f"\n  横向位置:")
    print(f"    偏移: {lat_pos:.3f}m")
    print(f"    曲率: {curvature:.4f}")

    has_sl = check_has_stopline(normed_frame)
    has_tl = check_has_traffic_light(normed_frame)
    print(f"\n  路口检测:")
    print(f"    停止线: {'有' if has_sl else '无'}")
    print(f"    信号灯: {'有' if has_tl else '无'}")

    test_dists = np.array([10.0, 12.0, 0.0, 11.0, 13.0, 0.0, 14.0, 12.0, 11.0, 10.0])
    smoothed = smooth_lead_distance(test_dists, window=3)
    print(f"\n  前车距离平滑测试:")
    print(f"    原始: {test_dists.tolist()}")
    print(f"    平滑: {[round(x, 1) for x in smoothed.tolist()]}")


# ============================================================
# 端到端测试 (从阶段1输出)
# ============================================================

def test_end_to_end_from_phase1(phase1_output: str, output_dir: str,
                                 max_samples: int = 0):
    """端到端测试: 加载阶段1采样 → 检测确认 → 合规 → 特征 → 输出"""
    print_header("测试3: 端到端流水线 (从阶段1输出)")

    from data_loader.pb_loader import load_frames_by_files
    from detection.detector import confirm_scene
    from compliance.checker import check_compliance
    from feature_extraction.base_extractor import extract_all_features
    from feature_extraction.general_features import extract_general_features
    from feature_extraction.scene_extractors import extract_scene_features
    from output.training_manifest import write_manifest
    from output.feature_writer import write_features

    # 1. 加载阶段1采样
    print(f"\n  加载阶段1输出: {phase1_output}")
    with open(phase1_output, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f"  总采样数: {len(samples)}")

    if not samples:
        print("  [结束] 无采样，请先运行 test_phase1_tag_screening.py")
        return

    if max_samples > 0 and len(samples) > max_samples:
        print(f"  限制测试数量: 取前 {max_samples} 个采样")
        samples = samples[:max_samples]

    scene_names = {1: "路口停车", 2: "起步", 3: "跟车", 4: "跟停", 5: "变道"}

    # 2. 逐采样处理
    print(f"\n  [2/5] 加载 pb + 检测确认...")
    for i, sample in enumerate(samples):
        pb_dir = sample.get('cloud_path', '')
        dir_key = sample.get('dir_key', '?')
        scene_type = sample.get('scene_type', 0)
        scene_name = scene_names.get(scene_type, f'scene{scene_type}')

        print(f"\n  [{i+1}/{len(samples)}] {dir_key} | Scene {scene_type} ({scene_name})")

        if not pb_dir or not os.path.isdir(pb_dir):
            print(f"    [跳过] pb 目录不存在: {pb_dir}")
            sample['confirmed'] = False
            sample['confirm_reason'] = f"pb 目录不存在"
            continue

        # 加载单个 pb 文件
        pb_file = sample['pb_file']
        frame_data = load_frames_by_files(pb_dir, [pb_file])
        if not frame_data:
            sample['confirmed'] = False
            sample['confirm_reason'] = "无有效帧数据"
            continue

        # 检测确认
        confirm_scene(sample, frame_data)
        status = "确认" if sample.get('confirmed') else "拒绝"
        reason = sample.get('confirm_reason', '')
        print(f"    检测: [{status}] {reason}")

        if not sample.get('confirmed'):
            continue

        # 合规检查
        print(f"  [3/5] 合规检查...")
        check_compliance(sample, frame_data)
        compliance_passed = sample.get('compliance', {}).get('passed', False)
        print(f"    合规: [{'通过' if compliance_passed else '不通过'}]")

        # 特征提取
        print(f"  [4/5] 特征提取...")
        features = extract_all_features(frame_data, [pb_file])
        if features is not None:
            sample['general_features'] = extract_general_features(features, features)
            sample['scene_features'] = extract_scene_features(features, sample['scene_type'])
            sample['features'] = features
            print(f"    特征: 通用 {len(sample['general_features'])} 个, "
                  f"场景 {len(sample['scene_features'])} 个")

    # 5. 输出
    print(f"\n  [5/5] 输出...")
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "training_manifest.json")
    write_manifest(samples, manifest_path)
    feature_path = write_features(samples, output_dir)

    # 汇总
    confirmed = sum(1 for s in samples if s.get('confirmed', False))
    compliant = sum(1 for s in samples
                    if s.get('compliance', {}).get('passed', False))

    print(f"\n  {'='*50}")
    print(f"  端到端完成:")
    print(f"    总采样:   {len(samples)}")
    print(f"    已确认:   {confirmed}")
    print(f"    合规:     {compliant}")
    print(f"    清单:     {manifest_path}")
    print(f"    特征:     {feature_path}")

    # 逐采样汇总
    if samples:
        print(f"\n  采样明细:")
        for sample in samples:
            scene_type = sample.get('scene_type', 0)
            name = scene_names.get(scene_type, f'scene{scene_type}')
            confirmed_mark = "Y" if sample.get('confirmed') else "N"
            compliant_mark = "Y" if sample.get('compliance', {}).get('passed') else "N"
            print(f"    {sample.get('dir_key', '?')[:50]}  "
                  f"S{scene_type}({name})  "
                  f"确认={confirmed_mark}  合规={compliant_mark}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="阶段2测试: 算法确认 + 合规检查 + 特征提取 (真实数据)"
    )
    parser.add_argument("--pb-file", type=str, default="",
                        help="单个 pb 文件路径 (测试1/2用)")
    parser.add_argument("--phase1-output", type=str, default="",
                        help="阶段1输出的采样文件路径 (默认 test/test_output/phase1_samples.json)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="端到端测试最大采样数 (0=不限制)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="输出目录")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "test", "test_output")

    # ========== 测试1/2: 使用真实 pb 文件 ==========

    pb_file = args.pb_file
    if not pb_file:
        ref_pb = os.path.join(DEFAULT_REF_DIR, "1771887926405091968.pb")
        if os.path.exists(ref_pb):
            pb_file = ref_pb

    if pb_file and os.path.exists(pb_file):
        raw, normed = test_pb_loading(pb_file)
        test_frame_utils(normed)
    else:
        print("[跳过] 无 pb 文件，测试1/2 跳过")
        print("  提示: 将 .pb 文件放入 reference/ 或使用 --pb-file 参数")

    # ========== 测试3: 端到端 (从阶段1输出) ==========

    phase1_output = args.phase1_output or DEFAULT_PHASE1_OUTPUT

    if os.path.exists(phase1_output):
        test_end_to_end_from_phase1(phase1_output, output_dir,
                                    max_samples=args.max_samples)
    else:
        print_header("端到端测试 (跳过)")
        print(f"  阶段1输出文件不存在: {phase1_output}")
        print(f"  请先运行: python test/test_phase1_tag_screening.py")

    # 总结
    print_header("阶段2测试完成")
    print(f"  输出目录: {output_dir}")
    if os.path.exists(output_dir):
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            size = os.path.getsize(fpath)
            print(f"    {f}: {size} bytes")


if __name__ == "__main__":
    main()
