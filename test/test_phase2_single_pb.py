"""
test_phase2_single_pb.py
单 pb 文件全流程测试: 场景识别 + 合规检查 + 特征提取

输入: 单个 .pb 文件路径
流程:
  1. 加载并归一化 pb 帧
  2. 提取时序特征 (历史 + 当前 + 未来)
  3. 依次调用 5 个场景确认函数，判断属于哪个场景
  4. 对确认的场景执行合规检查
  5. 提取通用特征和场景专用特征并输出

用法:
  # 使用默认测试 pb 文件
  python test/test_phase2_single_pb.py

  # 指定 pb 文件路径
  python test/test_phase2_single_pb.py --pb-file /path/to/frame.pb

  # 使用 test/downloads 下的测试数据
  python test/test_phase2_single_pb.py --pb-file test/downloads/scene_000_lane_change/1772754911605091968.pb
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    print(f"\n  --- {title} ---")


def print_kv(key: str, value, indent: int = 4):
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


# ============================================================
# Step 1: 加载 pb 帧
# ============================================================

def step1_load_pb(pb_path: str):
    """加载并归一化单个 pb 文件"""
    print_header("Step 1: 加载 pb 帧")

    from data_loader.pb_loader import load_pb_frame, normalize_frame

    print(f"\n  pb 文件: {pb_path}")
    print(f"  文件大小: {os.path.getsize(pb_path) / 1024:.1f} KB")

    raw = load_pb_frame(pb_path)
    print(f"  原始字段数: {len(raw)}")

    normed = normalize_frame(raw)
    print(f"  归一化字段: {list(normed.keys())}")

    # 自车状态摘要
    ego = normed.get('data_ego_curr_status', {})
    print(f"\n  自车状态:")
    print_kv("速度", f"{ego.get('v', 0):.2f} m/s")
    print_kv("航向角", f"{ego.get('yaw', 0):.4f} rad")
    print_kv("角速度", f"{ego.get('yaw_rate', 0):.4f} rad/s")

    agents = normed.get('data_agent', [])
    print(f"\n  周围目标: {len(agents)} 个")

    lanes = normed.get('data_laneline', [])
    print(f"  车道线: {len(lanes)} 条")

    hist_traj = normed.get('label_ego_hist_traj', {})
    fut_traj = normed.get('label_ego_traj', {})
    hist_n = len(hist_traj.get('mask', []))
    fut_n = len(fut_traj.get('mask', []))
    print(f"  历史轨迹: {hist_n} 帧")
    print(f"  未来轨迹: {fut_n} 帧")

    return normed


# ============================================================
# Step 2: 提取时序特征
# ============================================================

def step2_extract_features(normed: dict):
    """从单帧数据提取时序特征 (历史+当前+未来)"""
    print_header("Step 2: 提取时序特征")

    from detection.scene_detectors import extract_segment_features

    pb_filename = os.path.basename(normed.get('_raw', {}).get('timestamp_ns', 'unknown'))
    # 用文件名作为 key 构建 frame_data
    frame_data = {pb_filename: normed}
    pb_files = [pb_filename]

    features = extract_segment_features(frame_data, pb_files)
    if features is None:
        print("  [错误] 特征提取失败")
        return None

    print(f"\n  特征字段: {list(features.keys())}")
    print(f"  总帧数 (含历史+未来): {features.get('frame_count', '?')}")

    speed = features.get('speed', np.array([]))
    if len(speed) > 0:
        print(f"\n  速度序列:")
        print_kv("长度", f"{len(speed)} 帧")
        print_kv("范围", f"[{np.min(speed):.2f}, {np.max(speed):.2f}] m/s")
        print_kv("均值", f"{np.mean(speed):.2f} m/s")

    lead_dist = features.get('lead_dist', np.array([]))
    if len(lead_dist) > 0:
        valid = lead_dist[lead_dist > 0]
        print(f"\n  前车距离:")
        print_kv("有效帧比例", f"{len(valid)}/{len(lead_dist)} "
                 f"({len(valid)/len(lead_dist)*100:.0f}%)")
        if len(valid) > 0:
            print_kv("范围", f"[{np.min(valid):.1f}, {np.max(valid):.1f}] m")
            print_kv("均值", f"{np.mean(valid):.1f} m")

    lateral_pos = features.get('lateral_pos', np.array([]))
    if len(lateral_pos) > 0:
        print(f"\n  横向位置:")
        print_kv("范围", f"[{np.min(lateral_pos):.3f}, {np.max(lateral_pos):.3f}] m")
        print_kv("变化量", f"{np.max(lateral_pos) - np.min(lateral_pos):.3f} m")

    print(f"\n  静态布尔特征:")
    print_kv("is_at_intersection", features.get('is_at_intersection', False))
    print_kv("has_stopline", features.get('has_stopline', False))
    print_kv("has_traffic_light", features.get('has_traffic_light', False))
    print_kv("is_turning", features.get('is_turning', False))

    return features


# ============================================================
# Step 3: 5 场景确认 — 逐一尝试
# ============================================================

SCENE_NAMES = {
    1: "路口停车",
    2: "起步",
    3: "跟车",
    4: "跟停",
    5: "变道",
}


def step3_identify_scene(features: dict):
    """依次调用 5 个场景确认函数，返回所有确认通过的场景"""
    print_header("Step 3: 场景识别 (5 场景逐一确认)")

    from detection.scene_detectors import (
        confirm_intersection_stop,
        confirm_empty_start,
        confirm_following_vehicle,
        confirm_following_stop,
        confirm_lane_change,
    )

    confirm_funcs = {
        1: confirm_intersection_stop,
        2: confirm_empty_start,
        3: confirm_following_vehicle,
        4: confirm_following_stop,
        5: confirm_lane_change,
    }

    results = {}
    matched_scenes = []

    for scene_type, func in confirm_funcs.items():
        scene_name = SCENE_NAMES[scene_type]
        result = func(features)
        confirmed = result.get('confirmed', False)
        reason = result.get('reason', '')
        metrics = result.get('metrics', {})

        results[scene_type] = {
            'confirmed': confirmed,
            'reason': reason,
            'metrics': metrics,
            'method': func.__name__,
        }

        status = "✓ 通过" if confirmed else "✗ 拒绝"
        print(f"\n  Scene {scene_type} ({scene_name}): {status}")
        print_kv("原因", reason)
        if metrics:
            print_kv("指标", "")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"        {k}: {v:.4f}")
                else:
                    print(f"        {k}: {v}")

        if confirmed:
            matched_scenes.append(scene_type)

    # 汇总
    print(f"\n  {'='*50}")
    if matched_scenes:
        scene_labels = [f"Scene {s} ({SCENE_NAMES[s]})" for s in matched_scenes]
        print(f"  匹配场景: {', '.join(scene_labels)}")
    else:
        print(f"  未匹配任何场景")

    return results, matched_scenes


# ============================================================
# Step 4: 合规检查
# ============================================================

def step4_compliance_check(features: dict, matched_scenes: list, normed: dict):
    """对匹配的场景执行合规检查"""
    print_header("Step 4: 合规检查")

    from compliance.ttc_checker import check_ttc
    from compliance.stopline_checker import check_stopline_crossing
    from compliance.following_distance_checker import check_following_distance

    if not matched_scenes:
        print("  无匹配场景，跳过合规检查")
        return {}

    compliance_results = {}

    for scene_type in matched_scenes:
        scene_name = SCENE_NAMES[scene_type]
        print(f"\n  Scene {scene_type} ({scene_name}) 合规检查:")

        checks = {}

        if scene_type == 1:
            # 路口停车: 停止线检查
            pb_filename = os.path.basename(
                normed.get('_raw', {}).get('timestamp_ns', 'unknown')
            )
            frame_data = {pb_filename: normed}
            result = check_stopline_crossing(frame_data, [pb_filename])
            checks['stopline_crossing'] = result
            status = "通过" if result.get('passed') else "不通过"
            print(f"    停止线压线检查: [{status}] {result.get('reason', '')}")
            if result.get('metrics'):
                for k, v in result['metrics'].items():
                    print(f"      {k}: {v}")

        elif scene_type == 3:
            # 跟车: TTC 检查
            result = check_ttc(features)
            checks['ttc_check'] = result
            status = "通过" if result.get('passed') else "不通过"
            print(f"    TTC 检查: [{status}] {result.get('reason', '')}")
            if result.get('metrics'):
                for k, v in result['metrics'].items():
                    if isinstance(v, float):
                        print(f"      {k}: {v:.4f}")
                    else:
                        print(f"      {k}: {v}")

        elif scene_type == 4:
            # 跟停: TTC + 跟停距离
            ttc_result = check_ttc(features)
            checks['ttc_check'] = ttc_result
            status = "通过" if ttc_result.get('passed') else "不通过"
            print(f"    TTC 检查: [{status}] {ttc_result.get('reason', '')}")

            dist_result = check_following_distance(features)
            checks['following_distance'] = dist_result
            status = "通过" if dist_result.get('passed') else "不通过"
            print(f"    跟停距离检查: [{status}] {dist_result.get('reason', '')}")
            if dist_result.get('metrics'):
                for k, v in dist_result['metrics'].items():
                    if isinstance(v, float):
                        print(f"      {k}: {v:.4f}")
                    else:
                        print(f"      {k}: {v}")

        elif scene_type in (2, 5):
            print(f"    (Scene {scene_type} 无特殊合规检查)")

        compliance_results[scene_type] = checks

    return compliance_results


# ============================================================
# Step 5: 特征提取并输出
# ============================================================

def step5_feature_extraction(features: dict, matched_scenes: list):
    """提取通用特征和场景专用特征"""
    print_header("Step 5: 特征提取")

    from feature_extraction.general_features import extract_general_features
    from feature_extraction.scene_extractors import extract_scene_features

    # 通用特征
    print_section("通用特征 (14个)")
    general = extract_general_features(features, features)

    print(f"\n  速度特征:")
    print_kv("avg_speed", f"{general['avg_speed']} m/s")
    print_kv("speed_var", f"{general['speed_var']}")
    print_kv("speed_95th", f"{general['speed_95th']} m/s")
    print_kv("speed_5th", f"{general['speed_5th']} m/s")

    print(f"\n  加速度特征:")
    print_kv("avg_acceleration", f"{general['avg_acceleration']} m/s²")
    print_kv("max_acceleration", f"{general['max_acceleration']} m/s²")
    print_kv("min_acceleration", f"{general['min_acceleration']} m/s²")
    print_kv("acceleration_std", f"{general['acceleration_std']}")

    print(f"\n  横向加速度特征:")
    print_kv("avg_lateral_acc", f"{general['avg_lateral_acc']}")
    print_kv("peak_lateral_acc", f"{general['peak_lateral_acc']}")
    print_kv("lateral_acc_std", f"{general['lateral_acc_std']}")

    print(f"\n  制动特征:")
    print_kv("max_brake_decel", f"{general['max_brake_decel']}")

    # 场景专用特征
    scene_features = {}
    for scene_type in matched_scenes:
        scene_name = SCENE_NAMES[scene_type]
        print_section(f"场景专用特征: Scene {scene_type} ({scene_name})")

        sf = extract_scene_features(features, scene_type)
        scene_features[scene_type] = sf

        if 'error' in sf:
            print(f"\n  [错误] {sf['error']}")
            continue

        print(f"\n  特征数量: {len(sf)} 个")
        for k, v in sf.items():
            if isinstance(v, list):
                # 列表类型 (如 5 阶段特征)
                formatted = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in v]
                print(f"    {k}: [{', '.join(formatted)}]")
            elif isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    return general, scene_features


# ============================================================
# 汇总输出
# ============================================================

def print_summary(features, matched_scenes, scene_results,
                  compliance_results, general_features, scene_features):
    """打印最终汇总"""
    print_header("汇总")

    print(f"\n  场景识别结果:")
    for scene_type in sorted(scene_results.keys()):
        r = scene_results[scene_type]
        mark = "✓" if r['confirmed'] else "✗"
        print(f"    Scene {scene_type} ({SCENE_NAMES[scene_type]}): "
              f"{mark} - {r['reason']}")

    if matched_scenes:
        print(f"\n  最终匹配场景: "
              f"{', '.join(f'Scene {s} ({SCENE_NAMES[s]})' for s in matched_scenes)}")

        print(f"\n  合规检查:")
        for scene_type in matched_scenes:
            checks = compliance_results.get(scene_type, {})
            if not checks:
                print(f"    Scene {scene_type} ({SCENE_NAMES[scene_type]}): 无特殊合规检查")
            else:
                all_passed = all(c.get('passed', False) for c in checks.values())
                status = "全部通过" if all_passed else "存在不合规"
                print(f"    Scene {scene_type} ({SCENE_NAMES[scene_type]}): {status}")

        print(f"\n  特征提取:")
        print(f"    通用特征: {len(general_features)} 个")
        for scene_type in matched_scenes:
            sf = scene_features.get(scene_type, {})
            n = len(sf) if 'error' not in sf else 0
            print(f"    Scene {scene_type} 场景特征: {n} 个")
    else:
        print(f"\n  未匹配任何场景，无合规检查和场景特征输出")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="单 pb 文件全流程测试: 场景识别 + 合规检查 + 特征提取"
    )
    parser.add_argument("--pb-file", type=str, default="",
                        help="pb 文件路径")
    parser.add_argument("--output-dir", type=str, default="",
                        help="结果输出目录 (默认不写文件，仅打印)")
    args = parser.parse_args()

    # 确定 pb 文件
    pb_path = args.pb_file
    if not pb_path:
        # 尝试 reference 目录
        ref_dir = os.path.join(PROJECT_ROOT, "reference")
        ref_pbs = [f for f in os.listdir(ref_dir) if f.endswith('.pb')] if os.path.isdir(ref_dir) else []
        if ref_pbs:
            pb_path = os.path.join(ref_dir, ref_pbs[0])
            print(f"  未指定 --pb-file，使用 reference 目录: {os.path.basename(pb_path)}")
        else:
            # 尝试 test/downloads 目录
            downloads_dir = os.path.join(PROJECT_ROOT, "test", "downloads")
            if os.path.isdir(downloads_dir):
                for scene_dir in sorted(os.listdir(downloads_dir)):
                    scene_path = os.path.join(downloads_dir, scene_dir)
                    if os.path.isdir(scene_path):
                        pbs = [f for f in os.listdir(scene_path) if f.endswith('.pb')]
                        if pbs:
                            pb_path = os.path.join(scene_path, sorted(pbs)[0])
                            print(f"  未指定 --pb-file，使用测试数据: "
                                  f"{scene_dir}/{sorted(pbs)[0]}")
                            break

    if not pb_path or not os.path.exists(pb_path):
        print("[错误] 未找到 pb 文件。请通过 --pb-file 指定路径")
        print("  示例: python test/test_phase2_single_pb.py --pb-file path/to/frame.pb")
        sys.exit(1)

    print(f"\n  {'='*56}")
    print(f"  单 pb 文件全流程测试")
    print(f"  pb 文件: {pb_path}")
    print(f"  {'='*56}")

    # Step 1: 加载
    normed = step1_load_pb(pb_path)

    # Step 2: 特征提取
    features = step2_extract_features(normed)
    if features is None:
        print("\n[终止] 特征提取失败，无法继续")
        sys.exit(1)

    # Step 3: 场景识别
    scene_results, matched_scenes = step3_identify_scene(features)

    # Step 4: 合规检查
    compliance_results = step4_compliance_check(features, matched_scenes, normed)

    # Step 5: 特征提取
    general_features, scene_features = step5_feature_extraction(features, matched_scenes)

    # 汇总
    print_summary(features, matched_scenes, scene_results,
                  compliance_results, general_features, scene_features)

    # 可选: 写入 JSON
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        result = {
            "pb_file": pb_path,
            "matched_scenes": [
                {
                    "scene_type": s,
                    "scene_name": SCENE_NAMES[s],
                    "detection": scene_results[s],
                    "compliance": {
                        k: {
                            "passed": v.get("passed"),
                            "reason": v.get("reason", ""),
                            "metrics": v.get("metrics", {}),
                        }
                        for k, v in compliance_results.get(s, {}).items()
                    },
                    "scene_features": scene_features.get(s, {}),
                }
                for s in matched_scenes
            ],
            "general_features": general_features,
        }

        output_path = os.path.join(output_dir, "single_pb_result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
        print(f"\n  结果已写入: {output_path}")

    print_header("测试完成")


if __name__ == "__main__":
    main()
