"""
checker.py
合规检查调度器: 根据场景类型分发到对应检查函数
"""

from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compliance.ttc_checker import check_ttc
from compliance.stopline_checker import check_stopline_crossing
from compliance.following_distance_checker import check_following_distance


def check_compliance(
    segment: dict,
    frame_data: Dict[str, dict],
) -> dict:
    """对已确认的场景片段执行合规检查

    根据场景类型分发到对应检查:
    - Scene 1 (路口停车): 停止线压线检查
    - Scene 3 (跟车): TTC有效性检查
    - Scene 4 (跟停): TTC统计 + 跟停距离检查

    Args:
        segment: 已确认的场景片段
        frame_data: {pb_filename: normalized_frame}

    Returns:
        更新后的 segment，增加 compliance 字段
    """
    scene_type = segment['scene_type']
    features = segment.get('features')

    compliance = {
        "checks": [],
        "passed": True,
        "fail_reasons": [],
    }

    if scene_type == 1:
        # 路口停车: 停止线检查
        result = check_stopline_crossing(
            frame_data, segment['pb_files']
        )
        compliance["checks"].append({
            "name": "stopline_crossing",
            **result,
        })
        if not result["passed"]:
            compliance["passed"] = False
            compliance["fail_reasons"].append(result["reason"])

    elif scene_type == 3:
        # 跟车: TTC 检查
        if features:
            result = check_ttc(features)
            compliance["checks"].append({
                "name": "ttc_check",
                **result,
            })
            if not result["passed"]:
                compliance["passed"] = False
                compliance["fail_reasons"].append(result["reason"])

    elif scene_type == 4:
        # 跟停: TTC + 跟停距离
        if features:
            # TTC
            ttc_result = check_ttc(features)
            compliance["checks"].append({
                "name": "ttc_check",
                **ttc_result,
            })

            # 跟停距离
            dist_result = check_following_distance(features)
            compliance["checks"].append({
                "name": "following_distance",
                **dist_result,
            })
            if not dist_result["passed"]:
                compliance["passed"] = False
                compliance["fail_reasons"].append(dist_result["reason"])

    # Scene 2 (起步) 和 Scene 5 (变道) 无特殊合规检查

    segment['compliance'] = {
        "passed": compliance["passed"],
        "fail_reasons": compliance["fail_reasons"],
        "metrics": {
            check["name"]: check.get("metrics", {})
            for check in compliance["checks"]
        },
    }

    status = "通过" if compliance["passed"] else "不合规"
    scene_names = {1: "路口停车", 2: "起步", 3: "跟车", 4: "跟停", 5: "变道"}
    print(f"[compliance] {scene_names.get(scene_type, '?')} "
          f"#{segment.get('segment_id', '?')}: {status}")

    return segment


def check_all_segments(segments: List[dict]) -> List[dict]:
    """对所有已确认的片段执行合规检查

    Args:
        segments: 已确认的场景片段列表

    Returns:
        更新后的片段列表
    """
    passed = 0
    failed = 0

    for segment in segments:
        if not segment.get('confirmed', False):
            continue

        frame_data_for_segment = {}
        # 从 segment 的 features 构建简单的 frame_data 引用
        # 实际使用时需要传入完整 frame_data
        check_compliance(segment, frame_data_for_segment)

        if segment.get('compliance', {}).get('passed', True):
            passed += 1
        else:
            failed += 1

    print(f"\n[compliance] 合规检查完成: {passed} 通过, {failed} 不合规")
    return segments
