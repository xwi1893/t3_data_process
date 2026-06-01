"""
following_distance_checker.py
跟停最小距离合规检查
"""

import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMPLIANCE_PARAMS


def check_following_distance(
    features: dict,
    min_distance: float = COMPLIANCE_PARAMS["min_following_distance"],
    min_lead_ratio: float = COMPLIANCE_PARAMS["min_lead_ratio"],
    max_avg_dist: float = COMPLIANCE_PARAMS["avg_lead_dist_max"],
) -> dict:
    """检查跟停过程中的前车距离是否合规

    检查:
    1. 前车存在比例 >= min_lead_ratio (60%)
    2. 平均前车距离 < max_avg_dist (50m)
    3. 最小前车距离 >= min_distance (5m)

    Args:
        features: 特征字典 (含 lead_dist)
        min_distance: 最小跟停距离 (米)
        min_lead_ratio: 前车存在比例下限
        max_avg_dist: 平均距离上限

    Returns:
        {passed: bool, reason: str, metrics: dict}
    """
    lead_dist = np.array(features.get("lead_dist", []))

    if len(lead_dist) < 3:
        return {"passed": False, "reason": "帧数不足", "metrics": {}}

    # 前车存在比例
    valid_mask = lead_dist > 0
    lead_ratio = float(np.sum(valid_mask)) / len(lead_dist)

    if lead_ratio < min_lead_ratio:
        return {
            "passed": False,
            "reason": f"前车比例不足: {lead_ratio:.2f} < {min_lead_ratio}",
            "metrics": {"lead_ratio": lead_ratio},
        }

    valid_dists = lead_dist[valid_mask]
    if len(valid_dists) == 0:
        return {
            "passed": False,
            "reason": "无有效前车距离",
            "metrics": {"lead_ratio": lead_ratio},
        }

    avg_dist = float(np.mean(valid_dists))
    min_dist = float(np.min(valid_dists))
    max_dist = float(np.max(valid_dists))

    # 平均距离检查
    if avg_dist > max_avg_dist:
        return {
            "passed": False,
            "reason": f"平均前车距离过大: {avg_dist:.1f}m > {max_avg_dist}m",
            "metrics": {
                "lead_ratio": lead_ratio,
                "avg_lead_dist": avg_dist,
                "min_lead_dist": min_dist,
            },
        }

    # 最小距离检查
    if min_dist < min_distance:
        return {
            "passed": False,
            "reason": f"跟停距离过近: {min_dist:.1f}m < {min_distance}m",
            "metrics": {
                "lead_ratio": lead_ratio,
                "avg_lead_dist": avg_dist,
                "min_lead_dist": min_dist,
            },
        }

    return {
        "passed": True,
        "reason": "跟停距离检查通过",
        "metrics": {
            "lead_ratio": lead_ratio,
            "avg_lead_dist": avg_dist,
            "min_lead_dist": min_dist,
            "max_lead_dist": max_dist,
        },
    }
