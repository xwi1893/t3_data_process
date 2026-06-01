"""
ttc_checker.py
TTC (Time-to-Collision) 合规检查

TTC = lead_dist / (ego_speed - lead_speed)
仅在 ego 接近前车 (relative_speed > 0) 且前车存在 (lead_dist > 0) 时有效
"""

import numpy as np
from typing import Dict, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMPLIANCE_PARAMS


def check_ttc(
    features: dict,
    ttc_max: float = COMPLIANCE_PARAMS["ttc_max"],
) -> dict:
    """计算并检查 TTC 指标

    Args:
        features: 特征字典 (含 speed, lead_dist, lead_speed)
        ttc_max: TTC 上限 (秒)

    Returns:
        {passed: bool, reason: str, metrics: dict}
    """
    speed = np.array(features.get("speed", []))
    lead_dist = np.array(features.get("lead_dist", []))
    lead_speed = np.array(features.get("lead_speed", []))

    if len(speed) < 3:
        return {"passed": False, "reason": "帧数不足", "metrics": {}}

    # 计算 TTC
    relative_speed = speed - lead_speed
    valid_mask = (relative_speed > 0) & (lead_dist > 0)

    ttc = np.full(len(speed), np.nan)
    ttc[valid_mask] = lead_dist[valid_mask] / relative_speed[valid_mask]

    # 截断
    ttc[ttc > ttc_max] = np.nan

    valid_ttc = ttc[~np.isnan(ttc)]
    valid_ratio = len(valid_ttc) / len(ttc) if len(ttc) > 0 else 0

    if len(valid_ttc) == 0:
        return {
            "passed": True,
            "reason": "无有效TTC (非接近状态)",
            "metrics": {"ttc_valid_ratio": 0.0},
        }

    ttc_5th = float(np.percentile(valid_ttc, 5))
    ttc_mean = float(np.mean(valid_ttc))
    ttc_std = float(np.std(valid_ttc))
    ttc_95th = float(np.percentile(valid_ttc, 95))

    return {
        "passed": True,  # TTC 检查主要提供统计指标
        "reason": "TTC统计完成",
        "metrics": {
            "ttc_5th": ttc_5th,
            "ttc_mean": ttc_mean,
            "ttc_std": ttc_std,
            "ttc_95th": ttc_95th,
            "ttc_valid_ratio": float(valid_ratio),
            "ttc_min": float(np.min(valid_ttc)),
        },
    }
