"""
general_features.py
通用特征提取 (12个特征)
移植自参考项目 main_extra.py extract_general_features()
"""

import numpy as np
from typing import Dict


def extract_general_features(
    direct: Dict[str, np.ndarray],
    indirect: Dict[str, np.ndarray],
) -> dict:
    """提取通用特征

    12个特征:
    - 速度: avg, var, 95th, 5th
    - 加速度: avg, max, min, std
    - 横向加速度: avg, peak, std
    - 制动: max_brake_decel
    """
    speed = direct["speed"]
    acceleration = indirect["acceleration"]
    lateral_acc = indirect["lateral_acc"]

    return {
        # 速度
        "avg_speed": round(float(np.mean(speed)), 2),
        "speed_var": round(float(np.var(speed)), 2),
        "speed_95th": round(float(np.percentile(speed, 95)), 2),
        "speed_5th": round(float(np.percentile(speed, 5)), 2),
        # 加速度
        "avg_acceleration": round(float(np.mean(acceleration)), 2),
        "max_acceleration": round(float(np.max(acceleration)), 2),
        "min_acceleration": round(float(np.min(acceleration)), 2),
        "acceleration_std": round(float(np.std(acceleration)), 2),
        # 横向加速度
        "avg_lateral_acc": round(float(np.mean(lateral_acc)), 2),
        "peak_lateral_acc": round(float(np.max(np.abs(lateral_acc))), 2),
        "lateral_acc_std": round(float(np.std(lateral_acc)), 2),
        # 制动
        "max_brake_decel": round(float(np.min(acceleration)), 2),
    }
