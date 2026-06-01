"""
scene_extractors.py
5种场景专用特征提取器
移植自参考项目 main_extra.py
"""

import numpy as np
from typing import Dict, Optional


def _phase_means(arr: np.ndarray, n_phases: int = 5) -> list:
    """将数组等分为 n_phases 段，返回每段均值"""
    if len(arr) == 0:
        return [0.0] * n_phases

    phase_size = max(1, len(arr) // n_phases)
    means = []
    for i in range(n_phases):
        start = i * phase_size
        end = min((i + 1) * phase_size, len(arr))
        if i == n_phases - 1:
            end = len(arr)
        segment = arr[start:end]
        if len(segment) > 0:
            valid = segment[~np.isnan(segment)] if np.any(~np.isnan(segment)) else segment
            means.append(round(float(np.mean(valid)), 2) if len(valid) > 0 else 0.0)
        else:
            means.append(0.0)
    return means


# ============================================================
# Scene 1: 路口停车 (17个特征)
# ============================================================

def extract_scene1_features(features: dict) -> dict:
    """路口停车特征: 减速过程分析

    移植自参考项目 extract_scene1_features()
    """
    speed = features["speed"]
    acceleration = features["acceleration"]
    time_sec = features["time_sec"]

    if len(speed) < 5:
        return {"error": "帧数不足"}

    # 找减速起点 (最大速度帧)
    start_idx = int(np.argmax(speed))
    start_speed = speed[start_idx]

    # 目标速度 = 20% 初始速度
    target_speed = 0.2 * start_speed

    # 找达到目标速度的帧
    end_idx = None
    for i in range(start_idx, len(speed)):
        if speed[i] <= target_speed:
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(speed) - 1

    if end_idx - start_idx < 5:
        return {"error": "减速过程过短"}

    # 减速阶段切片
    decel_speed = speed[start_idx:end_idx + 1]
    decel_accel = acceleration[start_idx:end_idx + 1]
    decel_time = time_sec[end_idx] - time_sec[start_idx]

    # 5阶段特征
    speed_phases = _phase_means(decel_speed)
    accel_phases = _phase_means(decel_accel)

    avg_accel = float(np.mean(decel_accel))
    max_accel = float(np.min(decel_accel))  # 最负 = 最大减速

    return {
        "decel_time": round(decel_time, 2),
        "avg_acceleration": round(avg_accel, 2),
        "max_deceleration": round(max_accel, 2),
        "accel_std": round(float(np.std(decel_accel)), 2),
        "speed_phase1": speed_phases[0],
        "speed_phase2": speed_phases[1],
        "speed_phase3": speed_phases[2],
        "speed_phase4": speed_phases[3],
        "speed_phase5": speed_phases[4],
        "accel_phase1": accel_phases[0],
        "accel_phase2": accel_phases[1],
        "accel_phase3": accel_phases[2],
        "accel_phase4": accel_phases[3],
        "accel_phase5": accel_phases[4],
        "accel_efficiency": round(
            avg_accel / max_accel if max_accel != 0 else 0, 2
        ),
        "speed_ratio_at_midpoint": round(
            float(decel_speed[len(decel_speed) // 2]) / start_speed
            if start_speed > 0 else 0, 2
        ),
        "start_speed": round(float(start_speed), 2),
    }


# ============================================================
# Scene 2: 起步 (16个特征)
# ============================================================

def extract_scene2_features(features: dict) -> dict:
    """起步特征: 加速过程分析

    移植自参考项目 extract_scene2_features()
    """
    speed = features["speed"]
    acceleration = features["acceleration"]
    time_sec = features["time_sec"]

    if len(speed) < 5:
        return {"error": "帧数不足"}

    # 起步起点 (速度最低帧)
    start_idx = int(np.argmin(speed))
    max_speed = float(np.max(speed[start_idx:]))
    target_speed = 0.8 * max_speed

    # 找达到 80% 最大速度的帧
    end_idx = None
    for i in range(start_idx, len(speed)):
        if speed[i] >= target_speed:
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(speed) - 1

    if end_idx - start_idx < 3:
        return {"error": "加速过程过短"}

    accel_speed = speed[start_idx:end_idx + 1]
    accel_accel = acceleration[start_idx:end_idx + 1]
    accel_time = time_sec[end_idx] - time_sec[start_idx]

    speed_phases = _phase_means(accel_speed)
    accel_phases = _phase_means(accel_accel)

    avg_accel = float(np.mean(accel_accel))
    peak_accel = float(np.max(accel_accel))

    return {
        "time_to_80pct_max": round(accel_time, 2),
        "avg_acceleration": round(avg_accel, 2),
        "peak_acceleration": round(peak_accel, 2),
        "acceleration_std": round(float(np.std(accel_accel)), 2),
        "speed_phase1": speed_phases[0],
        "speed_phase2": speed_phases[1],
        "speed_phase3": speed_phases[2],
        "speed_phase4": speed_phases[3],
        "speed_phase5": speed_phases[4],
        "accel_phase1": accel_phases[0],
        "accel_phase2": accel_phases[1],
        "accel_phase3": accel_phases[2],
        "accel_phase4": accel_phases[3],
        "accel_phase5": accel_phases[4],
        "acceleration_efficiency": round(
            avg_accel / peak_accel if peak_accel != 0 else 0, 2
        ),
        "max_speed": round(max_speed, 2),
    }


# ============================================================
# Scene 3: 跟车 (20个特征)
# ============================================================

def extract_scene3_features(features: dict) -> dict:
    """跟车特征: TTC, THW, 相对速度

    移植自参考项目 extract_scene3_features()
    """
    speed = features["speed"]
    lead_dist = features["lead_dist"]
    lead_speed = features["lead_speed"]
    time_headway = features["time_headway"]

    if len(speed) < 3:
        return {"error": "帧数不足"}

    # 相对速度
    relative_speed = speed - lead_speed

    # TTC (仅在接近前车时有效)
    valid_ttc_mask = (relative_speed > 0) & (lead_dist > 0)
    ttc = np.full(len(speed), np.nan)
    ttc[valid_ttc_mask] = lead_dist[valid_ttc_mask] / relative_speed[valid_ttc_mask]
    ttc[ttc > 10] = np.nan

    valid_ttc = ttc[~np.isnan(ttc)]

    # 基本统计
    valid_lead = lead_dist[lead_dist > 0]

    result = {
        "avg_lead_dist": round(float(np.mean(valid_lead)), 2) if len(valid_lead) > 0 else 0,
        "lead_dist_var": round(float(np.var(valid_lead)), 2) if len(valid_lead) > 0 else 0,
        "min_lead_dist": round(float(np.min(valid_lead)), 2) if len(valid_lead) > 0 else 0,
        "avg_thw": round(float(np.mean(time_headway)), 2),
        "min_thw": round(float(np.min(time_headway)), 2),
        "thw_5th": round(float(np.percentile(time_headway, 5)), 2),
    }

    # TTC 特征
    if len(valid_ttc) > 0:
        result.update({
            "ttc_5th": round(float(np.percentile(valid_ttc, 5)), 2),
            "ttc_mean": round(float(np.mean(valid_ttc)), 2),
            "ttc_std": round(float(np.std(valid_ttc)), 2),
            "ttc_valid_ratio": round(len(valid_ttc) / len(ttc), 2),
        })
    else:
        result.update({
            "ttc_5th": None, "ttc_mean": None,
            "ttc_std": None, "ttc_valid_ratio": 0.0,
        })

    # 相对速度
    result.update({
        "relative_speed_mean": round(float(np.mean(relative_speed)), 2),
        "relative_speed_std": round(float(np.std(relative_speed)), 2),
        "relative_speed_max": round(float(np.max(relative_speed)), 2),
        "relative_speed_min": round(float(np.min(relative_speed)), 2),
    })

    # 5阶段
    result["relative_speed_phases"] = _phase_means(relative_speed)
    result["lead_dist_phases"] = _phase_means(lead_dist)

    return result


# ============================================================
# Scene 4: 跟停 (19个特征)
# ============================================================

def extract_scene4_features(features: dict) -> dict:
    """跟停特征: 增强版 TTC + 前车距离

    移植自参考项目 extract_scene4_features()
    """
    speed = features["speed"]
    lead_dist = features["lead_dist"]
    lead_speed = features["lead_speed"]
    time_headway = features["time_headway"]

    if len(speed) < 3:
        return {"error": "帧数不足"}

    # 静止阶段分析
    static_mask = speed < 0.1
    static_lead = lead_dist[static_mask]
    min_static_dist = float(np.min(static_lead[static_lead > 0])) if np.any(static_lead > 0) else 0

    # THW 增强
    valid_thw = time_headway[time_headway < 15]

    result = {
        "avg_lead_dist": round(float(np.mean(lead_dist[lead_dist > 0])), 2)
            if np.any(lead_dist > 0) else 0,
        "min_static_dist": round(min_static_dist, 2),
        "avg_thw": round(float(np.mean(valid_thw)), 2) if len(valid_thw) > 0 else 15.0,
        "min_thw": round(float(np.min(valid_thw)), 2) if len(valid_thw) > 0 else 15.0,
        "thw_5th": round(float(np.percentile(valid_thw, 5)), 2) if len(valid_thw) > 0 else 15.0,
        "thw_95th": round(float(np.percentile(valid_thw, 95)), 2) if len(valid_thw) > 0 else 15.0,
        "thw_std": round(float(np.std(valid_thw)), 2) if len(valid_thw) > 0 else 0,
    }

    # TTC (截断到 10s)
    relative_speed = speed - lead_speed
    valid_ttc_mask = (relative_speed > 0) & (lead_dist > 0)
    ttc = np.full(len(speed), np.nan)
    ttc[valid_ttc_mask] = lead_dist[valid_ttc_mask] / relative_speed[valid_ttc_mask]
    ttc[ttc > 10] = np.nan
    valid_ttc = ttc[~np.isnan(ttc)]

    if len(valid_ttc) > 0:
        result.update({
            "ttc_5th": round(float(np.percentile(valid_ttc, 5)), 2),
            "ttc_95th": round(float(np.percentile(valid_ttc, 95)), 2),
            "ttc_std": round(float(np.std(valid_ttc)), 2),
            "ttc_mean": round(float(np.mean(valid_ttc)), 2),
            "ttc_valid_ratio": round(len(valid_ttc) / len(ttc), 2),
        })
    else:
        result.update({
            "ttc_5th": None, "ttc_95th": None, "ttc_std": None,
            "ttc_mean": None, "ttc_valid_ratio": 0.0,
        })

    # 前车距离增强
    valid_lead = lead_dist[lead_dist > 0]
    result["lead_dist_max"] = round(float(np.max(valid_lead)), 2) if len(valid_lead) > 0 else 0
    result["lead_dist_std"] = round(float(np.std(valid_lead)), 2) if len(valid_lead) > 0 else 0

    # 5阶段前车距离
    result["lead_dist_phases"] = _phase_means(lead_dist)

    return result


# ============================================================
# Scene 5: 变道 (简化版, 核心特征)
# ============================================================

def extract_scene5_features(features: dict) -> dict:
    """变道特征: 参考帧变化 + 车道距离

    简化版移植自参考项目 extract_scene5_features()
    完整版需要 extract_lane_vehicles_for_frame (待后续补充)
    """
    speed = features["speed"]
    lead_dist = features["lead_dist"]
    lateral_pos = features["lateral_pos"]
    time_sec = features["time_sec"]

    if len(speed) < 5:
        return {"error": "帧数不足"}

    # 参考帧变化检测 (lead_dist 跳变 > 10m)
    ref_frame_idx = len(lead_dist) // 2
    for i in range(1, len(lead_dist)):
        if abs(lead_dist[i] - lead_dist[i - 1]) > 10:
            ref_frame_idx = i
            break

    # 变道前后统计
    before_lead = lead_dist[:ref_frame_idx]
    after_lead = lead_dist[ref_frame_idx:]
    before_lat = lateral_pos[:ref_frame_idx]
    after_lat = lateral_pos[ref_frame_idx:]

    lateral_shift = float(np.mean(after_lat) - np.mean(before_lat)) if (
        len(before_lat) > 0 and len(after_lat) > 0
    ) else 0

    result = {
        "reference_frame_idx": ref_frame_idx,
        "lateral_shift": round(lateral_shift, 2),
        "lateral_range": round(float(np.max(lateral_pos) - np.min(lateral_pos)), 2),
        "min_lead_dist_during_lc": round(float(np.min(lead_dist[lead_dist > 0])), 2)
            if np.any(lead_dist > 0) else 0,
        # 前车距离前后统计
        "before_lead_mean": round(float(np.mean(before_lead[before_lead > 0])), 2)
            if np.any(before_lead > 0) else 0,
        "after_lead_mean": round(float(np.mean(after_lead[after_lead > 0])), 2)
            if np.any(after_lead > 0) else 0,
        "before_lead_count": int(np.sum(before_lead > 0)),
        "after_lead_count": int(np.sum(after_lead > 0)),
    }

    # 5阶段特征
    result["speed_phases"] = _phase_means(speed)
    result["lateral_pos_phases"] = _phase_means(lateral_pos)
    result["lead_dist_phases"] = _phase_means(lead_dist)

    return result


# ============================================================
# 场景特征分发
# ============================================================

_SCENE_EXTRACTORS = {
    1: extract_scene1_features,
    2: extract_scene2_features,
    3: extract_scene3_features,
    4: extract_scene4_features,
    5: extract_scene5_features,
}


def extract_scene_features(
    features: dict,
    scene_type: int,
) -> Optional[dict]:
    """根据场景类型分发到对应特征提取器

    Args:
        features: 完整特征 (direct + indirect)
        scene_type: 场景类型 1-5

    Returns:
        场景专用特征字典
    """
    extractor = _SCENE_EXTRACTORS.get(scene_type)
    if extractor is None:
        return {"error": f"未知场景类型: {scene_type}"}
    return extractor(features)
