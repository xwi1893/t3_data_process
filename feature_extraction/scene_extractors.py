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
# Scene 1: 路口停车 (6个特征)
# ============================================================

def extract_scene1_features(features: dict) -> dict:
    """路口停车特征: 减速过程分析

    输出特征 (6个):
    - scene1_stop_avg_deceleration: 停车平均减速度
    - scene1_stop_decel_phase2~5: 停车第2~5阶段减速度
    - scene1_stop_max_deceleration: 停车最大减速度
    """
    speed = features["speed"]
    acceleration = features["acceleration"]

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

    decel_accel = acceleration[start_idx:end_idx + 1]

    # 5阶段减速度均值 (只取 phase2~5)
    accel_phases = _phase_means(decel_accel)

    avg_decel = float(np.mean(decel_accel))
    max_decel = float(np.min(decel_accel))  # 最负 = 最大减速

    return {
        "scene1_stop_avg_deceleration": round(avg_decel, 2),
        "scene1_stop_decel_phase2": accel_phases[1],
        "scene1_stop_decel_phase3": accel_phases[2],
        "scene1_stop_decel_phase4": accel_phases[3],
        "scene1_stop_decel_phase5": accel_phases[4],
        "scene1_stop_max_deceleration": round(max_decel, 2),
    }


# ============================================================
# Scene 2: 空旷起步 (6个特征)
# ============================================================

def extract_scene2_features(features: dict) -> dict:
    """起步特征: 加速过程分析

    输出特征 (6个):
    - scene2_empty_start_acceleration_phase1/2/4/5: 起步各阶段加速度
    - scene2_empty_start_peak_acceleration: 空旷起步峰值加速度
    - scene2_empty_start_speed_phase1: 起步第1阶段/早期速度
    """
    speed = features["speed"]
    acceleration = features["acceleration"]

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

    accel_accel = acceleration[start_idx:end_idx + 1]
    accel_speed = speed[start_idx:end_idx + 1]

    accel_phases = _phase_means(accel_accel)
    speed_phases = _phase_means(accel_speed)

    peak_accel = float(np.max(accel_accel))

    return {
        "scene2_empty_start_acceleration_phase1": accel_phases[0],
        "scene2_empty_start_acceleration_phase2": accel_phases[1],
        "scene2_empty_start_acceleration_phase4": accel_phases[3],
        "scene2_empty_start_acceleration_phase5": accel_phases[4],
        "scene2_empty_start_peak_acceleration": round(peak_accel, 2),
        "scene2_empty_start_speed_phase1": speed_phases[0],
    }


# ============================================================
# Scene 3: 跟车 (8个特征)
# ============================================================

def extract_scene3_features(features: dict) -> dict:
    """跟车特征: 前车距离分阶段 + TTC标准差

    输出特征 (8个):
    - scene3_following_avg_lead_dist: 跟车平均前车距离
    - scene3_following_lead_dist_phase1~5: 跟车各阶段前车距离
    - scene3_following_ttc_std: 跟车TTC标准差
    """
    speed = features["speed"]
    lead_dist = features["lead_dist"]
    lead_speed = features["lead_speed"]

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

    # 平均前车距离
    valid_lead = lead_dist[lead_dist > 0]
    avg_lead_dist = round(float(np.mean(valid_lead)), 2) if len(valid_lead) > 0 else 0

    # 5阶段前车距离
    lead_dist_phases = _phase_means(lead_dist)

    # TTC 标准差
    ttc_std = round(float(np.std(valid_ttc)), 2) if len(valid_ttc) > 0 else 0

    return {
        "scene3_following_avg_lead_dist": avg_lead_dist,
        "scene3_following_lead_dist_phase1": lead_dist_phases[0],
        "scene3_following_lead_dist_phase2": lead_dist_phases[1],
        "scene3_following_lead_dist_phase3": lead_dist_phases[2],
        "scene3_following_lead_dist_phase4": lead_dist_phases[3],
        "scene3_following_lead_dist_phase5": lead_dist_phases[4],
        "scene3_following_ttc_std": ttc_std,
    }


# ============================================================
# Scene 4: 跟停 (8个特征)
# ============================================================

def extract_scene4_features(features: dict) -> dict:
    """跟停特征: 前车距离分阶段 + THW/TTC标准差

    输出特征 (8个):
    - scene4_following_stop_lead_dist_phase1~5: 跟停各阶段前车距离
    - scene4_following_stop_thw_std: 跟停THW标准差
    - scene4_following_stop_ttc_std: 跟停TTC标准差
    """
    speed = features["speed"]
    lead_dist = features["lead_dist"]
    lead_speed = features["lead_speed"]
    time_headway = features["time_headway"]

    if len(speed) < 3:
        return {"error": "帧数不足"}

    # THW 标准差
    valid_thw = time_headway[time_headway < 15]
    thw_std = round(float(np.std(valid_thw)), 2) if len(valid_thw) > 0 else 0

    # TTC 标准差 (截断到 10s)
    relative_speed = speed - lead_speed
    valid_ttc_mask = (relative_speed > 0) & (lead_dist > 0)
    ttc = np.full(len(speed), np.nan)
    ttc[valid_ttc_mask] = lead_dist[valid_ttc_mask] / relative_speed[valid_ttc_mask]
    ttc[ttc > 10] = np.nan
    valid_ttc = ttc[~np.isnan(ttc)]
    ttc_std = round(float(np.std(valid_ttc)), 2) if len(valid_ttc) > 0 else 0

    # 5阶段前车距离
    lead_dist_phases = _phase_means(lead_dist)

    return {
        "scene4_following_stop_lead_dist_phase1": lead_dist_phases[0],
        "scene4_following_stop_lead_dist_phase2": lead_dist_phases[1],
        "scene4_following_stop_lead_dist_phase3": lead_dist_phases[2],
        "scene4_following_stop_lead_dist_phase4": lead_dist_phases[3],
        "scene4_following_stop_lead_dist_phase5": lead_dist_phases[4],
        "scene4_following_stop_thw_std": thw_std,
        "scene4_following_stop_ttc_std": ttc_std,
    }


# ============================================================
# Scene 5: 变道 (7个特征)
# ============================================================

def extract_scene5_features(features: dict) -> dict:
    """变道特征: 变道前后前车距离 + 目标车道后车距离

    变道时刻检测：使用横向位置(lateral_pos)的变化率峰值来定位，
    而非前车距离跳变——前车距离跳变可能由传感器噪声或前车驶离引起，
    不能可靠表征变道时刻。

    前车区分：
    - lead_dist: 当前车道前车距离（变道后自动切换为新车道前车）
    - origin_lead_dist: 原车道前车距离（始终追踪变道前的那辆前车）

    输出特征 (7个):
    - scene5_lane_change_before_lead_dist_mean: 变道前本车道前车平均距离
    - scene5_lane_change_before_lead_dist_min: 变道前本车道前车最小距离
    - scene5_lane_change_after_lead_dist_mean: 变道后前车平均距离（新车道前车）
    - scene5_lane_change_after_origin_front_mean: 变道后原车道前车距离均值
    - scene5_lane_change_min_lead_dist_during_lc: 变道过程中最小前车距离
    - scene5_lane_change_lc_target_rear_phase2: 变道第2阶段目标车道后车距离
    - scene5_lane_change_lc_target_rear_phase5: 变道第5阶段/末段目标车道后车距离
    """
    speed = features["speed"]
    lead_dist = features["lead_dist"]
    lateral_pos = features["lateral_pos"]
    origin_lead_dist = features.get("origin_lead_dist", lead_dist)
    target_rear_dist = features.get("target_rear_dist", np.zeros(len(speed)))

    n = len(speed)
    if n < 5:
        return {"error": "帧数不足"}

    # ── 变道时刻检测：横向位置变化率峰值 ──
    # 计算横向位置的差分（近似横向速度），取绝对值最大的帧作为变道中点
    if len(lateral_pos) >= 3:
        lat_diff = np.abs(np.diff(lateral_pos))
        # 平滑差分以抑制噪声
        if len(lat_diff) >= 3:
            kernel = np.ones(3) / 3
            lat_diff = np.convolve(lat_diff, kernel, mode='same')
        lc_idx = int(np.argmax(lat_diff)) + 1  # +1 因为 diff 比 original 少一帧
    else:
        lc_idx = n // 2

    # 变道前后切片
    before_lead = lead_dist[:lc_idx]
    after_lead = lead_dist[lc_idx:]
    after_origin = origin_lead_dist[lc_idx:]

    # 变道前本车道前车距离 (lead_dist 在变道前追踪的就是原车道前车)
    valid_before = before_lead[before_lead > 0]
    before_mean = round(float(np.mean(valid_before)), 2) if len(valid_before) > 0 else 0
    before_min = round(float(np.min(valid_before)), 2) if len(valid_before) > 0 else 0

    # 变道后前车距离 (lead_dist 已切换为新车道前车)
    valid_after = after_lead[after_lead > 0]
    after_mean = round(float(np.mean(valid_after)), 2) if len(valid_after) > 0 else 0

    # 变道后原车道前车距离 (origin_lead_dist 始终追踪变道前的那辆前车)
    valid_origin = after_origin[after_origin > 0]
    after_origin_front_mean = (
        round(float(np.mean(valid_origin)), 2) if len(valid_origin) > 0 else 0
    )

    # 变道过程中最小前车距离
    valid_lead = lead_dist[lead_dist > 0]
    min_during_lc = round(float(np.min(valid_lead)), 2) if len(valid_lead) > 0 else 0

    # 目标车道后车距离：自车变道后所在车道与后车的距离，无后车为 0
    # target_rear_dist 已在特征提取阶段通过检测目标车道后方车辆得到
    target_rear_dist_arr = np.array(target_rear_dist)
    target_rear_phases = _phase_means(target_rear_dist_arr)

    return {
        "scene5_lane_change_before_lead_dist_mean": before_mean,
        "scene5_lane_change_before_lead_dist_min": before_min,
        "scene5_lane_change_after_lead_dist_mean": after_mean,
        "scene5_lane_change_after_origin_front_mean": after_origin_front_mean,
        "scene5_lane_change_min_lead_dist_during_lc": min_during_lc,
        "scene5_lane_change_lc_target_rear_phase2": target_rear_phases[1],
        "scene5_lane_change_lc_target_rear_phase5": target_rear_phases[4],
    }


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
