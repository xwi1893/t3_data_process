"""
scene_detectors.py
移植参考项目的5种场景检测/过滤算法，用于候选片段的二次确认

每种场景的检测流程:
1. 从归一化帧数据中提取特征 (extract_segment_features)
2. 运行对应的 detect_* 函数检测事件
3. 运行对应的 filter_* 函数验证
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DETECTION_PARAMS, FEATURE_PARAMS
from utils.frame_utils import (
    get_lead_vehicle, calculate_lateral_position,
    check_has_stopline, check_has_traffic_light,
    smooth_lead_distance,
)


# ============================================================
# 特征提取 (供检测器使用)
# ============================================================

def extract_segment_features(
    frame_data: Dict[str, dict],
    pb_files: List[str],
    fps: int = 10,
) -> Optional[dict]:
    """从候选片段的帧数据中提取检测所需特征

    移植自参考项目 tiqvtest.py extract_raw_60s_features()

    Args:
        frame_data: {pb_filename: normalized_frame}
        pb_files: 片段中的 pb 文件名列表 (有序)
        fps: 帧率

    Returns:
        特征字典，或 None (有效帧不足)
    """
    valid_frames = []
    for f in pb_files:
        if f in frame_data:
            valid_frames.append((f, frame_data[f]))

    if len(valid_frames) < 3:
        return None

    time_list = []
    speed_list = []
    yaw_rate_list = []
    yaw_list = []
    lead_dist_list = []
    lead_speed_list = []
    lateral_pos_list = []
    lane_width_list = []
    curvature_list = []
    is_at_intersection_list = []
    has_stopline_list = []
    has_traffic_light_list = []
    is_turning_list = []
    is_u_turn_list = []

    base_time = None

    for i, (fname, frame) in enumerate(valid_frames):
        ego = frame.get('data_ego_curr_status', {})
        agents = frame.get('data_agent', [])
        lanelines = frame.get('data_laneline', [])

        v = ego.get('v', 0.0)
        yaw = ego.get('yaw', 0.0)
        yr = ego.get('yaw_rate', 0.0)

        # 前车检测
        lead_agent, lead_dist, lead_spd = get_lead_vehicle(agents, lanelines, yaw)

        # 横向位置
        lat_pos, curv = calculate_lateral_position(lanelines)

        # 车道宽度
        _, _, lw = get_lead_vehicle.__code__  # 简化: 使用默认
        from utils.frame_utils import get_lane_boundaries
        _, _, lw = get_lane_boundaries(lanelines)

        # 路口检测
        has_sl = check_has_stopline(frame)
        has_tl = check_has_traffic_light(frame)
        is_intersection = has_sl or has_tl

        # 转弯检测
        yaw_rate_thresh = 0.1
        curv_thresh = 0.08
        is_turn = abs(yr) > yaw_rate_thresh or abs(curv) > curv_thresh

        # U-turn 检测 (简化: 航向角大幅变化)
        is_uturn = False

        t_sec = i / fps if base_time is None else (i / fps)
        if base_time is None:
            base_time = 0

        time_list.append(t_sec)
        speed_list.append(v)
        yaw_rate_list.append(yr)
        yaw_list.append(yaw)
        lead_dist_list.append(lead_dist if lead_agent is not None else 0.0)
        lead_speed_list.append(lead_spd)
        lateral_pos_list.append(lat_pos)
        lane_width_list.append(lw if lw > 0 else 3.5)
        curvature_list.append(curv)
        is_at_intersection_list.append(is_intersection)
        has_stopline_list.append(has_sl)
        has_traffic_light_list.append(has_tl)
        is_turning_list.append(is_turn)
        is_u_turn_list.append(is_uturn)

    speed_arr = np.array(speed_list)
    lead_dist_arr = np.array(lead_dist_list)

    # 平滑前车距离
    lead_dist_smooth = smooth_lead_distance(
        lead_dist_arr, FEATURE_PARAMS.get("smooth_lead_dist_window", 5)
    )

    # 计算加速度
    accel = np.zeros(len(speed_arr))
    if len(speed_arr) > 1:
        accel[1:] = np.diff(speed_arr) * fps
    # 平滑加速度
    if len(accel) >= 3:
        kernel = np.ones(3) / 3
        accel = np.convolve(accel, kernel, mode='same')

    # 时距
    time_headway = np.full(len(speed_arr), 15.0)
    valid_speed = speed_arr > 0.1
    time_headway[valid_speed] = np.minimum(
        lead_dist_smooth[valid_speed] / speed_arr[valid_speed], 15.0
    )

    return {
        'time': time_list,
        'speed': speed_list,
        'acceleration': accel.tolist(),
        'lead_dist': lead_dist_smooth.tolist(),
        'lead_speed': lead_speed_list,
        'lateral_pos': lateral_pos_list,
        'lane_width': lane_width_list,
        'time_headway': time_headway.tolist(),
        'curvature': curvature_list,
        'yaw': yaw_list,
        'yaw_rate': yaw_rate_list,
        'is_at_intersection': is_at_intersection_list,
        'has_stopline': has_stopline_list,
        'has_traffic_light': has_traffic_light_list,
        'is_turning': is_turning_list,
        'is_u_turn': is_u_turn_list,
        'frame_count': len(valid_frames),
    }


# ============================================================
# Scene 4: 跟停确认
# ============================================================

def confirm_following_stop(
    features: dict,
    config: dict = DETECTION_PARAMS["following_stop"],
) -> dict:
    """确认跟停场景

    移植自参考项目:
    - tiqvtest.py detect_following_stop()
    - global_detection_following_stop.py filter_following_stop_by_lead_distance()

    检查:
    1. 存在减速到停车的过程
    2. 前车存在比例 >= 60%
    3. 平均前车距离 < 50m
    """
    speed = np.array(features["speed"])
    lead_dist = np.array(features["lead_dist"])
    time_arr = np.array(features["time"])

    if len(speed) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    # 检查是否有减速到停车
    stop_speed = 0.1
    has_stop = np.any(speed < stop_speed)
    min_start_speed = 3.0
    has_decel = np.any(speed > min_start_speed)

    if not has_stop or not has_decel:
        return {"confirmed": False, "reason": "无有效减速停车过程"}

    # 前车存在比例
    valid_lead = lead_dist > 0
    lead_ratio = np.sum(valid_lead) / len(lead_dist) if len(lead_dist) > 0 else 0

    if lead_ratio < config["min_lead_ratio"]:
        return {
            "confirmed": False,
            "reason": f"前车比例不足: {lead_ratio:.2f} < {config['min_lead_ratio']}",
            "metrics": {"lead_ratio": float(lead_ratio)},
        }

    # 平均前车距离
    valid_dists = lead_dist[valid_lead]
    avg_lead_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else float('inf')

    if avg_lead_dist > config["max_lead_distance"]:
        return {
            "confirmed": False,
            "reason": f"平均前车距离过大: {avg_lead_dist:.1f}m > {config['max_lead_distance']}m",
            "metrics": {"avg_lead_dist": avg_lead_dist, "lead_ratio": float(lead_ratio)},
        }

    return {
        "confirmed": True,
        "reason": "跟停确认通过",
        "metrics": {
            "lead_ratio": float(lead_ratio),
            "avg_lead_dist": avg_lead_dist,
            "min_lead_dist": float(np.min(valid_dists)) if len(valid_dists) > 0 else 0,
            "max_lead_dist": float(np.max(valid_dists)) if len(valid_dists) > 0 else 0,
        },
    }


# ============================================================
# Scene 1: 路口停车确认
# ============================================================

def confirm_intersection_stop(
    features: dict,
    config: dict = DETECTION_PARAMS["intersection_stop"],
) -> dict:
    """确认路口停车场景

    移植自参考项目 global_detection_intersection_stop.py

    检查:
    1. 路口比例 >= 60%
    2. 存在停车阶段
    3. 停车时前车距离 > 50m 或无前车 (即自己是头车)
    """
    speed = np.array(features["speed"])
    lead_dist = np.array(features["lead_dist"])
    is_at_intersection = np.array(features["is_at_intersection"])

    if len(speed) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    # 路口比例
    intersection_ratio = np.sum(is_at_intersection) / len(is_at_intersection)
    if intersection_ratio < config["min_intersection_ratio"]:
        return {
            "confirmed": False,
            "reason": f"路口比例不足: {intersection_ratio:.2f} < {config['min_intersection_ratio']}",
            "metrics": {"intersection_ratio": float(intersection_ratio)},
        }

    # 检查停车
    stop_mask = speed < 0.1
    if not np.any(stop_mask):
        return {"confirmed": False, "reason": "无停车阶段"}

    # 停车时是否为头车 (前车距离 > 50m 或无前车)
    stop_lead = lead_dist[stop_mask]
    if len(stop_lead) > 0:
        first_vehicle_mask = (stop_lead == 0) | (stop_lead > 50)
        first_vehicle_ratio = np.sum(first_vehicle_mask) / len(stop_lead)
    else:
        first_vehicle_ratio = 1.0

    if first_vehicle_ratio < 0.5:
        return {
            "confirmed": False,
            "reason": f"停车时非头车: 头车比例 {first_vehicle_ratio:.2f}",
            "metrics": {
                "intersection_ratio": float(intersection_ratio),
                "first_vehicle_ratio": float(first_vehicle_ratio),
            },
        }

    return {
        "confirmed": True,
        "reason": "路口停车确认通过",
        "metrics": {
            "intersection_ratio": float(intersection_ratio),
            "first_vehicle_ratio": float(first_vehicle_ratio),
            "stopline_ratio": float(np.mean(features["has_stopline"])),
            "traffic_light_ratio": float(np.mean(features["has_traffic_light"])),
        },
    }


# ============================================================
# Scene 2: 起步确认
# ============================================================

def confirm_empty_start(
    features: dict,
    config: dict = DETECTION_PARAMS["empty_start"],
) -> dict:
    """确认起步场景 (空驶起步)

    移植自参考项目 global_detection_empty_start.py

    检查:
    1. 存在静止到加速的过程
    2. 平均前车距离 > 50m 或无前车
    """
    speed = np.array(features["speed"])
    acceleration = np.array(features["acceleration"])
    lead_dist = np.array(features["lead_dist"])

    if len(speed) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    # 检查静止阶段
    static_mask = speed < 0.1
    if not np.any(static_mask):
        return {"confirmed": False, "reason": "无静止阶段"}

    # 检查加速
    has_accel = np.any(acceleration > config["accelerate_thresh"])
    if not has_accel:
        return {"confirmed": False, "reason": "无有效加速"}

    # 检查前车距离
    valid_lead = lead_dist > 0
    if np.sum(valid_lead) == 0:
        # 完全无前车
        return {
            "confirmed": True,
            "reason": "起步确认通过 (无前车)",
            "metrics": {"has_lead": False, "avg_lead_dist": 0},
        }

    avg_lead = float(np.mean(lead_dist[valid_lead]))
    if avg_lead < config["min_lead_distance"]:
        return {
            "confirmed": False,
            "reason": f"前车距离不足: {avg_lead:.1f}m < {config['min_lead_distance']}m",
            "metrics": {"avg_lead_dist": avg_lead},
        }

    return {
        "confirmed": True,
        "reason": "起步确认通过",
        "metrics": {"has_lead": True, "avg_lead_dist": avg_lead},
    }


# ============================================================
# Scene 3: 跟车确认
# ============================================================

def confirm_following_vehicle(
    features: dict,
    config: dict = DETECTION_PARAMS["following_vehicle"],
) -> dict:
    """确认跟车场景

    移植自参考项目 tiqvtest.py detect_following_vehicle()

    检查:
    1. 速度 >= 5 m/s
    2. 时距在 [0, 15] 秒内
    3. 持续 >= 3 秒
    4. 时距波动 <= 1.0 秒
    """
    speed = np.array(features["speed"])
    time_headway = np.array(features["time_headway"])
    time_arr = np.array(features["time"])

    if len(speed) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    fps = len(time_arr) / (time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 10

    # 有效跟车帧
    valid_mask = (
        (time_headway > config["headway_min"]) &
        (time_headway < config["headway_max"]) &
        (speed > config["speed_thresh"])
    )

    # 找最长连续有效段
    max_consecutive = 0
    current_consecutive = 0
    best_start = 0
    best_end = 0
    current_start = 0

    for i in range(len(valid_mask)):
        if valid_mask[i]:
            if current_consecutive == 0:
                current_start = i
            current_consecutive += 1
        else:
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                best_start = current_start
                best_end = i - 1
            current_consecutive = 0

    if current_consecutive > max_consecutive:
        max_consecutive = current_consecutive
        best_start = current_start
        best_end = len(valid_mask) - 1

    duration = max_consecutive / fps
    if duration < config["min_follow_duration"]:
        return {
            "confirmed": False,
            "reason": f"跟车时长不足: {duration:.1f}s < {config['min_follow_duration']}s",
            "metrics": {"max_duration": duration},
        }

    # 时距波动
    thw_segment = time_headway[best_start:best_end + 1]
    fluct = float(np.max(thw_segment) - np.min(thw_segment))
    if fluct > config["headway_fluct_thresh"]:
        return {
            "confirmed": False,
            "reason": f"时距波动过大: {fluct:.2f}s > {config['headway_fluct_thresh']}s",
            "metrics": {"headway_fluctuation": fluct, "duration": duration},
        }

    return {
        "confirmed": True,
        "reason": "跟车确认通过",
        "metrics": {
            "duration": duration,
            "headway_mean": float(np.mean(thw_segment)),
            "headway_min": float(np.min(thw_segment)),
            "headway_max": float(np.max(thw_segment)),
            "headway_fluctuation": fluct,
        },
    }


# ============================================================
# Scene 5: 变道确认
# ============================================================

def confirm_lane_change(
    features: dict,
    config: dict = DETECTION_PARAMS["lane_change"],
) -> dict:
    """确认变道场景

    移植自参考项目 global_detection_lane_change.py filter_straight_road_lane_change()

    检查:
    1. 直行道路 (路口/转弯比例低)
    2. 横摆角速度/曲率在合理范围
    3. 横向位置变化明显
    """
    yaw_rate = np.array(features["yaw_rate"])
    curvature = np.array(features["curvature"])
    lateral_pos = np.array(features["lateral_pos"])
    is_at_intersection = np.array(features["is_at_intersection"])
    lead_dist = np.array(features["lead_dist"])

    if len(yaw_rate) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    # 路口比例
    intersection_ratio = float(np.mean(is_at_intersection))
    if intersection_ratio > config["max_intersection_ratio"]:
        return {
            "confirmed": False,
            "reason": f"路口比例过高: {intersection_ratio:.2f}",
            "metrics": {"intersection_ratio": intersection_ratio},
        }

    # 横摆角速度检查
    abs_yaw_rate = np.abs(yaw_rate)
    avg_yaw = float(np.mean(abs_yaw_rate))
    max_yaw = float(np.max(abs_yaw_rate))

    if avg_yaw > config.get("max_avg_yaw_rate", 0.08):
        return {
            "confirmed": False,
            "reason": f"平均横摆角速度过大: {avg_yaw:.3f}",
            "metrics": {"avg_yaw_rate": avg_yaw, "max_yaw_rate": max_yaw},
        }

    if max_yaw > config.get("max_yaw_rate", 0.2):
        return {
            "confirmed": False,
            "reason": f"最大横摆角速度过大: {max_yaw:.3f}",
            "metrics": {"avg_yaw_rate": avg_yaw, "max_yaw_rate": max_yaw},
        }

    # 曲率检查
    abs_curv = np.abs(curvature)
    avg_curv = float(np.mean(abs_curv))
    if avg_curv > config.get("max_avg_curvature", 0.08):
        return {
            "confirmed": False,
            "reason": f"平均曲率过大: {avg_curv:.3f}",
            "metrics": {"avg_curvature": avg_curv},
        }

    # 横向位置变化检查 (变道应有明显横向位移)
    lateral_range = float(np.max(lateral_pos) - np.min(lateral_pos))
    if lateral_range < 1.0:
        return {
            "confirmed": False,
            "reason": f"横向位移不足: {lateral_range:.2f}m",
            "metrics": {"lateral_range": lateral_range},
        }

    # 前车检查
    valid_lead = lead_dist[(lead_dist > 0) & (lead_dist < 50)]
    lead_ratio = len(valid_lead) / len(lead_dist) if len(lead_dist) > 0 else 0

    return {
        "confirmed": True,
        "reason": "变道确认通过",
        "metrics": {
            "intersection_ratio": intersection_ratio,
            "avg_yaw_rate": avg_yaw,
            "max_yaw_rate": max_yaw,
            "avg_curvature": avg_curv,
            "lateral_range": lateral_range,
            "front_vehicle_ratio": float(lead_ratio),
        },
    }
