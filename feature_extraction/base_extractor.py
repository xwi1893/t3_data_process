"""
base_extractor.py
直接特征 + 间接特征提取流水线
移植自参考项目 main_extra.py
"""

import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_PARAMS, WINDOW_EXTENSION
from utils.frame_utils import (
    get_lead_vehicle, calculate_lateral_position,
    smooth_lead_distance, get_lane_boundaries,
)


def extract_direct_features(
    frame_data: Dict[str, dict],
    pb_files: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """从片段帧数据提取直接特征

    移植自参考项目 main_extra.py extract_direct_features()
    支持时域窗口扩展: 第一帧历史数据 + 最后一帧未来数据

    Args:
        frame_data: {pb_filename: normalized_frame}
        pb_files: 有序的 pb 文件名列表

    Returns:
        直接特征字典，或 None
    """
    valid_files = [f for f in pb_files if f in frame_data]
    if len(valid_files) < 3:
        return None

    speed_list = []
    yaw_rate_list = []
    lead_dist_list = []
    lead_speed_list = []
    lateral_pos_list = []
    lanelines_list = []

    for fname in valid_files:
        frame = frame_data[fname]
        ego = frame.get('data_ego_curr_status', {})
        agents = frame.get('data_agent', [])
        lanelines = frame.get('data_laneline', [])

        speed_list.append(ego.get('v', 0.0))
        yaw_rate_list.append(ego.get('yaw_rate', 0.0))

        # 前车
        lead_agent, lead_dist, lead_spd = get_lead_vehicle(
            agents, lanelines, ego.get('yaw', 0.0)
        )
        lead_dist_list.append(lead_dist if lead_agent is not None else 0.0)
        lead_speed_list.append(lead_spd)

        # 横向位置
        lat_pos, _ = calculate_lateral_position(lanelines)
        lateral_pos_list.append(lat_pos)
        lanelines_list.append(lanelines)

    # === 时域窗口扩展 ===
    ext_cfg = WINDOW_EXTENSION

    # 历史数据前补 (从第一帧)
    if ext_cfg.get("use_history", False) and valid_files:
        from detection.scene_detectors import _extract_history_features
        first_frame = frame_data[valid_files[0]]
        hist = _extract_history_features(first_frame, fps=10)
        if hist and hist.get('speed'):
            max_hist = ext_cfg.get("max_history_frames", 30)
            hist_speed = hist['speed'][-max_hist:]
            n = len(hist_speed)
            speed_list = hist_speed + speed_list
            yaw_rate_list = hist.get('yaw_rate', [0.0]*n)[-max_hist:] + yaw_rate_list
            lead_dist_list = hist.get('lead_dist', [0.0]*n)[-max_hist:] + lead_dist_list
            lead_speed_list = hist.get('lead_speed', [0.0]*n)[-max_hist:] + lead_speed_list
            # 历史帧无车道线数据，用第一帧的横向位置和车道线填充
            first_lat = lateral_pos_list[0] if lateral_pos_list else 0.0
            first_ll = lanelines_list[0] if lanelines_list else []
            lateral_pos_list = [first_lat] * n + lateral_pos_list
            lanelines_list = [first_ll] * n + lanelines_list

    # 未来数据后补 (从最后一帧)
    if ext_cfg.get("use_future", False) and valid_files:
        from detection.scene_detectors import _extract_future_features
        last_frame = frame_data[valid_files[-1]]
        fut = _extract_future_features(last_frame, fps=10)
        if fut and fut.get('speed'):
            max_fut = ext_cfg.get("max_future_frames", 30)
            fut_speed = fut['speed'][:max_fut]
            n = len(fut_speed)
            speed_list = speed_list + fut_speed
            yaw_rate_list = yaw_rate_list + fut.get('yaw_rate', [0.0]*n)[:max_fut]
            lead_dist_list = lead_dist_list + fut.get('lead_dist', [0.0]*n)[:max_fut]
            lead_speed_list = lead_speed_list + fut.get('lead_speed', [0.0]*n)[:max_fut]
            last_lat = lateral_pos_list[-1] if lateral_pos_list else 0.0
            last_ll = lanelines_list[-1] if lanelines_list else []
            lateral_pos_list = lateral_pos_list + [last_lat] * n
            lanelines_list = lanelines_list + [last_ll] * n

    # 转 numpy + 平滑
    speed = np.array(speed_list)
    yaw_rate = np.array(yaw_rate_list)
    lead_dist_raw = np.array(lead_dist_list)
    lead_speed = np.array(lead_speed_list)

    # 前车距离平滑
    window = FEATURE_PARAMS.get("smooth_lead_dist_window", 5)
    lead_dist = smooth_lead_distance(lead_dist_raw, window)

    # 横向位置平滑
    lateral_pos = np.array(lateral_pos_list)
    lat_window = FEATURE_PARAMS.get("smooth_lateral_pos_window", 3)
    if len(lateral_pos) >= lat_window:
        kernel = np.ones(lat_window) / lat_window
        smoothed = np.convolve(lateral_pos, kernel, mode='same')
        # 车道宽度约束
        max_offsets = []
        for ll in lanelines_list:
            _, _, lw = get_lane_boundaries(ll)
            max_offsets.append((lw / 2) + 0.5 if lw > 0 else 2.5)
        max_offsets = np.array(max_offsets)
        lateral_pos = np.clip(smoothed, -max_offsets, max_offsets)

    # 时间
    time_sec = np.arange(len(valid_files)) / 10.0  # 假设 10fps

    return {
        "valid_files": valid_files,
        "time_sec": time_sec,
        "speed": speed,
        "yaw_rate": yaw_rate,
        "lead_dist": lead_dist,
        "lead_speed": lead_speed,
        "lateral_pos": lateral_pos,
    }


def calculate_indirect_features(
    direct: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """计算间接特征

    移植自参考项目 main_extra.py calculate_indirect_features()

    Args:
        direct: extract_direct_features() 的输出

    Returns:
        间接特征字典
    """
    speed = direct["speed"]
    time_sec = direct["time_sec"]

    # 帧率
    if len(time_sec) > 1 and time_sec[-1] > time_sec[0]:
        fps = len(time_sec) / (time_sec[-1] - time_sec[0])
    else:
        fps = 10.0

    # 加速度 (平滑, 窗口=3)
    raw_accel = np.diff(speed) * fps
    window_acc = 3
    if len(raw_accel) >= window_acc:
        kernel = np.ones(window_acc) / window_acc
        smoothed = np.convolve(raw_accel, kernel, mode='same')
    else:
        smoothed = raw_accel
    acceleration = np.concatenate([[0.0], smoothed])

    # 横向加速度
    lateral_acc = speed * direct["yaw_rate"]

    # 时距 THW
    speed_safe = np.where(speed < 0.1, 0.1, speed)
    max_thw = FEATURE_PARAMS.get("max_time_headway", 15.0)
    time_headway = np.minimum(direct["lead_dist"] / speed_safe, max_thw)

    return {
        "acceleration": acceleration,
        "lateral_acc": lateral_acc,
        "time_headway": time_headway,
        "fps": fps,
    }


def extract_all_features(
    frame_data: Dict[str, dict],
    pb_files: List[str],
) -> Optional[dict]:
    """完整特征提取流水线

    Args:
        frame_data: {pb_filename: normalized_frame}
        pb_files: 有序的 pb 文件名列表

    Returns:
        包含 direct + indirect 特征的字典，或 None
    """
    direct = extract_direct_features(frame_data, pb_files)
    if direct is None:
        return None

    indirect = calculate_indirect_features(direct)

    return {
        **direct,
        **indirect,
    }
