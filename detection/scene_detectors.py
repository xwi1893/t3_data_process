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
from config import DETECTION_PARAMS, FEATURE_PARAMS, WINDOW_EXTENSION
from utils.frame_utils import (
    get_lead_vehicle, calculate_lateral_position,
    check_has_stopline, check_has_traffic_light,
    get_lane_boundaries, smooth_lead_distance,
)


# ============================================================
# 时域窗口扩展辅助函数
# ============================================================

def _get_history_win(frame: dict) -> int:
    """从帧数据中获取历史窗口帧数"""
    config = frame.get('_raw', {}).get('config', {})
    return config.get('history_win', 30)


def _get_future_win(frame: dict) -> int:
    """从帧数据中获取未来窗口帧数"""
    config = frame.get('_raw', {}).get('config', {})
    return config.get('future_win', 30)


def _find_lead_agent(agents: list, lead_agent: Optional[dict]) -> Optional[dict]:
    """通过 id 匹配找到前车 agent 的完整数据（含轨迹）

    第 1 层 mask 过滤: 匹配后检查 agent_mask，整体无效则返回 None
    """
    if lead_agent is None:
        return None
    lead_id = lead_agent.get('id')
    if not lead_id:
        return None
    for a in agents:
        if a.get('id') == lead_id:
            if not a.get('agent_mask', False):
                return None  # agent 整体无效
            return a
    return None


def _extract_lead_from_trajectory(
    ego_traj_pos: list,
    agent: Optional[dict],
    indices: list,
    valid_masks: list,
) -> Tuple[list, list]:
    """从轨迹数据中提取前车距离和速度

    Args:
        ego_traj_pos: 自车轨迹位置 [[x,y], ...] (与 agent 同一坐标系)
        agent: 前车 agent (含 pos/velocity/valid_mask)
        indices: 要提取的时间步索引列表
        valid_masks: 每个时间步的有效性掩码

    Returns:
        (lead_dists, lead_speeds) 列表
    """
    lead_dists = []
    lead_speeds = []

    agent_pos = agent.get('pos', []) if agent else []
    agent_vel = agent.get('velocity', []) if agent else []
    agent_masks = agent.get('valid_mask', []) if agent else []

    for i, idx in enumerate(indices):
        # 自车 mask 检查
        if i < len(valid_masks) and not valid_masks[i]:
            continue  # 自车无效，跳过整个时间步

        # agent mask 检查
        if idx < len(agent_masks) and not agent_masks[idx]:
            lead_dists.append(0.0)
            lead_speeds.append(0.0)
            continue

        if idx < len(agent_pos) and idx < len(ego_traj_pos):
            ax, ay = agent_pos[idx]
            ex, ey = ego_traj_pos[idx]
            dist = np.sqrt((ax - ex) ** 2 + (ay - ey) ** 2)
            # 只取前方车辆 (ax > 0 即在自车前方)
            if ax - ex > 0 and dist < 200:
                lead_dists.append(dist)
                if idx < len(agent_vel):
                    vx, vy = agent_vel[idx]
                    lead_speeds.append(np.sqrt(vx ** 2 + vy ** 2))
                else:
                    lead_speeds.append(0.0)
            else:
                lead_dists.append(0.0)
                lead_speeds.append(0.0)
        else:
            lead_dists.append(0.0)
            lead_speeds.append(0.0)

    return lead_dists, lead_speeds


def _extract_history_features(frame: dict, fps: int = 10) -> dict:
    """从第一帧的历史数据中提取特征

    Args:
        frame: 归一化帧数据（第一帧）
        fps: 帧率

    Returns:
        特征 dict (与 extract_segment_features 格式一致)
    """
    hist_traj = frame.get('label_ego_hist_traj', {})
    masks = hist_traj.get('mask', [])
    positions = hist_traj.get('pos', [])
    velocities = hist_traj.get('v', [])
    yaws = hist_traj.get('yaw', [])
    yaw_rates = hist_traj.get('yaw_rate', [])

    if not positions or not masks:
        return {}

    history_win = _get_history_win(frame)
    n_steps = min(len(positions), len(masks), history_win)
    if n_steps == 0:
        return {}

    # 检测当前帧前车
    agents = frame.get('data_agent', [])
    lanelines = frame.get('data_laneline', [])
    ego_yaw = frame.get('data_ego_curr_status', {}).get('yaw', 0.0)
    lead_agent, _, _ = get_lead_vehicle(agents, lanelines, ego_yaw)
    lead_agent_full = _find_lead_agent(agents, lead_agent)

    # 提取前车历史轨迹
    # agent pos 中历史部分: pos[:history_win]
    hist_indices = list(range(history_win))
    ego_hist_pos = [[0.0, 0.0]] * history_win  # 自车历史在自身坐标系下为原点
    if len(positions) >= history_win:
        ego_hist_pos = positions[:history_win]

    lead_dists, lead_speeds = _extract_lead_from_trajectory(
        ego_hist_pos, lead_agent_full, hist_indices,
        masks[:history_win]
    )

    # 构建特征序列 (只保留 mask=True 的有效步)
    time_list = []
    speed_list = []
    yaw_list = []
    yaw_rate_list = []
    lead_dist_list = []
    lead_speed_list = []
    lateral_pos_list = []

    valid_count = 0
    for i in range(n_steps):
        if i < len(masks) and not masks[i]:
            continue
        if i < len(lead_dists):
            valid_count += 1
            # 时间: 负数，表示当前帧之前
            t = -(n_steps - i) / fps
            time_list.append(t)

            # 自车速度
            if i < len(velocities):
                vx, vy = velocities[i]
                speed_list.append(np.sqrt(vx ** 2 + vy ** 2))
            else:
                speed_list.append(0.0)

            yaw_list.append(yaws[i] if i < len(yaws) else 0.0)
            yaw_rate_list.append(yaw_rates[i] if i < len(yaw_rates) else 0.0)
            lead_dist_list.append(lead_dists[i])
            lead_speed_list.append(lead_speeds[i])
            # 横向位置: ego 轨迹 y 分量
            lateral_pos_list.append(positions[i][1] if i < len(positions) and len(positions[i]) > 1 else 0.0)

    return {
        'time': time_list,
        'speed': speed_list,
        'yaw': yaw_list,
        'yaw_rate': yaw_rate_list,
        'lead_dist': lead_dist_list,
        'lead_speed': lead_speed_list,
        'lateral_pos': lateral_pos_list,
    }


def _extract_future_features(frame: dict, fps: int = 10) -> dict:
    """从最后一帧的未来数据中提取特征

    Args:
        frame: 归一化帧数据（最后一帧）
        fps: 帧率

    Returns:
        特征 dict (与 extract_segment_features 格式一致)
    """
    future_traj = frame.get('label_ego_traj', {})
    masks = future_traj.get('mask', [])
    positions = future_traj.get('pos', [])
    velocities = future_traj.get('v', [])
    yaws = future_traj.get('yaw', [])
    yaw_rates = future_traj.get('yaw_rate', [])

    if not positions or not masks:
        return {}

    future_win = _get_future_win(frame)
    n_steps = min(len(positions), len(masks), future_win)
    if n_steps == 0:
        return {}

    # 检测当前帧前车
    agents = frame.get('data_agent', [])
    lanelines = frame.get('data_laneline', [])
    ego_yaw = frame.get('data_ego_curr_status', {}).get('yaw', 0.0)
    lead_agent, _, _ = get_lead_vehicle(agents, lanelines, ego_yaw)
    lead_agent_full = _find_lead_agent(agents, lead_agent)

    # 提取前车未来轨迹
    # agent pos 中未来部分: pos[history_win+1:]
    history_win = _get_history_win(frame)
    future_start = history_win + 1
    future_indices = list(range(future_start, future_start + future_win))
    ego_future_pos = positions[:n_steps]

    lead_dists, lead_speeds = _extract_lead_from_trajectory(
        ego_future_pos, lead_agent_full, future_indices,
        masks[:n_steps]
    )

    # 构建特征序列
    time_list = []
    speed_list = []
    yaw_list = []
    yaw_rate_list = []
    lead_dist_list = []
    lead_speed_list = []
    lateral_pos_list = []

    for i in range(n_steps):
        if i < len(masks) and not masks[i]:
            continue
        if i < len(lead_dists):
            # 时间: 正数，表示当前帧之后
            t = (i + 1) / fps
            time_list.append(t)

            if i < len(velocities):
                vx, vy = velocities[i]
                speed_list.append(np.sqrt(vx ** 2 + vy ** 2))
            else:
                speed_list.append(0.0)

            yaw_list.append(yaws[i] if i < len(yaws) else 0.0)
            yaw_rate_list.append(yaw_rates[i] if i < len(yaw_rates) else 0.0)
            lead_dist_list.append(lead_dists[i])
            lead_speed_list.append(lead_speeds[i])
            # 横向位置: ego 轨迹 y 分量
            lateral_pos_list.append(positions[i][1] if i < len(positions) and len(positions[i]) > 1 else 0.0)

    return {
        'time': time_list,
        'speed': speed_list,
        'yaw': yaw_list,
        'yaw_rate': yaw_rate_list,
        'lead_dist': lead_dist_list,
        'lead_speed': lead_speed_list,
        'lateral_pos': lateral_pos_list,
    }


def _concat_features(
    history_feats: dict,
    current_feats: dict,
    future_feats: dict,
    fps: int = 10,
) -> dict:
    """拼接历史 + 当前 + 未来特征序列

    Args:
        history_feats: 历史帧特征 (可能为空 dict)
        current_feats: 当前片段帧特征
        future_feats: 未来帧特征 (可能为空 dict)
        fps: 帧率

    Returns:
        拼接后的完整特征 dict
    """
    # 需要拼接的 list 类型字段 (动态特征)
    list_fields = [
        'speed', 'yaw', 'yaw_rate', 'lead_dist', 'lead_speed',
        'lateral_pos', 'lane_width', 'curvature',
    ]

    # 动态特征默认值 (用于历史/未来中缺失的字段)
    defaults = {
        'lateral_pos': 0.0, 'lane_width': 3.5, 'curvature': 0.0,
    }

    result = {}

    for field in list_fields:
        hist_vals = history_feats.get(field, [])
        curr_vals = current_feats.get(field, [])
        fut_vals = future_feats.get(field, [])

        if not hist_vals and history_feats:
            hist_vals = [defaults.get(field, 0.0)] * len(history_feats.get('speed', []))
        if not fut_vals and future_feats:
            fut_vals = [defaults.get(field, 0.0)] * len(future_feats.get('speed', []))

        result[field] = hist_vals + curr_vals + fut_vals

    # 重新计算 time 数组使其连续
    total_len = len(result['speed'])
    n_hist = len(history_feats.get('speed', []))
    n_curr = len(current_feats.get('speed', []))

    time_list = []
    for i in range(total_len):
        time_list.append((i - n_hist) / fps)
    result['time'] = time_list

    # 重新计算 acceleration (拼接后整体差分)
    speed_arr = np.array(result['speed'])
    accel = np.zeros(len(speed_arr))
    if len(speed_arr) > 1:
        accel[1:] = np.diff(speed_arr) * fps
    if len(accel) >= 3:
        kernel = np.ones(3) / 3
        accel = np.convolve(accel, kernel, mode='same')
    result['acceleration'] = accel.tolist()

    # 重新平滑 lead_dist (拼接后整体平滑)
    lead_dist_arr = np.array(result['lead_dist'])
    lead_dist_smooth = smooth_lead_distance(
        lead_dist_arr, FEATURE_PARAMS.get("smooth_lead_dist_window", 5)
    )
    result['lead_dist'] = lead_dist_smooth.tolist()

    # 重新计算时距
    time_headway = np.full(len(speed_arr), 15.0)
    valid_speed = speed_arr > 0.1
    time_headway[valid_speed] = np.minimum(
        lead_dist_smooth[valid_speed] / speed_arr[valid_speed], 15.0
    )
    result['time_headway'] = time_headway.tolist()

    result['frame_count'] = current_feats.get('frame_count', 0) + n_hist + len(future_feats.get('speed', []))

    return result


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

    if len(valid_frames) < 1:
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

    # === 时域窗口扩展 ===
    # 当前片段特征已提取完毕，存入 current_feats
    current_feats = {
        'speed': speed_list,
        'lead_dist': lead_dist_smooth.tolist(),
        'lead_speed': lead_speed_list,
        'lateral_pos': lateral_pos_list,
        'lane_width': lane_width_list,
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

    ext_cfg = WINDOW_EXTENSION
    history_feats = {}
    future_feats = {}

    # 历史数据前补 (从第一帧)
    if ext_cfg.get("use_history", False) and valid_frames:
        first_frame = valid_frames[0][1]
        history_feats = _extract_history_features(first_frame, fps)
        max_hist = ext_cfg.get("max_history_frames", 30)
        if history_feats and len(history_feats.get('speed', [])) > max_hist:
            for k, v in history_feats.items():
                if isinstance(v, list):
                    history_feats[k] = v[-max_hist:]

    # 未来数据后补 (从最后一帧)
    if ext_cfg.get("use_future", False) and valid_frames:
        last_frame = valid_frames[-1][1]
        future_feats = _extract_future_features(last_frame, fps)
        max_fut = ext_cfg.get("max_future_frames", 30)
        if future_feats and len(future_feats.get('speed', [])) > max_fut:
            for k, v in future_feats.items():
                if isinstance(v, list):
                    future_feats[k] = v[:max_fut]

    # 拼接: history + current + future
    result = _concat_features(history_feats, current_feats, future_feats, fps)

    # list → numpy
    for key in list(result.keys()):
        if isinstance(result[key], list):
            result[key] = np.array(result[key])

    # 补齐字段 (供 scene_extractors / general_features 使用)
    result['time_sec'] = result['time']
    result['lateral_acc'] = result['speed'] * result['yaw_rate']
    result['fps'] = 10.0

    # 静态特征: 直接从当前帧取标量值
    if current_feats.get('is_at_intersection'):
        result['is_at_intersection'] = current_feats['is_at_intersection'][0]
    else:
        result['is_at_intersection'] = False
    if current_feats.get('has_stopline'):
        result['has_stopline'] = current_feats['has_stopline'][0]
    else:
        result['has_stopline'] = False
    if current_feats.get('has_traffic_light'):
        result['has_traffic_light'] = current_feats['has_traffic_light'][0]
    else:
        result['has_traffic_light'] = False
    if current_feats.get('is_turning'):
        result['is_turning'] = current_feats['is_turning'][0]
    else:
        result['is_turning'] = False
    if current_feats.get('is_u_turn'):
        result['is_u_turn'] = current_feats['is_u_turn'][0]
    else:
        result['is_u_turn'] = False

    return result


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
    is_at_intersection = features["is_at_intersection"]

    if len(speed) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    # 路口判断 (静态特征, 直接用当前帧布尔值)
    if not is_at_intersection:
        return {
            "confirmed": False,
            "reason": "非路口区域",
            "metrics": {"is_at_intersection": False},
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
                "is_at_intersection": False,
                "first_vehicle_ratio": float(first_vehicle_ratio),
            },
        }

    return {
        "confirmed": True,
        "reason": "路口停车确认通过",
        "metrics": {
            "is_at_intersection": True,
            "first_vehicle_ratio": float(first_vehicle_ratio),
            "has_stopline": bool(features["has_stopline"]),
            "has_traffic_light": bool(features["has_traffic_light"]),
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
    is_at_intersection = features["is_at_intersection"]
    lead_dist = np.array(features["lead_dist"])

    if len(yaw_rate) < 3:
        return {"confirmed": False, "reason": "帧数不足"}

    # # 路口判断 (静态特征, 直接用当前帧布尔值)
    # if is_at_intersection:
    #     return {
    #         "confirmed": False,
    #         "reason": "路口区域内，不适用变道检测",
    #         "metrics": {"is_at_intersection": True},
    #     }

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

    # 横向位置变化检查 (变道应有明显横向位移) TODO: 出错，坐标系会随自车变化
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
            "is_at_intersection": False,
            "avg_yaw_rate": avg_yaw,
            "max_yaw_rate": max_yaw,
            "avg_curvature": avg_curv,
            "lateral_range": lateral_range,
            "front_vehicle_ratio": float(lead_ratio),
        },
    }
