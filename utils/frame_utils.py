"""
frame_utils.py
帧数据工具函数，移植自参考项目:
- feature_utils_test.py: get_lead_vehicless, 车道宽度计算
- tiqvandjiance.py: calculate_lateral_position
- global_detection_intersection_stop.py: check_has_stopline, check_has_traffic_light
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_PARAMS


# ============================================================
# 车道线处理
# ============================================================

def interpolate_laneline_y_at_x(
    pts: List[List[float]], target_x: float = 0.0
) -> Optional[float]:
    """在给定 x 位置插值车道线的 y 坐标

    Args:
        pts: [[x1,y1], [x2,y2], ...] 车道线采样点
        target_x: 目标 x 坐标

    Returns:
        插值得到的 y 坐标，或 None (无法插值)
    """
    if not pts or len(pts) < 2:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # 检查 target_x 是否在点的范围内（不 extrapolate）
    if target_x < min(xs) or target_x > max(xs):
        return None

    # 线性插值
    return float(np.interp(target_x, xs, ys))


def merge_close_lanelines(
    y_positions: List[float], threshold: float = 0.8
) -> List[float]:
    """合并距离较近的车道线 y 坐标

    Args:
        y_positions: 排序后的 y 坐标列表
        threshold: 合并阈值 (米)

    Returns:
        合并后的 y 坐标列表
    """
    if not y_positions:
        return []

    merged = [y_positions[0]]
    for y in y_positions[1:]:
        if abs(y - merged[-1]) < threshold:
            merged[-1] = (merged[-1] + y) / 2
        else:
            merged.append(y)
    return merged


def get_lane_boundaries(
    lanelines: List[dict],
    ego_x: float = 0.0,
    valid_types: Set[int] = None,
) -> Tuple[Optional[float], Optional[float], float]:
    """获取当前车道的左右边界

    Args:
        lanelines: 车道线列表
        ego_x: 自车 x 坐标 (通常为 0)
        valid_types: 有效车道线类型集合

    Returns:
        (left_boundary, right_boundary, lane_width)
        边界为 y 坐标值 (左正右负)
    """
    if valid_types is None:
        valid_types = FEATURE_PARAMS["valid_lane_types"]

    y_positions = []
    for lane in lanelines:
        lane_type = lane.get('laneline_type', lane.get('type', 0))
        if lane_type not in valid_types:
            continue
        pts = lane.get('pts_fixed_num', [])
        y_at_ego = interpolate_laneline_y_at_x(pts, ego_x)
        if y_at_ego is not None:
            y_positions.append(y_at_ego)

    if len(y_positions) < 2:
        return None, None, 0.0

    y_positions.sort()
    y_positions = merge_close_lanelines(y_positions)

    # 找到自车所在车道 (y=0 附近)
    left_bound = None
    right_bound = None

    for y in y_positions:
        if y > 0:
            if left_bound is None or y < left_bound:
                left_bound = y
        elif y < 0:
            if right_bound is None or y > right_bound:
                right_bound = y

    if left_bound is not None and right_bound is not None:
        lane_width = left_bound - right_bound
    else:
        lane_width = 3.5  # 默认车道宽度

    return left_bound, right_bound, lane_width


# ============================================================
# 前车检测
# ============================================================

def get_lead_vehicle(
    agents: List[dict],
    lanelines: List[dict],
    ego_yaw: float = 0.0,
) -> Tuple[Optional[dict], float, float]:
    """检测当前车道前车

    移植自参考项目 feature_utils_test.py get_lead_vehicless()

    Args:
        agents: 周围目标列表 (data_agent)
        lanelines: 车道线列表 (data_laneline)
        ego_yaw: 自车航向角

    Returns:
        (lead_agent, lead_dist, lead_speed)
        lead_agent 为 None 表示无前车
    """
    if not agents or not lanelines:
        return None, 0.0, 0.0

    # 获取当前车道边界
    left_bound, right_bound, lane_width = get_lane_boundaries(lanelines)

    if left_bound is None or right_bound is None:
        return None, 0.0, 0.0

    tolerance = FEATURE_PARAMS.get("lane_vehicle_tolerance", 0.5)

    best_lead = None
    best_dist = float('inf')
    best_speed = 0.0

    for agent in agents:
        # 只考虑车辆 (cls=2)
        if agent.get('cls', -1) != 2:
            continue
        if not agent.get('agent_mask', False):
            continue

        agent_dist = agent.get('dist', 0.0)
        agent_x = agent.get('x', 0.0)
        agent_y = agent.get('y', 0.0)

        # 必须在自车前方
        if agent_x <= 0:
            continue

        # 获取该 x 位置的车道边界 (允许外推 20m)
        agent_left = None
        agent_right = None
        for lane in lanelines:
            lane_type = lane.get('laneline_type', lane.get('type', 0))
            if lane_type not in FEATURE_PARAMS["valid_lane_types"]:
                continue
            pts = lane.get('pts_fixed_num', [])
            y_at_agent = interpolate_laneline_y_at_x(pts, agent_x)
            if y_at_agent is not None:
                if y_at_agent > 0:
                    if agent_left is None or y_at_agent < agent_left:
                        agent_left = y_at_agent
                elif y_at_agent < 0:
                    if agent_right is None or y_at_agent > agent_right:
                        agent_right = y_at_agent

        if agent_left is None or agent_right is None:
            # 使用自车处边界 + tolerance
            if (right_bound - tolerance) <= agent_y <= (left_bound + tolerance):
                if agent_dist < best_dist:
                    best_lead = agent
                    best_dist = agent_dist
                    best_speed = np.sqrt(
                        agent.get('vx', 0.0) ** 2 + agent.get('vy', 0.0) ** 2
                    )
            continue

        # 检查是否在当前车道
        if (agent_right - tolerance) <= agent_y <= (agent_left + tolerance):
            if agent_dist < best_dist:
                best_lead = agent
                best_dist = agent_dist
                best_speed = np.sqrt(
                    agent.get('vx', 0.0) ** 2 + agent.get('vy', 0.0) ** 2
                )

    return best_lead, best_dist, best_speed


# ============================================================
# 横向位置
# ============================================================

def calculate_lateral_position(
    lanelines: List[dict],
    ego_x: float = 0.0,
) -> Tuple[float, float]:
    """计算自车在当前车道中的横向位置

    移植自参考项目 tiqvandjiance.py

    Args:
        lanelines: 车道线列表
        ego_x: 自车 x 坐标

    Returns:
        (lateral_pos, curvature)
        lateral_pos: 距车道中心的偏移 (左正右负)
        curvature: 曲率 (当前简化为 0)
    """
    left_bound, right_bound, lane_width = get_lane_boundaries(lanelines, ego_x)

    if left_bound is None or right_bound is None:
        return 0.0, 0.0

    center = (left_bound + right_bound) / 2.0
    lateral_pos = -center  # 自车在 y=0, 偏移 = -(中心)

    return lateral_pos, 0.0


# ============================================================
# 停止线与信号灯检测
# ============================================================

def check_has_stopline(frame: dict, distance_threshold: float = 50.0) -> bool:
    """检查前方是否有停止线

    移植自参考项目 global_detection_intersection_stop.py

    Args:
        frame: 归一化帧数据
        distance_threshold: 检测距离 (米)

    Returns:
        True: 前方有停止线
    """
    roaditems = frame.get('data_roaditems', {})
    stoplines = roaditems.get('stoplines', [])

    if not stoplines:
        return False

    for stopline in stoplines:
        points = stopline.get('points', [])
        if len(points) < 2:
            continue

        sl_x = (points[0].get('x', 0) + points[1].get('x', 0)) / 2
        sl_y = (points[0].get('y', 0) + points[1].get('y', 0)) / 2

        if 0 < sl_x < distance_threshold and -15 < sl_y < 15:
            return True

    return False


def check_has_traffic_light(frame: dict, distance_threshold: float = 50.0) -> bool:
    """检查前方是否有有效信号灯

    移植自参考项目 tiqvtest.py

    Args:
        frame: 归一化帧数据
        distance_threshold: 检测距离 (米)

    Returns:
        True: 前方有有效信号灯 (status >= 1 即红/黄/绿均算，且距离 < threshold)
    """
    ego_status = frame.get('data_ego_curr_status', {})
    lights = ego_status.get('egolane_traffic_lights', [])

    if not lights:
        return False

    # lights 结构: [{'node_id': int, 'status': int, 'countdown': float,
    #                 'pos_x': float, 'pos_y': float, 'type': int}, ...]
    first_light = lights[0]
    if not isinstance(first_light, dict):
        return False

    # status: 0=无效, 1=红, 2=黄, 3=绿; 有效灯色(>=1)均视为路口标志
    status = first_light.get('status', 0)
    if status < 1:
        return False

    light_x = first_light.get('pos_x', 0.0)
    light_y = first_light.get('pos_y', 0.0)

    if light_x <= 0:
        return False

    distance = np.sqrt(light_x ** 2 + light_y ** 2)
    return distance < distance_threshold


# ============================================================
# 特征提取辅助
# ============================================================

def smooth_lead_distance(lead_dists: np.ndarray, window: int = 5) -> np.ndarray:
    """平滑前车距离

    Args:
        lead_dists: 原始前车距离数组
        window: 平滑窗口大小

    Returns:
        平滑后的数组
    """
    if len(lead_dists) < window:
        return lead_dists.copy()

    result = lead_dists.copy()
    valid_mask = lead_dists > 0

    if np.sum(valid_mask) < window:
        return result

    kernel = np.ones(window) / window
    smoothed_valid = np.convolve(valid_mask.astype(float), kernel, mode='same')
    smoothed_vals = np.convolve(
        np.where(valid_mask, lead_dists, 0), kernel, mode='same'
    )

    smooth_mask = smoothed_valid > 0.5
    result[smooth_mask] = smoothed_vals[smooth_mask] / np.maximum(
        smoothed_valid[smooth_mask], 1e-6
    )

    return result
