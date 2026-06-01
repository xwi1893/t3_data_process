"""
stopline_checker.py
路口停车压线合规检查

检查车辆在停车过程中是否越过停止线
"""

import numpy as np
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMPLIANCE_PARAMS
from utils.frame_utils import check_has_stopline


def check_stopline_crossing(
    frame_data: Dict[str, dict],
    pb_files: List[str],
    stop_speed_thresh: float = COMPLIANCE_PARAMS["stop_speed_thresh"],
    distance_threshold: float = COMPLIANCE_PARAMS["stopline_distance_threshold"],
) -> dict:
    """检查路口停车时是否越过停止线

    逻辑:
    1. 找到停车帧 (speed < stop_speed_thresh)
    2. 检查停车前最后几帧是否有停止线
    3. 如果停止线从有变无 (越过)，标记为不合规

    Args:
        frame_data: {pb_filename: normalized_frame}
        pb_files: 有序的 pb 文件名列表
        stop_speed_thresh: 停车速度阈值
        distance_threshold: 停止线检测距离

    Returns:
        {passed: bool, reason: str, metrics: dict}
    """
    stopline_presence = []
    speeds = []
    valid_files = [f for f in pb_files if f in frame_data]

    for fname in valid_files:
        frame = frame_data[fname]
        ego = frame.get('data_ego_curr_status', {})
        speed = ego.get('v', 0.0)
        has_sl = check_has_stopline(frame, distance_threshold)

        speeds.append(speed)
        stopline_presence.append(has_sl)

    if not speeds:
        return {"passed": False, "reason": "无有效帧数据", "metrics": {}}

    speeds_arr = np.array(speeds)
    sl_arr = np.array(stopline_presence)

    # 找停车帧
    stop_indices = np.where(speeds_arr < stop_speed_thresh)[0]
    if len(stop_indices) == 0:
        return {
            "passed": True,
            "reason": "未检测到停车",
            "metrics": {"has_stop": False},
        }

    first_stop = stop_indices[0]

    # 检查停车前是否有停止线
    pre_stop_has_stopline = False
    lookback = min(first_stop, 20)  # 回看 20 帧
    for i in range(max(0, first_stop - lookback), first_stop):
        if sl_arr[i]:
            pre_stop_has_stopline = True
            break

    # 检查停车时及停车后停止线是否消失 (被越过)
    at_stop_has_stopline = False
    for i in stop_indices[:min(10, len(stop_indices))]:
        if sl_arr[i]:
            at_stop_has_stopline = True
            break

    stopline_ratio = float(np.mean(sl_arr))

    # 判定: 如果停车前能看到停止线，但停车时看不到了，可能越线
    crossed = pre_stop_has_stopline and not at_stop_has_stopline

    if crossed:
        return {
            "passed": False,
            "reason": "疑似越过停止线",
            "metrics": {
                "pre_stop_stopline": True,
                "at_stop_stopline": False,
                "stopline_ratio": stopline_ratio,
            },
        }

    return {
        "passed": True,
        "reason": "停止线检查通过",
        "metrics": {
            "pre_stop_stopline": pre_stop_has_stopline,
            "at_stop_stopline": at_stop_has_stopline,
            "stopline_ratio": stopline_ratio,
        },
    }
