"""
pb_loader.py
Protobuf 帧数据加载器，移植自参考项目 pd_loader.py

对外接口:
- scan_pb_frames():     扫描目录，建立帧号→时间戳/路径映射
- load_pb_frame():      读取单个 .pb 文件
- normalize_frame():    pb帧归一化为标准格式
"""

import os
import sys
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 确保 proto 目录在路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proto import dlp_raw_data_pb2
from google.protobuf.json_format import MessageToDict


def scan_pb_frames(one_frame_dir: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    """扫描目录中的 .pb 文件，按时间戳排序建立映射

    Args:
        one_frame_dir: 包含 .pb 文件的目录路径

    Returns:
        frame2ts:   {帧序号: 纳秒时间戳}
        frame2path: {帧序号: .pb 文件绝对路径}
    """
    if not os.path.exists(one_frame_dir):
        raise FileNotFoundError(f"目录不存在: {one_frame_dir}")

    pb_files = []
    for fname in os.listdir(one_frame_dir):
        if fname.endswith(".pb") and not fname.endswith(".pb.gz"):
            pb_files.append(fname)

    if not pb_files:
        raise ValueError(f"目录中未找到 .pb 文件: {one_frame_dir}")

    pb_files_sorted = sorted(pb_files, key=lambda x: int(x.replace(".pb", "")))

    frame2ts: Dict[int, int] = {}
    frame2path: Dict[int, str] = {}

    for frame_idx, fname in enumerate(pb_files_sorted):
        ts_ns = int(fname.replace(".pb", ""))
        abs_path = os.path.join(one_frame_dir, fname)
        frame2ts[frame_idx] = ts_ns
        frame2path[frame_idx] = abs_path

    print(f"[pb_loader] 扫描完成: 共 {len(pb_files_sorted)} 帧")
    if len(frame2ts) >= 2:
        ts_list = [frame2ts[i] for i in sorted(frame2ts.keys())]
        intervals = [(ts_list[i + 1] - ts_list[i]) / 1e9
                      for i in range(len(ts_list) - 1)]
        avg_interval = sum(intervals) / len(intervals)
        print(f"[pb_loader] 平均间隔: {avg_interval * 1000:.2f}ms "
              f"-> {1 / avg_interval:.1f}fps")

    return frame2ts, frame2path


def load_pb_frame(pb_path: str) -> dict:
    """读取单个 .pb 文件并反序列化

    Args:
        pb_path: .pb 文件路径

    Returns:
        帧数据字典 (snake_case 字段名)
    """
    if not os.path.exists(pb_path):
        raise FileNotFoundError(f"文件不存在: {pb_path}")

    msg = dlp_raw_data_pb2.DLPRawData()
    with open(pb_path, "rb") as f:
        msg.ParseFromString(f.read())

    # 兼容 protobuf 3.x 和 4.x
    try:
        frame_dict = MessageToDict(
            msg,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:
        # protobuf 3.x 使用不同的参数名
        frame_dict = MessageToDict(
            msg,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )
    return frame_dict


def normalize_frame(pb_frame: dict) -> dict:
    """将 pb 解析出的帧字典转换为标准格式

    移植自参考项目 pd_loader.py normalize_frame()
    输出字段与旧 JSON 格式兼容:
    - data_ego_curr_status: 自车状态
    - data_agent:           周围目标列表
    - data_laneline:        车道线
    - data_roaditems:       道路元素(停止线等)
    - data_ego_local_pose:  自车位姿
    - label_ego_hist_traj:  历史轨迹
    - label_ego_traj:       未来轨迹
    """
    ego = pb_frame.get('ego', {})
    ego_world = pb_frame.get('ego_world_pos', {})
    cross = pb_frame.get('cross', {})

    # 1. data_ego_curr_status
    data_ego_curr_status = {
        'v':                          ego.get('speed', 0.0),
        'yaw':                        ego_world.get('yaw', 0.0),
        'yaw_rate':                   ego.get('angular_velocity', 0.0),
        'x':                          0.0,
        'y':                          0.0,
        'egolane_traffic_lights':     ego.get('egolane_traffic_lights', []),
        'egolane_traffic_lights_pos': ego.get('egolane_traffic_lights_pos', []),
    }

    # 2. data_ego_local_pose
    data_ego_local_pose = {
        'v':        ego_world.get('v', ego.get('speed', 0.0)),
        'yaw':      ego_world.get('yaw', 0.0),
        'yaw_rate': ego_world.get('yaw_rate', ego.get('angular_velocity', 0.0)),
        'x':        0.0,
        'y':        0.0,
        'heading':  ego_world.get('yaw', 0.0),
        'timestamp': pb_frame.get('timestamp_ns', -1),
    }

    # 3. label_ego_hist_traj
    def _parse_traj_points(points_list):
        pos, yaw, v, yaw_rate, mask = [], [], [], [], []
        for p in points_list:
            pos.append([p.get('pos_x', 0.0), p.get('pos_y', 0.0)])
            yaw.append(p.get('yaw', 0.0))
            v.append([p.get('v_x', 0.0), p.get('v_y', 0.0)])
            yaw_rate.append(p.get('yaw_rate', 0.0))
            mask.append(bool(p.get('mask', 0.0)))
        return {'pos': pos, 'yaw': yaw, 'v': v, 'yaw_rate': yaw_rate, 'mask': mask}

    label_ego_hist_traj = _parse_traj_points(pb_frame.get('ego_history_time', []))

    # 4. label_ego_traj
    future_points = ego.get('future_traj', {}).get('points', [])
    label_ego_traj = _parse_traj_points(future_points)

    # 5. data_laneline: pts_fixed_num [{x,y},...] -> [[x,y],...]
    raw_lanelines = pb_frame.get('lanelines', [])
    data_laneline = []
    for lane in raw_lanelines:
        new_lane = dict(lane)
        pts = lane.get('pts_fixed_num', [])
        new_lane['pts_fixed_num'] = [
            [p['x'], p['y']] for p in pts
            if isinstance(p, dict)
        ]
        data_laneline.append(new_lane)

    # 6. data_laneline_navi_topo
    raw_navi = pb_frame.get('data_laneline_navi_topo', [])
    data_laneline_navi_topo = []
    for lane in raw_navi:
        new_lane = dict(lane)
        pts_raw = lane.get('pts_fixed_num', {})
        if isinstance(pts_raw, dict):
            pts_list = pts_raw.get('points', [])
        elif isinstance(pts_raw, list):
            pts_list = pts_raw
        else:
            pts_list = []
        new_lane['pts_fixed_num'] = [
            [p['x'], p['y']] for p in pts_list
            if isinstance(p, dict)
        ]
        data_laneline_navi_topo.append(new_lane)

    # 7. data_agent
    def _parse_agent(a: dict) -> dict:
        def _pts(field):
            raw = a.get(field, {})
            if isinstance(raw, dict):
                return [[p['x'], p['y']] for p in raw.get('points', [])
                        if isinstance(p, dict)]
            return []

        def _flat(field):
            raw = a.get(field, [])
            return raw if isinstance(raw, list) else []

        return {
            'id':           a.get('id', -1),
            'cls':          a.get('cls', -1),
            'x':            a.get('x', 0.0),
            'y':            a.get('y', 0.0),
            'vx':           a.get('vx', 0.0),
            'vy':           a.get('vy', 0.0),
            'yaw':          a.get('yaw', 0.0),
            'size_x':       a.get('size_x', 2.0),
            'size_y':       a.get('size_y', 4.5),
            'agent_mask':   a.get('agent_mask', False),
            'pos':          _pts('pos'),
            'velocity':     _pts('velocity'),
            'heading':      _flat('heading'),
            'valid_mask':   _flat('valid_mask'),
            'scores':       _flat('scores'),
            'raw_pos':      _pts('pos'),
            'raw_velocity': _pts('velocity'),
            'raw_heading':  _flat('heading'),
            'raw_valid_mask': _flat('valid_mask'),
        }

    data_agent = [_parse_agent(a) for a in pb_frame.get('agents', [])]

    # 8. data_roaditems
    data_roaditems = {
        'stoplines': cross.get('stoplines', []),
    }

    return {
        'data_ego_curr_status':    data_ego_curr_status,
        'data_ego_local_pose':     data_ego_local_pose,
        'label_ego_hist_traj':     label_ego_hist_traj,
        'label_ego_traj':          label_ego_traj,
        'data_laneline':           data_laneline,
        'data_laneline_navi_topo': data_laneline_navi_topo,
        'data_agent':              data_agent,
        'data_roaditems':          data_roaditems,
        'data_traffic_light':      pb_frame.get('data_traffic_light', []),
        'data_section':            {},
        'data_route':              None,
        '_raw':                    pb_frame,
    }


def load_frames_by_files(
    pb_dir: str,
    pb_filenames: List[str],
    max_workers: int = 0,
) -> Dict[str, dict]:
    """按文件名列表加载并归一化帧数据

    支持线程池并行加载: max_workers > 1 时使用 ThreadPoolExecutor

    Args:
        pb_dir: .pb 文件所在目录
        pb_filenames: pb 文件名列表 (如 ['1771887926405091968.pb', ...])
        max_workers: 并行加载线程数 (0=串行)

    Returns:
        {pb_filename: normalized_frame_dict}
    """
    result = {}

    if max_workers <= 1:
        # 串行加载
        for fname in pb_filenames:
            pb_path = os.path.join(pb_dir, fname)
            if not os.path.exists(pb_path):
                continue
            try:
                raw = load_pb_frame(pb_path)
                result[fname] = normalize_frame(raw)
            except Exception as e:
                print(f"[pb_loader] 加载失败 {fname}: {e}")
                continue
    else:
        # 并行加载
        def _load_one(fname):
            pb_path = os.path.join(pb_dir, fname)
            if not os.path.exists(pb_path):
                return fname, None
            try:
                raw = load_pb_frame(pb_path)
                return fname, normalize_frame(raw)
            except Exception as e:
                print(f"[pb_loader] 加载失败 {fname}: {e}")
                return fname, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load_one, f): f for f in pb_filenames}
            for future in as_completed(futures):
                fname, frame = future.result()
                if frame is not None:
                    result[fname] = frame

    return result
