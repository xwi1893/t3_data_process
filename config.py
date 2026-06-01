"""
场景清洗与提取 - 全局配置

所有可配置参数集中管理，包括：
- 场景标签映射（阶段1初步筛选）
- 合规检查阈值
- 片段合并参数
- 时间窗口扩展参数
- 数据路径配置
"""

# === 场景标签映射 (阶段1: 标签初步筛选) ===
# frame_scene.json 中的标签 → (scene_type, 中文名, 英文名)
# 一帧有多个标签时，按此表匹配第一个有效标签；均无效则忽略
SCENE_LABEL_MAP = {
    "at_intersection":                                (1, "路口停车", "intersection_stop"),
    "static2move":                                    (2, "起步",     "empty_start"),
    "longi_interaction_follow_front_large_vehicle":   (3, "跟车",     "following_vehicle"),
    "longi_interaction_follow_front_small_vehicle":   (3, "跟车",     "following_vehicle"),
    "brake2stop":                                     (4, "跟停",     "following_stop"),
    "left_lane_change_effi":                          (5, "变道",     "lane_change"),
    "right_lane_change_effi":                         (5, "变道",     "lane_change"),
}

# 场景类型 → 英文名的反向映射（方便查找）
SCENE_TYPE_NAMES = {v[0]: v[2] for v in SCENE_LABEL_MAP.values()}

# === 合规检查参数 ===
# 先沿用参考项目参数，后续可通过 JSON 配置文件覆盖
COMPLIANCE_PARAMS = {
    "stop_speed_thresh": 0.1,          # m/s, 停车速度阈值
    "following_speed_thresh": 5.0,      # m/s, 跟车最低速度
    "headway_min": 0.0,                 # 秒, 时距下限
    "headway_max": 15.0,               # 秒, 时距上限
    "ttc_max": 10.0,                   # 秒, TTC 上限
    "min_following_distance": 5.0,      # 米, 最小跟停距离
    "min_lead_ratio": 0.6,             # 前车存在比例下限
    "avg_lead_dist_max": 50.0,         # 米, 平均前车距离上限
    "stopline_distance_threshold": 50.0,  # 米, 停止线检测距离
}

# === 片段合并参数 ===
MERGE_PARAMS = {
    "max_gap_seconds": 0.5,            # 允许的最大帧间隔(秒), 约5帧
    "min_segment_duration": 3.0,       # 最短片段时长(秒)
    "fps": 10,                         # 帧率
    "frame_interval_ns": 100_000_000,  # ~0.1s = 100ms (纳秒)
}

# === 时间窗口扩展 ===
WINDOW_EXTENSION = {
    "max_extend_seconds": 5.0,         # 最大扩展秒数
    "stop_labels": {"error_data"},     # 遇到这些标签停止扩展
}

# === 检测确认参数 (阶段2: 算法二次确认) ===
# 移植自参考项目的检测阈值
DETECTION_PARAMS = {
    "intersection_stop": {
        "min_intersection_ratio": 0.6,
        "min_first_vehicle_ratio": 0.6,
        "window_size": 600,
    },
    "empty_start": {
        "min_lead_distance": 50.0,
        "max_static_duration": 1.0,
        "accelerate_thresh": 0.2,
    },
    "following_vehicle": {
        "speed_thresh": 5.0,
        "headway_min": 0.0,
        "headway_max": 15.0,
        "headway_fluct_thresh": 1.0,
        "min_follow_duration": 3.0,
    },
    "following_stop": {
        "max_lead_distance": 50.0,
        "min_lead_distance": 5.0,
        "min_lead_ratio": 0.6,
    },
    "lane_change": {
        "max_intersection_ratio": 0.05,
        "yaw_rate_threshold": 0.1,
        "curvature_threshold": 0.08,
        "max_yaw_rate": 0.2,
        "max_avg_curvature": 0.08,
        "window_size": 600,
    },
}

# === 特征提取参数 ===
FEATURE_PARAMS = {
    "smooth_lead_dist_window": 5,
    "smooth_lateral_pos_window": 3,
    "max_time_headway": 15.0,
    "valid_lane_types": {1, 2, 3, 4, 5, 6, 10},
    "lane_width_min": 2.0,
    "lane_width_max": 5.0,
    "boundary_tolerance": 1.0,
    "lane_vehicle_tolerance": 0.5,
    "num_phases": 5,
}

# === 数据路径配置 ===
# 用户需根据实际环境填写
DATA_PATHS = {
    "frame_scene_base_dir": "",        # frame_scene.json 所在基础目录
    "driver_split_file": "",           # driver_split.json 路径
    "date_split_file": "",             # date_split.json 路径
    "pb_data_base_dir": "",            # pb 数据文件所在基础目录
    "output_dir": "",                  # 输出目录
}
