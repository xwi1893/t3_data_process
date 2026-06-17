"""
场景清洗与提取 - 全局配置

所有可配置参数集中管理，包括：
- 场景标签映射（阶段1初步筛选）
- 合规检查阈值
- 片段合并参数
- 数据路径配置
"""

# === 场景标签映射 (阶段1: 标签初步筛选) ===
# frame_scene.json 中的标签 → (scene_type, 中文名, 英文名)
# 匹配规则: 先检查复合标签规则 (需多个标签同时存在), 再检查单标签规则

# 复合标签规则: 需要多个标签同时存在才能匹配场景
# 格式: {"required_labels": [...], "result": (scene_type, 中文名, 英文名)}
COMPOUND_LABEL_RULES = [
    {
        "required_labels": {"at_intersection", "brake2stop"},
        "result": (1, "路口停车", "intersection_stop"),
    },
]

# 单标签规则: 单个标签即可匹配
SCENE_LABEL_MAP = {
    "static2move":                                    (2, "起步",     "empty_start"),
    "brake2stop":                                     (4, "跟停",     "following_stop"),
    "left_lane_change_effi":                          (5, "变道",     "lane_change"),
    "right_lane_change_effi":                         (5, "变道",     "lane_change"),
}

# 排除标签规则: 需要某些标签存在且某些标签不存在才能匹配场景
# 格式: {scene_type: {"required_labels": set, "excluded_labels": set}}
SCENE_EXCLUDE_RULES = {
    3: {
        "required_labels": {"non_intersection_lane_keep"},
        "excluded_labels": {"large_curvature_lane_keep"},
        "result": (3, "跟车", "following_vehicle"),
    },
}

# 场景类型 → 英文名的反向映射（方便查找）
SCENE_TYPE_NAMES = {v[0]: v[2] for v in SCENE_LABEL_MAP.values()}
SCENE_TYPE_NAMES[1] = "intersection_stop"  # 复合规则补充
for _st, _rule in SCENE_EXCLUDE_RULES.items():
    SCENE_TYPE_NAMES[_st] = _rule["result"][2]

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

# === 采样参数 ===
# 从每个连续场景片段中均匀采样 k 帧，每帧独立做场景检测
# k_frames_per_scene: 按场景类型设置不同的采样帧数
SAMPLE_PARAMS = {
    "k_frames_default": 3,             # 未单独配置的场景使用此默认值
    "k_frames_per_scene": {
        1: 8,  # 路口停车
        2: 8,  # 起步
        3: 8,  # 跟车
        4: 4,  # 跟停
        5: 2,  # 变道
    },
}

# === 片段合并参数 ===
MERGE_PARAMS = {
    "max_gap_seconds": 0.5,            # 允许的最大帧间隔(秒), 约5帧
    "min_segment_duration": 1.0,       # 最短片段时长(秒)
    "fps": 10,                         # 帧率
    "frame_interval_ns": 100_000_000,  # ~0.1s = 100ms (纳秒)
}

# === 检测确认参数 (阶段2: 算法二次确认) ===
# 移植自参考项目的检测阈值
DETECTION_PARAMS = {
    "intersection_stop": {
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
        "max_lead_distance": 50.0,
        "min_lead_ratio": 0.6,
    },
    "following_stop": {
        "max_lead_distance": 50.0,
        "min_lead_distance": 5.0,
        "min_lead_ratio": 0.6,
    },
    "lane_change": {
        "yaw_rate_threshold": 0.1,
        "curvature_threshold": 0.08,
        "max_yaw_rate": 0.5,
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

# === 时域窗口扩展配置 ===
# 利用第一帧的历史数据和最后一帧的未来数据扩展特征时间窗口
WINDOW_EXTENSION = {
    "use_history": True,       # 使用第一帧的历史数据前补
    "use_future": True,        # 使用最后一帧的未来数据后补
    "max_history_frames": 30,  # 最多前补帧数
    "max_future_frames": 30,   # 最多后补帧数
}

# === 性能配置 ===
PERFORMANCE_CFG = {
    "batch_workers": 4,               # batch 级多进程 worker 数 (0=单进程)
    "pb_load_workers": 4,             # pb 文件并行加载线程数 (0=串行)
    "use_date_split_cache": True,     # 启用 date_split pickle 缓存
    "use_label_prescreen": True,      # 启用 label_count.json 预筛
    "progress_bar": True,             # 启用 tqdm 进度条
}

# === 数据路径配置 ===
# 用户需根据实际环境填写
DATA_PATHS = {
    "driver_split_file": "./reference/driver_split.json",   # driver_split.json 路径
    "data_batch_file": "/root/export/xuewei/data_batch_full.json",                                  # data_batch.json 路径 (记录所有 batch 目录)
    "t3_root_dir": "/root/t3-data/Preproduction_Data",     # cloud_path 中 gt_label 之前的前缀替换为此路径
    "output_dir": "./results",                              # 输出目录
}
