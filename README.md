# 驾驶场景清洗与提取

从自动驾驶数据中识别、清洗和提取 5 类驾驶场景特征，生成训练用 JSON 清单。

## 场景类型

| 类型 | 名称 | 触发标签 |
|------|------|----------|
| Scene 1 | 路口停车 | `at_intersection` + `brake2stop`|
| Scene 2 | 头车起步 | `static2move` |
| Scene 3 | 跟车 | `non_intersection_lane_keep` - `large_curvature_lane_keep`|
| Scene 4 | 跟停 | `brake2stop` |
| Scene 5 | 变道 | `left_lane_change_effi`, `right_lane_change_effi` |

## 流水线流程

```
frame_scene.json ──> 标签初筛 ──> 分组合并 ──> pb帧加载 ──> 算法确认 ──> 合规检查 ──> 特征提取 ──> 输出
```

**两阶段检测架构：**

1. **阶段1 - 标签初筛**：从 `frame_scene.json` 匹配 7 种有效标签，合并连续帧为候选片段，扩展时间窗口
2. **阶段2 - 算法确认**：加载 pb 帧数据，运行场景专用检测算法做二次确认，避免标签误判

**合规检查**（通过后进入特征提取）：
- TTC (碰撞时间) 阈值检查
- 路口停车不压停止线检查
- 跟停最小距离检查

## 项目结构

```
personalized_data_process/
├── main.py                          # CLI 入口，流水线编排
├── config.py                        # 全局配置参数
├── proto/                           # Protobuf schema
│   └── dlp_raw_data_pb2.py
├── data_loader/
│   ├── index_loader.py              # driver_split / date_split / frame_scene 加载
│   └── pb_loader.py                 # pb 帧反序列化与归一化
├── grouper/
│   ├── scene_classifier.py          # 标签 -> 场景分类
│   ├── frame_grouper.py             # 按 (driver, scene) 分组
│   └── segment_merger.py            # 连续帧合并 + 时间窗口扩展
├── detection/
│   ├── detector.py                  # 检测调度器
│   └── scene_detectors.py           # 5 种场景检测算法
├── compliance/
│   ├── checker.py                   # 合规调度器
│   ├── ttc_checker.py               # TTC 检查
│   ├── stopline_checker.py          # 停止线压线检查
│   └── following_distance_checker.py # 跟停距离检查
├── feature_extraction/
│   ├── base_extractor.py            # 直接/间接特征计算
│   ├── general_features.py          # 14 个通用特征
│   └── scene_extractors.py          # 5 种场景专用特征
├── output/
│   ├── training_manifest.py         # 训练清单 JSON 生成
|   ├── streaming_writer.py          # 流写入的writer对象
│   └── feature_writer.py            # 场景特征输出
└── utils/
    └── frame_utils.py               # 前车检测、横向位置、车道线等工具
```

## 环境依赖

- Python >= 3.8
- protobuf >= 3.19
- numpy >= 1.21

## 快速开始

### 1. 配置数据路径

编辑 `config.py` 中的 `DATA_PATHS`：

```python
DATA_PATHS = {
    "driver_split_file": "./reference/driver_split.json",   # driver_split.json 路径
    "data_batch_file": "/path/to/data_batch.json",        # data_batch.json 路径 (记录所有 batch 目录)
    "t3_root_dir": "/path/to/t3_root",            # cloud_path 中 gt_label 之前的前缀替换为此路径
    "output_dir": "/path/to/output",                              # 输出目录
}
```

### 2. 运行

```bash
# 使用 config.py 中的路径
python main.py

# 或通过命令行参数覆盖
python main.py \
    --phase 1 \
    --config /path/to/config \
    --output-dir /path/to/output
```

### 3. 输出

运行后在输出目录生成两个文件：

- **`training_manifest.json`** — 训练数据清单，按驾驶员和场景类型组织，包含片段元信息、检测指标和合规结果
- **`scene_features.json`** — 场景特征文件，包含每个已确认片段的 14 个通用特征和场景专用特征

## 数据格式

### 输入

- **`frame_scene.json`**：每个数据包一个，包含每帧的场景标签列表
- **`driver_split.json`**：驾驶员切分信息，结构为 `{driver_id: {date: {vehicle_id: {}}}}`
- **`date_split.json`**：日期切分索引
- **`.pb` 文件**：Protobuf 格式的帧数据，包含 ego 状态、agent 列表、车道线、road items 等

### 目录命名规则

pb 数据目录名格式为 `YYYY_MM_DD_HH_MM_SS_dlp-VEHICLE_ID`，系统自动从中提取日期和车辆 ID，通过 `driver_split.json` 映射到驾驶员。

## 特征说明

### 通用特征 (14 个)

| 特征 | 说明 |
|------|------|
| `avg_speed` / `speed_var` / `speed_95th` / `speed_5th` | 速度统计 |
| `avg_acceleration` / `max_acceleration` / `min_acceleration` / `acceleration_std` | 加速度统计 |
| `avg_lateral_acc` / `peak_lateral_acc` / `lateral_acc_std` | 横向加速度统计 |
| `avg_yaw_rate` / `peak_yaw_rate` / `yaw_rate_std` | 横摆角速度统计 |

### 场景专用特征

- **Scene 1 (路口停车)** — 17 个特征：减速时间、5 阶段速度/加速度均值、减速效率
- **Scene 2 (起步)** — 16 个特征：加速到 80% 时间、5 阶段速度/加速度均值、加速效率
- **Scene 3 (跟车)** — 20 个特征：TTC 统计、THW 统计、相对速度统计及 5 阶段均值
- **Scene 4 (跟停)** — 19 个特征：静止距离、增强 TTC/THW、5 阶段前车距离
- **Scene 5 (变道)** — 15 个特征：横向位移、参考帧变化、变道前后前车距离对比

5 阶段时序特征 (`_phase_means`) 将事件过程等分为 5 段，取每段均值，用于捕捉过程的时序模式。

## 配置参数

所有阈值集中在 `config.py`，主要分组：

- **`SCENE_LABEL_MAP`** — 标签到场景的映射规则
- **`COMPLIANCE_PARAMS`** — 合规检查阈值 (TTC、跟停距离、停止线距离等)
- **`MERGE_PARAMS`** — 片段合并参数 (最大间隔、最短时长、帧率)
- **`WINDOW_EXTENSION`** — 时间窗口扩展参数
- **`DETECTION_PARAMS`** — 各场景检测确认阈值
- **`FEATURE_PARAMS`** — 特征提取参数 (平滑窗口、车道宽度范围等)

支持通过 JSON 配置文件覆盖，格式：

```json
{
  "data_paths": {
    "frame_scene_base_dir": "/custom/path"
  }
}
```
