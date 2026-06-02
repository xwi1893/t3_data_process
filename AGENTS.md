# AGENTS.md — 云平台适配指南

## 云平台目录结构

```
{cloud_root}/
├── batch0/
│   ├── date/                          # 每个 dir_key 一个子目录
│   │   ├── 2026_02_24_06_56_09_dlp-LDP41B960SD165599/
│   │   │   ├── frame_scene.json       ← 帧级场景标签 {pb_file: [labels]}
│   │   │   ├── frame_label.json
│   │   │   ├── frame_analysis.json
│   │   │   ├── label_count.json
│   │   │   ├── scene_count.json
│   │   │   └── scene_frame.json
│   │   ├── 2026_02_24_07_05_24_dlp-LDP41B96XSD165867/
│   │   │   └── ... (同上)
│   │   └── ...                        # 约 790 个 dir_key 目录
│   ├── date_split.json                ← 云路径映射 {dir_key: cloud_path}
│   ├── dlp_raw_data.proto             # protobuf 定义
│   ├── scene/                         # 场景数据 (约 76 个子目录)
│   ├── slot_scene/
│   ├── scene_statistics/
│   ├── version.json
│   └── vis/
├── batch1/
│   └── ... (同 batch0)
└── batchN/
    └── ...
```

### 关键文件说明

| 文件 | 路径 | 内容 | 用途 |
|------|------|------|------|
| `date_split.json` | `{batch}/date_split.json` | `{dir_key: cloud_path}` | dir_key → pb 云路径映射 |
| `frame_scene.json` | `{batch}/date/{dir_key}/frame_scene.json` | `{pb_filename: [labels]}` | 每帧的场景标签 |
| `driver_split.json` | config 中配置路径 | `{driver_id: {date: {vehicle_id: {}}}}` | 驾驶员索引，确保每个 dir_key 能关联驾驶员 ID |

### 云路径格式

`date_split.json` 中的 cloud_path 示例:
```
t3-st-dataloop-1360643540/gt_label/vd-dlp-data-new/20260326/dlpReplayerGt/.../gt_labels/prod_dumping_version_3
```
实际使用时需将 `gt_label` 之前的前缀替换为 `config.DATA_PATHS["t3_root_dir"]`，得到:
```
{t3_root_dir}/gt_label/vd-dlp-data-new/20260326/dlpReplayerGt/.../gt_labels/prod_dumping_version_3
```
pb 文件位于该 cloud_path 目录下，文件名为 `{timestamp_ns}.pb`。

---

## 文件查找方案

### 核心思路

程序入口参数为 **`batch_root_dir`** (云平台根目录) 和 **`driver_split_file`** (驾驶员索引文件)，自动发现并遍历所有 batch。每个处理的 dir_key 必须能关联到有效的驾驶员 ID。

### Stage 1: Batch 发现

```
config文件中会给定data_batch.json的地址，里面记录了所有batch目录，遍历这些目录
```

### Stage 2: driver_split.json 加载

```
加载 driver_split.json: {driver_id: {date: {vehicle_id: {}}}}
构建反向索引: (date, vehicle_id) → driver_id
```

该索引用于后续从 dir_key 中解析驾驶员 ID。dir_key 格式为 `YYYY_MM_DD_HH_MM_SS_dlp-{vehicle_id}`，
可提取出 `(date=YYYY-MM-DD, vehicle_id)` 进行查找。

**约束**: 无法匹配到 driver_id 的 dir_key 直接跳过，不进入后续流水线。

### Stage 3: date_split.json 加载与合并

```
对每个 batch:
  加载 {batch}/date_split.json
  合并到全局 date_split: {dir_key: cloud_path}
```

合并后得到完整的 `{dir_key: cloud_path}` 字典。

### Stage 4: frame_scene.json 发现与加载

```
对每个 batch:
  遍历 {batch}/date/ 下的每个子目录:
    子目录名即为 dir_key (如 2026_02_24_06_56_09_dlp-LDP41B960SD165599)
    验证 driver_id: 从 dir_key 提取 (date, vehicle_id) → 查反向索引
    若无匹配 driver_id → 跳过该目录
    加载 {batch}/date/{dir_key}/frame_scene.json
    得到 {pb_filename: [labels]}
    执行后续的标签筛选 → 分组 → 合并流程
```

### Stage 5: pb 数据加载

对每个产生的片段 (segment)，利用 cloud_path 定位 pb 文件:
```
pb_dir = cloud_path  (gt_label 前缀已替换为 t3_root_dir)
加载 {pb_dir}/{timestamp_ns}.pb
```

### 配置变更

| 原配置项 | 变更后 | 说明 |
|----------|--------|------|
| `frame_scene_base_dir` | **移除** | 改为从 batch 结构自动发现 |
| `date_split_file` | **移除** | 改为从各 batch 自动发现合并 |
| `pb_data_base_dir` | **移除** | pb 路径改为从 cloud_path 推导 |
| `batch_root_dir` | **新增** | 云平台根目录 (如 `/mnt/data/`) |
| `t3_root_dir` | **保留** | cloud_path 前缀替换 |
| `driver_split_file` | **保留** | 驾驶员索引文件路径，用于构建 (date, vehicle_id) → driver_id 反向索引 |

### 流水线改造要点

1. **`main.py`**: `run_pipeline` 参数简化为 `(batch_root_dir, driver_split_file)`，加载 driver_split 构建反向索引作为前置步骤
2. **`data_loader/index_loader.py`**: 新增 `discover_batches(root)` 和 `load_merged_date_split(batches)` 函数
3. **`main.py` Stage 4**: frame_scene 发现时先验证 driver_id，无匹配的 dir_key 跳过
4. **`main.py` Stage 5**: pb 目录从 `cloud_path` 推导，不再用 `pb_data_dir + dir_key` 拼接
5. **`grouper/frame_grouper.py`**: cloud_path 前缀替换逻辑 (已实现)

---

## 提速方案

目录众多 (每个 batch 约 790 个 date 目录，多个 batch)，以下是可行的优化方向:

### 1. 多进程并行处理 batch

**适用场景**: Stage 4 (标签筛选 + 分组合并) 和 Stage 5 (帧加载 + 检测 + 特征提取)

- **batch 级并行**: 用 `concurrent.futures.ProcessPoolExecutor` 并行处理不同 batch 的 date 目录扫描和 frame_scene 加载
- **dir_key 级并行**: 在每个 batch 内，将 dir_key 列表分片到多个 worker 并行执行 group_and_merge
- 注意: 多进程下需避免共享状态冲突，date_split 和 driver_reverse_index 为只读可安全共享

### 2. 懒加载 frame_scene.json

**现状**: Stage 4 中先加载所有 frame_scene.json，再执行 group_and_merge

- 改为先检查目录是否包含有效标签 (如通过 label_count.json 快速预筛)，再加载 frame_scene.json
- 避免加载大量无目标场景标签的目录

### 3. date_split.json 缓存

- 合并后的 date_split 字典可序列化为 pickle/msgpack 缓存
- 下次运行时若 date_batch.json 未变化，直接加载缓存，跳过 JSON 解析

### 4. pb 加载优化

- pb 反序列化是 Stage 5 的主要耗时，可用线程池并行加载同一 segment 内的多个 pb 文件
- 对于只需要部分帧的 segment，只加载所需的 pb 文件 (当前已实现)

### 5. 进度显示

- 用 `tqdm` 替换 print，显示 batch/dir_key 级别的进度条
- 方便定位慢点并评估优化效果
