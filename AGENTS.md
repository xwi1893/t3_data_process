# AGENTS.md

本文件给在本仓库工作的 AI agent 使用。修改代码前请先阅读 `README.md`、`PRD.md` 和相关模块实现；当文档与代码不一致时，以当前代码行为为准，并同步更新文档。

## 项目概览

这是一个自动驾驶数据清洗与场景特征提取流水线。当前实现采用两阶段流程：

1. Phase 1：加载驾驶员索引和 batch 索引，读取 `frame_scene.json`，按标签筛选 5 类驾驶场景，识别连续片段并均匀采样 k 帧，输出 `phase1_samples.json`。
2. Phase 2：读取 Phase 1 采样结果，加载单个 `.pb` 帧，进行算法确认、合规检查和特征提取，输出训练清单和特征文件。

主要入口是 `main.py`。全局阈值、路径和性能参数集中在 `config.py`。

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
│   └── feature_writer.py            # 场景特征输出
└── utils/
    └── frame_utils.py               # 前车检测、横向位置、车道线等工具
```

## 目录职责

- `main.py`：CLI 入口和 Phase 1/Phase 2 流水线编排。
- `config.py`：场景标签、采样、合规、检测、特征、性能和数据路径配置。
- `data_loader/`：加载 `driver_split.json`、`data_batch.json`、各 batch 的 `date_split.json`、`frame_scene.json`，以及 `.pb` 帧数据。
- `grouper/`：标签分类、按驾驶员/场景分组、连续片段识别和 k 帧采样。
- `detection/`：5 类场景的算法二次确认。
- `compliance/`：TTC、停止线、跟停距离等合规检查。
- `feature_extraction/`：通用特征和场景专用特征提取。
- `output/`：流式 JSON 写入、训练清单和特征文件输出。
- `proto/`：protobuf schema 与生成代码。
- `test/`：阶段测试脚本，依赖真实数据路径。
- `reference/`：本地参考数据目录，可能包含 `driver_split.json` 等环境相关文件。

## 运行命令

默认运行两个阶段：

```bash
python main.py
```

只运行 Phase 1：

```bash
python main.py --phase 1
```

只运行 Phase 2：

```bash
python main.py --phase 2
```

通过 JSON 覆盖配置：

```bash
python main.py --config override.json --output-dir ./results
```

配置覆盖文件示例：

```json
{
  "data_paths": {
    "driver_split_file": "./reference/driver_split.json",
    "data_batch_file": "/path/to/data_batch.json",
    "output_dir": "./results"
  }
}
```

注意：`DATA_PATHS["t3_root_dir"]` 用于将云端 `cloud_path` 中 `gt_label` 之前的前缀替换为本地数据根路径。涉及 pb 路径推导时请检查 `grouper/frame_grouper.py` 和 `PRD.md`。

## 测试与验证

本仓库没有发现 `requirements.txt`、`pyproject.toml` 或 pytest 配置。测试脚本是可直接执行的 Python 脚本，需要真实数据路径：

```bash
python test/test_phase1_tag_screening.py --max-batches 2 --max-dirs 5
python test/test_phase2_detection.py --max-samples 5
```

全量测试：

```bash
python test/test_phase1_tag_screening.py
python test/test_phase2_detection.py
```

如果只改纯函数或局部逻辑，优先补充小范围验证；如果改动影响数据路径、采样、检测确认、合规或输出结构，至少运行对应阶段测试。若本地缺少真实数据，请在最终说明中明确哪些测试未运行以及原因。

## 依赖

README 中声明：

- Python >= 3.8
- protobuf >= 3.19
- numpy >= 1.21

代码还可选使用 `tqdm` 作为进度条；缺失时测试脚本会自动退化为无进度条。不要在没有确认的情况下新增重量级依赖。

## 编码约定

- 代码风格以现有 Python 文件为准：标准库优先，函数职责清晰，中文日志和中文注释可以保留。
- 配置阈值集中放在 `config.py`，不要把新阈值散落在业务逻辑里。
- 数据结构字段名要保持稳定，尤其是 Phase 1 输出给 Phase 2 使用的采样字段：
  - `sample_id`
  - `driver_id`
  - `scene_type`
  - `scene_name`
  - `scene_name_en`
  - `dir_key`
  - `cloud_path`
  - `pb_file`
  - `timestamp_ns`
  - `labels`
  - `confirmed`
  - `confirm_reason`
- 输出给训练使用的字段变更要同步检查 `output/training_manifest.py` 和 `output/feature_writer.py`。
- protobuf 生成文件 `proto/dlp_raw_data_pb2.py` 不要手改；如需更新，先修改 `.proto` 并用对应工具重新生成。

## 架构注意事项

- 当前 Phase 2 是“单 pb 采样条目”处理模型，不是旧版多 pb segment 模型。检测和特征提取应从单个 pb 内部的历史/未来时序数据构建序列。
- Phase 1 跳过无法从 `(date, vehicle_id)` 匹配到 `driver_id` 的 `dir_key`；不要绕过这个约束，除非需求明确变更。
- `date_split` 可能启用 `.cache/date_split_*.pkl` 缓存。修改 batch 发现或路径合并逻辑后，要考虑缓存是否会影响验证结果。
- `PERFORMANCE_CFG` 中有多进程和预筛选配置。并行 worker 的参数和返回值必须可序列化，不要传入不可 pickle 的对象。
- `label_count.json` 预筛应保持保守策略：缺失或解析失败时继续加载 `frame_scene.json`，避免误丢数据。
- 场景标签映射由复合规则和单标签规则共同决定，复合规则优先。新增标签时同步更新 `SCENE_LABEL_MAP`、`COMPOUND_LABEL_RULES` 或相关测试说明。

## 修改建议

- 小改动先从调用链读起：`main.py` -> 目标模块 -> 输出模块或测试脚本。
- 涉及路径解析时，同时检查 `data_loader/index_loader.py`、`grouper/frame_grouper.py` 和 `PRD.md` 的云平台路径约定。
- 涉及场景检测时，确认 `detection/scene_detectors.py`、`feature_extraction/general_features.py`、`feature_extraction/scene_extractors.py` 对输入 `features` 的预期一致。
- 涉及合规逻辑时，确认 `compliance/checker.py` 的调度结果字段是否仍被 manifest/features 输出消费。
- 不要提交本地输出目录、缓存、`__pycache__` 或大体积数据文件。

## 文档维护

`README.md` 仍包含部分旧版路径说明；`PRD.md` 和当前代码更接近云平台 batch + 单帧采样实现。若继续改造流水线，请同步更新 README、PRD 和本文件，避免后续 agent 按过期流程工作。
