"""
scene_classifier.py
阶段1: 根据帧标签数组初步判定场景类型
"""

from typing import List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCENE_LABEL_MAP


def classify_frame(labels: List[str]) -> Optional[Tuple[int, str, str]]:
    """根据帧标签列表初步判定场景类型

    遍历标签列表，匹配 SCENE_LABEL_MAP 中第一个有效标签。
    一帧可能有多个标签，返回第一个匹配的场景。

    Args:
        labels: 帧标签列表, 如 ["brake", "at_intersection", "across"]

    Returns:
        (scene_type, scene_name_cn, scene_name_en) 或 None (忽略该帧)
        - scene_type: 1-5 整数
        - scene_name_cn: 中文名
        - scene_name_en: 英文名
    """
    if not labels:
        return None

    for label in labels:
        if label in SCENE_LABEL_MAP:
            return SCENE_LABEL_MAP[label]

    return None


def classify_frame_scene_all(frame_scene: dict) -> dict:
    """对整个 frame_scene.json 进行标签分类

    Args:
        frame_scene: {pb_filename: [labels]} 映射

    Returns:
        {pb_filename: (scene_type, name_cn, name_en)}
        仅包含匹配到有效场景的帧
    """
    classified = {}
    stats = {}

    for pb_file, labels in frame_scene.items():
        result = classify_frame(labels)
        if result is not None:
            classified[pb_file] = result
            scene_type = result[0]
            stats[scene_type] = stats.get(scene_type, 0) + 1

    total = len(frame_scene)
    matched = len(classified)
    print(f"[classifier] 帧标签分类: {total} 帧中 {matched} 帧匹配到场景")
    for st, count in sorted(stats.items()):
        name = {1: "路口停车", 2: "起步", 3: "跟车", 4: "跟停", 5: "变道"}.get(st, "?")
        print(f"  Scene {st} ({name}): {count} 帧")

    return classified
