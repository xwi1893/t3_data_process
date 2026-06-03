"""
pb_downloader.py
从 phase1_segments.json 中选取场景，通过 rclone 下载对应的 pb 数据文件

路径转换规则:
  cloud_path 中以 /root/t3-data 开头的部分替换为 rclone 远程名
  例: /root/t3-data/Preproduction_Data/... -> Tianyiyun:t3-data/Preproduction_Data/...

用法:
  # 默认下载 test/test_output/phase1_segments.json 中的场景
  python utils/pb_downloader.py --remote Tianyiyun:t3-data

  # 指定 segments 文件和下载目录
  python utils/pb_downloader.py --remote Tianyiyun:t3-data \
      --input test/test_output/phase1_segments.json \
      --output test/downloads

  # 列出场景不下载
  python utils/pb_downloader.py --list
"""

import argparse
import json
import os
import sys
import subprocess


def load_segments(path: str) -> list:
    """加载 segments JSON"""
    if not os.path.exists(path):
        print(f"错误: 文件不存在: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_scene_list(segments: list):
    """打印场景列表"""
    print(f"\n共 {len(segments)} 个场景:\n")
    print(f"  {'#':>3}  {'场景类型':<12} {'驾驶员':<15} {'时长':>6} {'帧数':>4}  {'dir_key'}")
    print(f"  {'─'*3}  {'─'*12} {'─'*15} {'─'*6} {'─'*4}  {'─'*40}")
    for i, seg in enumerate(segments):
        print(f"  {i:>3}  "
              f"Scene {seg['scene_type']} ({seg['scene_name']}){'':<2}"
              f"{seg['driver_id']:<15} "
              f"{seg['duration_sec']:.1f}s{'':>3}"
              f"{seg['frame_count']:>4}  "
              f"{seg['dir_key'][:45]}")


def parse_selection(user_input: str, total: int) -> list:
    """解析用户选择输入

    支持: 单个索引、范围 (2-5)、逗号分隔 (0,2,4)、"all"
    可组合: "0,2-5,8"
    """
    user_input = user_input.strip()
    if user_input.lower() == 'all':
        return list(range(total))

    selected = set()
    for part in user_input.split(','):
        part = part.strip()
        if '-' in part:
            bounds = part.split('-')
            try:
                start, end = int(bounds[0]), int(bounds[1])
                selected.update(range(max(0, start), min(total, end + 1)))
            except ValueError:
                print(f"  [跳过] 无效范围: {part}")
        else:
            try:
                idx = int(part)
                if 0 <= idx < total:
                    selected.add(idx)
                else:
                    print(f"  [跳过] 索引越界: {idx}")
            except ValueError:
                print(f"  [跳过] 无效输入: {part}")

    return sorted(selected)


def build_rclone_path(cloud_path: str, remote: str, local_prefix: str) -> str:
    """将本地云路径转换为 rclone 远程路径

    例: cloud_path = /root/t3-data/Preproduction_Data/gt_label/...
        remote = Tianyiyun:t3-data
        local_prefix = /root/t3-data
        -> Tianyiyun:t3-data/Preproduction_Data/gt_label/...
    """
    prefix = local_prefix.rstrip('/')
    if cloud_path.startswith(prefix):
        relative = cloud_path[len(prefix):].lstrip('/')
        return f"{remote}/{relative}"
    # 前缀不匹配时直接拼接 (保底)
    return f"{remote}/{cloud_path.lstrip('/')}"


def download_scene(seg: dict, index: int, remote: str,
                   local_prefix: str, output_dir: str):
    """下载单个场景的 pb 文件"""
    cloud_path = seg['cloud_path']
    pb_files = seg['pb_files']
    scene_name = seg['scene_name_en']

    # 场景目录: scene_000_lane_change/
    scene_dir = os.path.join(output_dir, f"scene_{index:03d}_{scene_name}")
    os.makedirs(scene_dir, exist_ok=True)

    remote_path = build_rclone_path(cloud_path, remote, local_prefix)
    total = len(pb_files)

    print(f"\n  [{index}] Scene {seg['scene_type']} ({seg['scene_name']})")
    print(f"      远程: {remote_path[:80]}...")
    print(f"      文件: {total} 个 pb")
    print(f"      本地: {scene_dir}")

    # 构建 rclone 命令: 逐个文件用 --include 过滤
    cmd = ["rclone", "copy", remote_path, scene_dir]
    for pb_file in pb_files:
        cmd.extend(["--include", pb_file])
    cmd.extend(["--progress", "--stats-one-line"])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"      完成: {total} 个文件 -> {scene_dir}")
    except FileNotFoundError:
        print(f"      [错误] rclone 未安装，请先安装: https://rclone.org/install/")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"      [错误] rclone 返回 {e.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="从 segments 文件中选取场景，通过 rclone 下载 pb 数据"
    )
    parser.add_argument("--remote", type=str, default="Tianyiyun:t3-data",
                        help="rclone 远程路径 (默认 Tianyiyun:t3-data)")
    parser.add_argument("--local-prefix", type=str, default="/root/t3-data",
                        help="cloud_path 中需替换的本地前缀 (默认 /root/t3-data)")
    parser.add_argument("--input", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "test", "test_output", "phase1_segments.json"),
                        help="segments JSON 文件路径")
    parser.add_argument("--output", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "test", "downloads"),
                        help="下载输出目录")
    parser.add_argument("--list", action="store_true",
                        help="仅列出场景列表，不下载")
    args = parser.parse_args()

    # 加载 segments
    segments = load_segments(args.input)
    if not segments:
        print("无场景数据")
        sys.exit(0)

    # 打印场景列表
    print_scene_list(segments)

    if args.list:
        return

    # 验证 remote
    if not args.remote:
        print("\n错误: 请通过 --remote 指定 rclone 远程路径 (如 Tianyiyun:t3-data)")
        sys.exit(1)

    remote = args.remote

    # 交互式选择
    print(f"\n请选择要下载的场景 (索引号):")
    print("  示例: 0       下载第 0 个")
    print("        0,2,4   下载第 0、2、4 个")
    print("        0-5     下载第 0 到 5 个")
    print("        all     下载全部")
    user_input = input("\n选择> ").strip()

    if not user_input:
        print("未选择，退出")
        return

    selected = parse_selection(user_input, len(segments))
    if not selected:
        print("无有效选择，退出")
        return

    print(f"\n将下载 {len(selected)} 个场景到: {args.output}")
    confirm = input("确认? (y/N) ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 逐个下载
    for idx in selected:
        download_scene(segments[idx], idx, remote, args.local_prefix, args.output)

    print(f"\n下载完成，共 {len(selected)} 个场景 -> {args.output}")


if __name__ == "__main__":
    main()
