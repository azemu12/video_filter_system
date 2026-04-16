#!/usr/bin/env python3
"""
视频数据自动筛选与标签生成系统 - CLI入口

用法:
    python main.py run                    # 运行完整管道
    python main.py run --limit 50         # 限制处理50个视频
    python main.py run --no-gpt4o         # 禁用GPT-4o评估
    python main.py run --db-type mysql    # 使用MySQL数据库
    python main.py setup-db               # 初始化示例数据库
    python main.py info                   # 查看数据库信息
"""

import argparse
import os
import sys
import logging

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging


def cmd_run(args):
    """运行完整筛选管道"""
    from src.pipeline import VideoFilterPipeline

    # 构建数据库配置
    db_config = None
    if args.db_type:
        from config.settings import DB_CONFIG
        db_config = DB_CONFIG.copy()
        db_config["type"] = args.db_type

    pipeline = VideoFilterPipeline(
        db_config=db_config,
        enable_yolo=not args.no_yolo,
        enable_mediapipe=not args.no_mediapipe,
        enable_gpt4o=not args.no_gpt4o,
    )

    report = pipeline.run(
        limit=args.limit,
        batch_size=args.batch_size,
        skip_existing=not args.force,
    )

    # 输出JSON报告
    if args.output_report:
        import json
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存到: {args.output_report}")


def cmd_setup_db(args):
    """初始化示例数据库（用于测试）"""
    from src.db.sample_db import create_sample_database
    create_sample_database(args.db_path)
    print(f"示例数据库已创建: {args.db_path}")


def cmd_info(args):
    """查看数据库信息"""
    from src.db import VideoDatabase

    db = VideoDatabase()
    try:
        total = db.get_video_count()
        tags = db.get_all_tags()

        print(f"\n数据库信息:")
        print(f"  总视频数: {total}")
        print(f"  符合筛选条件的视频数: {db.get_video_count()}")
        print(f"  所有标签 ({len(tags)}):")
        for tag in tags[:50]:
            print(f"    - {tag}")
        if len(tags) > 50:
            print(f"    ... 还有 {len(tags) - 50} 个标签")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="视频数据自动筛选与标签生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", help="日志级别 (DEBUG/INFO/WARNING)")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行完整筛选管道")
    run_parser.add_argument("--limit", type=int, default=None, help="限制处理视频数量")
    run_parser.add_argument("--batch-size", type=int, default=10, help="批处理大小")
    run_parser.add_argument("--force", action="store_true", help="强制重新处理（跳过已处理的视频）")
    run_parser.add_argument("--no-yolo", action="store_true", help="禁用YOLO检测")
    run_parser.add_argument("--no-mediapipe", action="store_true", help="禁用MediaPipe姿态分析")
    run_parser.add_argument("--no-gpt4o", action="store_true", help="禁用GPT-4o评估")
    run_parser.add_argument("--db-type", choices=["sqlite", "mysql", "postgresql"], help="数据库类型")
    run_parser.add_argument("--output-report", type=str, help="输出处理报告到JSON文件")

    # setup-db 命令
    setup_parser = subparsers.add_parser("setup-db", help="初始化示例数据库")
    setup_parser.add_argument("--db-path", default="videos.db", help="数据库文件路径")

    # info 命令
    subparsers.add_parser("info", help="查看数据库信息")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "setup-db":
        cmd_setup_db(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
