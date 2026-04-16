"""
主流程管道
串联所有模块，实现完整的视频筛选与标签生成流程
"""

import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.db import VideoDatabase
from src.frame_extractor import FrameExtractor, QualityAnalyzer
from src.ai_analysis import YOLODetector, MediaPipeAnalyzer, GPT4oEvaluator
from src.filter import VideoScorer, LabelGenerator
from src.storage import DatasetStorage
from src.utils import ProgressTracker

logger = logging.getLogger(__name__)


class VideoFilterPipeline:
    """视频数据自动筛选与标签生成管道"""

    def __init__(
        self,
        db_config: Optional[Dict] = None,
        enable_yolo: bool = True,
        enable_mediapipe: bool = True,
        enable_gpt4o: bool = True,
    ):
        """
        初始化管道

        Args:
            db_config: 数据库配置（None则使用默认配置）
            enable_yolo: 是否启用YOLO检测
            enable_mediapipe: 是否启用MediaPipe姿态分析
            enable_gpt4o: 是否启用GPT-4o评估
        """
        logger.info("=" * 60)
        logger.info("视频数据自动筛选与标签生成系统 v1.0")
        logger.info("=" * 60)

        # 初始化各模块
        self.db = VideoDatabase(db_config)
        self.frame_extractor = FrameExtractor()
        self.quality_analyzer = QualityAnalyzer()

        self.yolo = YOLODetector() if enable_yolo else None
        self.mediapipe = MediaPipeAnalyzer() if enable_mediapipe else None
        self.gpt4o = GPT4oEvaluator() if enable_gpt4o else None

        self.scorer = VideoScorer()
        self.label_generator = LabelGenerator()
        self.storage = DatasetStorage()

        self._report = {
            "total_db_videos": 0,
            "filtered_videos": 0,
            "processed_videos": 0,
            "usable_videos": 0,
            "rejected_videos": 0,
            "errors": [],
        }

    def run(
        self,
        limit: Optional[int] = None,
        batch_size: int = 10,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        运行完整的筛选管道

        Args:
            limit: 处理视频数量限制（None表示处理全部）
            batch_size: 批处理大小
            skip_existing: 是否跳过已处理的视频

        Returns:
            处理报告
        """
        start_time = time.time()

        try:
            # Step 1: 数据库筛选
            logger.info("\n📋 Step 1/6: 数据库筛选")
            logger.info("-" * 40)
            self._report["total_db_videos"] = self.db.get_video_count()
            video_records = self.db.filter_videos(limit=limit)
            self._report["filtered_videos"] = len(video_records)
            logger.info(f"数据库中共 {self._report['total_db_videos']} 个视频")
            logger.info(f"筛选后剩余 {len(video_records)} 个横屏视频")

            if not video_records:
                logger.warning("没有符合条件的视频，流程结束")
                return self._finalize_report(start_time)

            # Step 2: 逐视频处理
            logger.info(f"\n🎬 Step 2/6: 开始处理 {len(video_records)} 个视频")
            logger.info("-" * 40)

            all_labels = []
            all_stored = []
            progress = ProgressTracker(len(video_records), "处理进度")

            for idx, record in enumerate(video_records):
                video_id = str(record.get("id", record.get("video_id", f"vid_{idx}")))
                video_path = self.db.get_full_path(record)

                logger.info(f"\n--- 处理视频 [{idx + 1}/{len(video_records)}]: {video_id} ---")

                # 检查是否已处理
                if skip_existing and self._is_already_processed(video_id):
                    logger.info(f"视频 {video_id} 已处理，跳过")
                    progress.update(1)
                    continue

                try:
                    result = self._process_single_video(
                        video_id, video_path, record
                    )
                    if result:
                        label, stored = result
                        all_labels.append(label)
                        if stored:
                            all_stored.append(stored)
                except Exception as e:
                    logger.error(f"处理视频 {video_id} 失败: {e}")
                    self._report["errors"].append({
                        "video_id": video_id,
                        "error": str(e),
                    })

                self._report["processed_videos"] += 1
                progress.update(1)

            # Step 3: 生成数据集清单
            logger.info(f"\n📊 Step 3/6: 生成数据集清单")
            logger.info("-" * 40)
            manifest_path = self.storage.generate_manifest(all_stored, all_labels)

            # Step 4: 输出统计信息
            self._report["usable_videos"] = len(all_stored)
            self._report["rejected_videos"] = (
                self._report["processed_videos"] - self._report["usable_videos"]
            )

            stats = self.storage.get_statistics()
            logger.info(f"\n📈 数据集统计:")
            logger.info(f"  总处理视频: {self._report['processed_videos']}")
            logger.info(f"  可用视频: {self._report['usable_videos']}")
            logger.info(f"  拒绝视频: {self._report['rejected_videos']}")
            logger.info(f"  类别数量: {stats['total_categories']}")
            logger.info(f"  错误数量: {len(self._report['errors'])}")

            if stats["category_distribution"]:
                logger.info(f"\n  类别分布 (Top 10):")
                for cat, count in list(stats["category_distribution"].items())[:10]:
                    logger.info(f"    {cat}: {count}")

            return self._finalize_report(start_time)

        finally:
            # 清理资源
            self.db.close()
            if self.mediapipe:
                self.mediapipe.close()

    def _process_single_video(
        self,
        video_id: str,
        video_path: str,
        record: Dict,
    ) -> Optional[tuple]:
        """处理单个视频的完整流程"""
        from pathlib import Path

        # 检查视频文件是否存在
        if not Path(video_path).exists():
            logger.warning(f"视频文件不存在: {video_path}")
            return None

        # Step 2.1: 抽帧
        logger.info(f"  [1/4] 抽帧中...")
        frames = self.frame_extractor.extract_frames(video_path, video_id)
        if not frames:
            logger.warning(f"  抽帧失败，跳过视频 {video_id}")
            return None

        # Step 2.2: 质量分析
        logger.info(f"  [2/4] 质量分析中...")
        frames_with_quality = self.quality_analyzer.analyze_frames(frames)

        # 过滤低质量帧
        good_frames = [f for f in frames_with_quality if f["quality"]["is_pass"]]
        if not good_frames:
            logger.warning(f"  所有帧质量不达标，跳过视频 {video_id}")
            return None
        logger.info(f"  质量通过: {len(good_frames)}/{len(frames_with_quality)} 帧")

        # Step 2.3: AI分析
        logger.info(f"  [3/4] AI分析中...")
        frame_paths = [f["frame_path"] for f in good_frames]

        # YOLO检测
        yolo_results = []
        if self.yolo:
            yolo_results = self.yolo.detect_batch(frame_paths)
            has_face = any(r.get("has_face_frontal", False) for r in yolo_results)
            if has_face:
                logger.info(f"  ⚠️ 检测到正面人脸，标记为不可用")

        # MediaPipe动作分析
        motion_results = []
        if self.mediapipe:
            for fp in frame_paths:
                motion_results.append(self.mediapipe.analyze_motion(fp))

        # GPT-4o评估（仅对前3帧评估以节省API调用）
        gpt_results = []
        if self.gpt4o and self.gpt4o.enabled:
            sample_frames = frame_paths[:min(3, len(frame_paths))]
            gpt_results = self.gpt4o.evaluate_batch(sample_frames)

        # Step 2.4: 评分与标签生成
        logger.info(f"  [4/4] 评分与标签生成...")
        score_result = self.scorer.score_video(
            video_id, good_frames, yolo_results, gpt_results, motion_results
        )

        label = self.label_generator.generate_label(
            record, score_result, good_frames, yolo_results, gpt_results,
            motion_results, good_frames,
        )
        # 补充视频路径
        label["video"]["file_path"] = video_path

        # 保存标签
        self.label_generator.save_label(label, video_id)

        # 输出评分结果
        logger.info(
            f"  评分: {score_result['total_score']:.1f} | "
            f"可用: {'✅' if score_result['is_usable'] else '❌'} | "
            f"拒绝原因: {', '.join(score_result['rejection_reasons']) or '无'}"
        )

        # 存储到数据集
        stored = None
        if score_result["is_usable"]:
            stored = self.storage.store_video(label, video_path)
            if stored:
                logger.info(f"  ✅ 已存储到: {stored['category']}")

        return label, stored

    def _is_already_processed(self, video_id: str) -> bool:
        """检查视频是否已处理"""
        label_path = self.label_generator.output_dir / f"{video_id}_label.json"
        return label_path.exists()

    def _finalize_report(self, start_time: float) -> Dict[str, Any]:
        """生成最终报告"""
        elapsed = time.time() - start_time
        self._report["elapsed_seconds"] = round(elapsed, 2)
        self._report["elapsed_formatted"] = self._format_time(elapsed)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"处理完成！耗时: {self._report['elapsed_formatted']}")
        logger.info(f"{'=' * 60}")

        return self._report

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}分钟"
        else:
            return f"{seconds / 3600:.1f}小时"
