"""
分类存储与数据集输出模块
将筛选后的视频按类别分类存储，输出训练数据集
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import STORAGE_CONFIG, DATASET_DIR

logger = logging.getLogger(__name__)


class DatasetStorage:
    """数据集分类存储器"""

    def __init__(self, output_dir: Optional[Path] = None, config: Optional[Dict] = None):
        self.config = config or STORAGE_CONFIG
        self.output_dir = output_dir or DATASET_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 按类别统计
        self.category_counts: Dict[str, int] = {}
        self.max_per_category = 100  # 可配置

    def store_video(
        self,
        label: Dict[str, Any],
        video_source_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        将视频分类存储到对应目录

        Args:
            label: 视频标签
            video_source_path: 视频源文件路径

        Returns:
            存储信息字典，如果超出类别限制则返回None
        """
        if not label["scoring"]["is_usable"]:
            return None

        scene_type = label["scene"]["type"]
        subject_category = label["subject"]["category"]

        # 构建分类路径
        category_key = f"{scene_type}/{subject_category}"
        category_dir = self.output_dir / category_key
        category_dir.mkdir(parents=True, exist_ok=True)

        # 检查类别数量限制
        current_count = self.category_counts.get(category_key, 0)
        if current_count >= self.max_per_category:
            logger.debug(f"类别 {category_key} 已达上限 {self.max_per_category}，跳过")
            return None

        video_id = label["video"]["id"]
        video_ext = Path(video_source_path).suffix
        dest_filename = f"{video_id}{video_ext}"
        dest_path = category_dir / dest_filename

        # 存储视频文件
        if self.config.get("copy_videos", False):
            if os.path.exists(video_source_path):
                shutil.copy2(video_source_path, dest_path)
                logger.debug(f"视频已复制: {dest_path}")
            else:
                logger.warning(f"视频文件不存在: {video_source_path}")
                return None
        elif self.config.get("symlink_videos", False):
            if os.path.exists(video_source_path):
                # 使用相对路径的符号链接
                try:
                    os.symlink(os.path.abspath(video_source_path), dest_path)
                    logger.debug(f"符号链接已创建: {dest_path}")
                except OSError:
                    # 符号链接创建失败时回退为复制
                    shutil.copy2(video_source_path, dest_path)
                    logger.debug(f"符号链接失败，已复制: {dest_path}")
            else:
                logger.warning(f"视频文件不存在: {video_source_path}")
                return None

        # 复制标签文件到分类目录
        label_dest = category_dir / f"{video_id}_label.json"
        with open(label_dest, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        # 复制最佳帧图像
        best_frame = label["frames"].get("best_frame_path", "")
        if best_frame and os.path.exists(best_frame):
            frame_ext = Path(best_frame).suffix
            frame_dest = category_dir / f"{video_id}_best_frame{frame_ext}"
            shutil.copy2(best_frame, frame_dest)

        self.category_counts[category_key] = current_count + 1

        return {
            "video_id": video_id,
            "category": category_key,
            "video_path": str(dest_path),
            "label_path": str(label_dest),
        }

    def store_batch(
        self,
        labels: List[Dict[str, Any]],
        video_records: List[Dict[str, Any]],
        get_video_path_func,
    ) -> List[Dict[str, Any]]:
        """
        批量存储视频

        Args:
            labels: 标签列表
            video_records: 视频记录列表
            get_video_path_func: 从视频记录获取文件路径的函数
        """
        results = []
        for label, record in zip(labels, video_records):
            video_path = get_video_path_func(record)
            result = self.store_video(label, video_path)
            if result:
                results.append(result)

        logger.info(f"批量存储完成: {len(results)}/{len(labels)} 个视频已存储")
        return results

    def generate_manifest(
        self,
        stored_results: List[Dict[str, Any]],
        labels: List[Dict[str, Any]],
    ) -> Path:
        """
        生成数据集清单文件

        Returns:
            清单文件路径
        """
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_videos": len(stored_results),
            "category_distribution": self.category_counts,
            "videos": [],
        }

        for result, label in zip(stored_results, labels):
            if label["scoring"]["is_usable"]:
                manifest["videos"].append({
                    "video_id": result["video_id"],
                    "category": result["category"],
                    "video_path": result["video_path"],
                    "label_path": result["label_path"],
                    "total_score": label["scoring"]["total_score"],
                    "scene_type": label["scene"]["type"],
                    "subject_category": label["subject"]["category"],
                })

        manifest_path = self.output_dir / self.config["manifest_filename"]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"数据集清单已生成: {manifest_path}")
        logger.info(f"总计: {len(stored_results)} 个视频, "
                     f"{len(self.category_counts)} 个类别")
        return manifest_path

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        total = sum(self.category_counts.values())
        return {
            "total_videos": total,
            "total_categories": len(self.category_counts),
            "category_distribution": dict(sorted(
                self.category_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )),
        }
