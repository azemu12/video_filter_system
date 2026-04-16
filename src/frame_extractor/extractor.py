"""
视频抽帧与质量分析模块
负责从视频中抽取关键帧并进行基础质量评估
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config.settings import FRAME_CONFIG, QUALITY_CONFIG, FRAMES_DIR

logger = logging.getLogger(__name__)


class FrameExtractor:
    """视频抽帧器"""

    def __init__(self, output_dir: Optional[Path] = None, config: Optional[Dict] = None):
        self.config = config or FRAME_CONFIG
        self.output_dir = output_dir or FRAMES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self,
        video_path: str,
        video_id: str,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        从视频中抽取帧

        Args:
            video_path: 视频文件路径
            video_id: 视频唯一标识
            max_frames: 最大抽帧数

        Returns:
            帧信息列表，每项包含 frame_path, frame_index, timestamp
        """
        import cv2

        max_frames = max_frames or self.config["max_frames"]
        strategy = self.config["strategy"]
        fps = self.config["fps"]
        img_format = self.config["image_format"]
        quality = self.config["image_quality"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps if video_fps > 0 else 0

        logger.info(
            f"视频 {video_id}: 总帧数={total_frames}, FPS={video_fps:.1f}, "
            f"时长={duration:.1f}s, 策略={strategy}"
        )

        # 创建视频专属输出目录
        video_frame_dir = self.output_dir / video_id
        video_frame_dir.mkdir(parents=True, exist_ok=True)

        frames = []

        if strategy == "uniform":
            frames = self._extract_uniform(
                cap, video_frame_dir, video_id, total_frames,
                video_fps, fps, max_frames, img_format, quality
            )
        elif strategy == "keyframe":
            frames = self._extract_keyframes(
                cap, video_frame_dir, video_id, total_frames,
                video_fps, max_frames, img_format, quality
            )
        elif strategy == "scene":
            frames = self._extract_scene_changes(
                cap, video_frame_dir, video_id, total_frames,
                video_fps, max_frames, img_format, quality
            )
        else:
            logger.warning(f"未知抽帧策略: {strategy}，使用均匀抽帧")
            frames = self._extract_uniform(
                cap, video_frame_dir, video_id, total_frames,
                video_fps, fps, max_frames, img_format, quality
            )

        cap.release()
        logger.info(f"视频 {video_id}: 抽取了 {len(frames)} 帧")
        return frames

    def _extract_uniform(
        self, cap, output_dir, video_id, total_frames,
        video_fps, sample_fps, max_frames, img_format, quality
    ) -> List[Dict]:
        """均匀抽帧"""
        import cv2

        frame_interval = max(1, int(video_fps / sample_fps))
        frames = []
        frame_idx = 0

        while frame_idx < total_frames and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / video_fps
            frame_path = self._save_frame(frame, output_dir, video_id, len(frames), img_format, quality)

            frames.append({
                "frame_path": str(frame_path),
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 3),
            })
            frame_idx += frame_interval

        return frames

    def _extract_keyframes(
        self, cap, output_dir, video_id, total_frames,
        video_fps, max_frames, img_format, quality
    ) -> List[Dict]:
        """基于场景变化的关键帧提取"""
        import cv2

        frames = []
        prev_frame = None
        frame_idx = 0
        threshold = QUALITY_CONFIG["scene_change_threshold"]

        while frame_idx < total_frames and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # 计算帧间差异
                diff = self._frame_difference(prev_frame, frame)
                if diff > threshold:
                    timestamp = frame_idx / video_fps
                    frame_path = self._save_frame(frame, output_dir, video_id, len(frames), img_format, quality)
                    frames.append({
                        "frame_path": str(frame_path),
                        "frame_index": frame_idx,
                        "timestamp": round(timestamp, 3),
                    })

            prev_frame = frame.copy()
            frame_idx += 1

        # 如果没有检测到场景变化，均匀抽取一些帧
        if not frames:
            return self._extract_uniform(
                cap, output_dir, video_id, total_frames,
                video_fps, 1.0, max_frames, img_format, quality
            )

        return frames

    def _extract_scene_changes(
        self, cap, output_dir, video_id, total_frames,
        video_fps, max_frames, img_format, quality
    ) -> List[Dict]:
        """场景变化检测（与keyframe类似但更激进）"""
        import cv2

        frames = []
        prev_hist = None
        frame_idx = 0
        threshold = QUALITY_CONFIG["scene_change_threshold"] * 0.7

        while frame_idx < total_frames and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算颜色直方图差异
            hist = self._compute_histogram(frame)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > threshold:
                    timestamp = frame_idx / video_fps
                    frame_path = self._save_frame(frame, output_dir, video_id, len(frames), img_format, quality)
                    frames.append({
                        "frame_path": str(frame_path),
                        "frame_index": frame_idx,
                        "timestamp": round(timestamp, 3),
                    })

            prev_hist = hist
            frame_idx += 1

        if not frames:
            return self._extract_uniform(
                cap, output_dir, video_id, total_frames,
                video_fps, 1.0, max_frames, img_format, quality
            )

        return frames

    def _frame_difference(self, frame1, frame2) -> float:
        """计算两帧之间的差异度"""
        import cv2

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0

    def _compute_histogram(self, frame):
        """计算颜色直方图"""
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _save_frame(self, frame, output_dir, video_id, frame_num, img_format, quality):
        """保存帧图像"""
        import cv2

        ext = f".{img_format}"
        frame_path = output_dir / f"{video_id}_frame_{frame_num:04d}{ext}"

        params = []
        if img_format.lower() in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif img_format.lower() == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 11]

        cv2.imwrite(str(frame_path), frame, params)
        return frame_path


class QualityAnalyzer:
    """画面质量分析器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or QUALITY_CONFIG

    def analyze(self, frame_path: str) -> Dict[str, float]:
        """
        分析单帧图像质量

        Returns:
            质量指标字典:
            - sharpness: 清晰度 (0-1)
            - brightness: 亮度 (0-1)
            - contrast: 对比度 (0-1)
            - is_pass: 是否通过质量检查
        """
        import cv2

        img = cv2.imread(frame_path)
        if img is None:
            logger.error(f"无法读取图像: {frame_path}")
            return {"sharpness": 0, "brightness": 0, "contrast": 0, "is_pass": False}

        sharpness = self._calc_sharpness(img)
        brightness = self._calc_brightness(img)
        contrast = self._calc_contrast(img)

        is_pass = (
            sharpness >= self.config["min_sharpness"]
            and self.config["min_brightness"] <= brightness <= self.config["max_brightness"]
            and contrast >= self.config["min_contrast"]
        )

        return {
            "sharpness": round(sharpness, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "is_pass": is_pass,
        }

    def analyze_frames(self, frames: List[Dict]) -> List[Dict]:
        """
        批量分析帧质量

        Args:
            frames: 帧信息列表（来自FrameExtractor）

        Returns:
            附加了质量指标的帧信息列表
        """
        results = []
        for frame_info in frames:
            quality = self.analyze(frame_info["frame_path"])
            frame_info["quality"] = quality
            results.append(frame_info)

        passed = sum(1 for f in results if f["quality"]["is_pass"])
        logger.info(f"质量分析完成: {passed}/{len(results)} 帧通过质量检查")
        return results

    def _calc_sharpness(self, img) -> float:
        """使用拉普拉斯算子计算清晰度"""
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 归一化到 0-1 范围
        return min(laplacian_var / 500.0, 1.0)

    def _calc_brightness(self, img) -> float:
        """计算平均亮度"""
        import cv2

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2])) / 255.0

    def _calc_contrast(self, img) -> float:
        """计算对比度（标准差）"""
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) / 128.0
