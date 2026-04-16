"""
自动筛选与标签生成模块
综合所有分析结果，进行最终筛选并生成结构化标签
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.settings import SCORING_CONFIG, LABELS_DIR

logger = logging.getLogger(__name__)


class VideoScorer:
    """视频综合评分器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or SCORING_CONFIG
        self.weights = self.config["weights"]

    def score_video(
        self,
        video_id: str,
        quality_results: List[Dict],
        yolo_results: List[Dict],
        gpt_results: List[Dict],
        motion_results: List[Dict],
    ) -> Dict[str, Any]:
        """
        综合评分

        Args:
            video_id: 视频ID
            quality_results: 质量分析结果列表
            yolo_results: YOLO检测结果列表
            gpt_results: GPT-4o评估结果列表
            motion_results: 动作分析结果列表

        Returns:
            综合评分结果
        """
        if not quality_results:
            return self._empty_score(video_id)

        # 1. 清晰度评分（取所有帧的平均值）
        sharpness_scores = [f["quality"]["sharpness"] for f in quality_results if "quality" in f]
        avg_sharpness = sum(sharpness_scores) / len(sharpness_scores) if sharpness_scores else 0
        sharpness_score = avg_sharpness * 100

        # 2. 构图评分（取GPT评估的平均值）
        composition_scores = [r.get("composition_score", 50) for r in gpt_results]
        avg_composition = sum(composition_scores) / len(composition_scores) if composition_scores else 50

        # 3. 背景动态感评分
        dynamics_scores = [r.get("background_dynamics", 50) for r in gpt_results]
        avg_dynamics = sum(dynamics_scores) / len(dynamics_scores) if dynamics_scores else 50

        # 4. 主体明确度评分
        clarity_scores = [r.get("subject_clarity", 50) for r in gpt_results]
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 50

        # 5. 动作幅度评分
        motion_scores = [r.get("motion_score", 0) for r in motion_results]
        avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0
        motion_score = avg_motion * 100

        # 6. 多样性评分（基于检测到的类别数量）
        all_categories = set()
        for r in yolo_results:
            for cat in r.get("category_counts", {}).keys():
                all_categories.add(cat)
        for r in gpt_results:
            for tag in r.get("tags", []):
                all_categories.add(tag)
        diversity_score = min(len(all_categories) / 5.0, 1.0) * 100

        # 加权计算总分
        total_score = (
            sharpness_score * self.weights["sharpness"]
            + avg_composition * self.weights["composition"]
            + avg_dynamics * self.weights["background_dynamics"]
            + avg_clarity * self.weights["subject_clarity"]
            + motion_score * self.weights["motion_intensity"]
            + diversity_score * self.weights["diversity"]
        )

        # 检查是否有正面人脸（一票否决）
        has_frontal_face = any(r.get("has_face_frontal", False) for r in yolo_results)

        # 检查GPT是否推荐
        gpt_suitable = any(r.get("is_suitable_for_bg_replacement", False) for r in gpt_results)

        # 质量通过率
        quality_pass_rate = (
            sum(1 for f in quality_results if f.get("quality", {}).get("is_pass", False))
            / len(quality_results)
            if quality_results else 0
        )

        is_usable = (
            total_score >= self.config["min_usability_score"]
            and not has_frontal_face
            and quality_pass_rate >= 0.5
        )

        return {
            "video_id": video_id,
            "total_score": round(total_score, 2),
            "is_usable": is_usable,
            "has_frontal_face": has_frontal_face,
            "quality_pass_rate": round(quality_pass_rate, 4),
            "gpt_recommended": gpt_suitable,
            "breakdown": {
                "sharpness": round(sharpness_score, 2),
                "composition": round(avg_composition, 2),
                "background_dynamics": round(avg_dynamics, 2),
                "subject_clarity": round(avg_clarity, 2),
                "motion_intensity": round(motion_score, 2),
                "diversity": round(diversity_score, 2),
            },
            "rejection_reasons": self._get_rejection_reasons(
                total_score, has_frontal_face, quality_pass_rate, gpt_suitable
            ),
        }

    def _get_rejection_reasons(
        self, total_score, has_frontal_face, quality_pass_rate, gpt_suitable
    ) -> List[str]:
        """获取拒绝原因列表"""
        reasons = []
        if total_score < self.config["min_usability_score"]:
            reasons.append(f"综合评分过低 ({total_score:.1f} < {self.config['min_usability_score']})")
        if has_frontal_face:
            reasons.append("检测到正面人脸")
        if quality_pass_rate < 0.5:
            reasons.append(f"质量通过率过低 ({quality_pass_rate:.1%})")
        if not gpt_suitable:
            reasons.append("AI评估不建议用于背景替换训练")
        return reasons

    def _empty_score(self, video_id: str) -> Dict:
        return {
            "video_id": video_id,
            "total_score": 0,
            "is_usable": False,
            "has_frontal_face": False,
            "quality_pass_rate": 0,
            "gpt_recommended": False,
            "breakdown": {
                "sharpness": 0, "composition": 0, "background_dynamics": 0,
                "subject_clarity": 0, "motion_intensity": 0, "diversity": 0,
            },
            "rejection_reasons": ["无分析数据"],
        }


class LabelGenerator:
    """结构化标签生成器"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or LABELS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_label(
        self,
        video_record: Dict,
        score_result: Dict,
        quality_results: List[Dict],
        yolo_results: List[Dict],
        gpt_results: List[Dict],
        motion_results: List[Dict],
        frames: List[Dict],
    ) -> Dict[str, Any]:
        """
        为单个视频生成完整的结构化标签

        Returns:
            完整的标签字典
        """
        video_id = video_record.get("id", video_record.get("video_id", "unknown"))

        # 聚合GPT评估结果
        scene_types = list(set(r.get("scene_type", "unknown") for r in gpt_results if r.get("scene_type")))
        subject_categories = list(set(
            r.get("subject_category", "unknown") for r in gpt_results if r.get("subject_category")
        ))
        all_tags = []
        for r in gpt_results:
            all_tags.extend(r.get("tags", []))
        all_tags = list(set(all_tags))

        # 聚合YOLO检测结果
        detected_objects = {}
        for r in yolo_results:
            for d in r.get("objects", []):
                cls = d["class"]
                if cls not in detected_objects:
                    detected_objects[cls] = {"count": 0, "max_confidence": 0, "max_area_ratio": 0}
                detected_objects[cls]["count"] += 1
                detected_objects[cls]["max_confidence"] = max(
                    detected_objects[cls]["max_confidence"], d["confidence"]
                )
                detected_objects[cls]["max_area_ratio"] = max(
                    detected_objects[cls]["max_area_ratio"], d["area_ratio"]
                )

        # 最佳帧信息
        best_frame_idx = 0
        if quality_results:
            best_quality = max(
                range(len(quality_results)),
                key=lambda i: quality_results[i].get("quality", {}).get("sharpness", 0),
            )
            best_frame_idx = best_quality

        label = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "video": {
                "id": video_id,
                "file_path": video_record.get("file_path", ""),
                "width": video_record.get("width", 0),
                "height": video_record.get("height", 0),
                "tags": video_record.get("tags", ""),
                "duration": video_record.get("duration", 0),
            },
            "scene": {
                "type": scene_types[0] if scene_types else "unknown",
                "type_candidates": scene_types,
                "description": gpt_results[0].get("description", "") if gpt_results else "",
                "ai_tags": all_tags,
            },
            "subject": {
                "category": subject_categories[0] if subject_categories else "unknown",
                "category_candidates": subject_categories,
                "detected_objects": detected_objects,
                "has_person": any(r.get("has_person", False) for r in yolo_results),
                "has_frontal_face": any(r.get("has_face_frontal", False) for r in yolo_results),
            },
            "quality": {
                "sharpness": score_result["breakdown"]["sharpness"],
                "composition": score_result["breakdown"]["composition"],
                "background_dynamics": score_result["breakdown"]["background_dynamics"],
                "subject_clarity": score_result["breakdown"]["subject_clarity"],
                "motion_intensity": score_result["breakdown"]["motion_intensity"],
            },
            "scoring": {
                "total_score": score_result["total_score"],
                "is_usable": score_result["is_usable"],
                "gpt_recommended": score_result["gpt_recommended"],
                "quality_pass_rate": score_result["quality_pass_rate"],
                "rejection_reasons": score_result["rejection_reasons"],
            },
            "frames": {
                "total_extracted": len(frames),
                "best_frame_path": frames[best_frame_idx]["frame_path"] if frames else "",
                "best_frame_timestamp": frames[best_frame_idx]["timestamp"] if frames else 0,
            },
        }

        return label

    def save_label(self, label: Dict, video_id: str) -> Path:
        """保存标签到JSON文件"""
        file_path = self.output_dir / f"{video_id}_label.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)
        logger.debug(f"标签已保存: {file_path}")
        return file_path

    def save_batch_labels(self, labels: List[Dict]) -> List[Path]:
        """批量保存标签"""
        paths = []
        for label in labels:
            video_id = label["video"]["id"]
            path = self.save_label(label, video_id)
            paths.append(path)
        return paths
