"""
AI内容理解模块
混合方案：YOLO目标检测 + MediaPipe姿态估计 + OpenAI GPT-4o内容评估
"""

import base64
import json
import logging
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

from config.settings import AI_CONFIG

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO 目标检测器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or AI_CONFIG["yolo"]
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            model_name = self.config["model_name"]
            logger.info(f"加载 YOLO 模型: {model_name}")
            self.model = YOLO(model_name)
            logger.info("YOLO 模型加载成功")
        except ImportError:
            logger.error("请安装 ultralytics: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            raise

    def detect(self, frame_path: str) -> Dict[str, Any]:
        """
        对单帧图像进行目标检测

        Returns:
            检测结果字典:
            - objects: 检测到的目标列表
            - has_person: 是否检测到人物
            - has_face_frontal: 是否检测到正面人脸
            - person_bbox_ratio: 人物占画面比例
            - dominant_category: 主要类别
        """
        results = self.model(frame_path, conf=self.config["confidence_threshold"])
        result = results[0]

        detections = []
        target_classes = self.config["target_classes"]
        has_person = False
        person_area = 0
        total_area = 0

        img_width = result.orig_shape[1]
        img_height = result.orig_shape[0]
        total_area = img_width * img_height

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_ratio = bbox_area / total_area if total_area > 0 else 0

            detection = {
                "class": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "area_ratio": round(bbox_ratio, 4),
            }
            detections.append(detection)

            if cls_name in target_classes:
                if cls_name == "person":
                    has_person = True
                    person_area = max(person_area, bbox_area)

        # 判断是否正面人脸：人物占比大且位于画面中心
        has_face_frontal = False
        person_bbox_ratio = person_area / total_area if total_area > 0 else 0

        if has_person and person_bbox_ratio > 0:
            # 找到最大的人物框
            person_boxes = [d for d in detections if d["class"] == "person"]
            if person_boxes:
                largest_person = max(person_boxes, key=lambda x: x["area_ratio"])
                bbox = largest_person["bbox"]
                # 判断人物是否在画面中心区域
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                center_x = img_width / 2
                center_y = img_height / 2

                # 如果人物在画面中心且占比大，可能是正面
                in_center = (
                    abs(cx - center_x) / img_width < 0.3
                    and abs(cy - center_y) / img_height < 0.3
                )
                if in_center and person_bbox_ratio > 0.15:
                    has_face_frontal = True

        # 统计各类别数量
        category_counts = {}
        for d in detections:
            cls = d["class"]
            category_counts[cls] = category_counts.get(cls, 0) + 1

        dominant_category = max(category_counts, key=category_counts.get) if category_counts else "none"

        return {
            "objects": detections,
            "has_person": has_person,
            "has_face_frontal": has_face_frontal,
            "person_bbox_ratio": round(person_bbox_ratio, 4),
            "dominant_category": dominant_category,
            "category_counts": category_counts,
        }

    def detect_batch(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """批量检测"""
        results = []
        for fp in frame_paths:
            try:
                result = self.detect(fp)
                results.append(result)
            except Exception as e:
                logger.error(f"检测失败 {fp}: {e}")
                results.append({
                    "objects": [], "has_person": False,
                    "has_face_frontal": False, "person_bbox_ratio": 0,
                    "dominant_category": "none", "category_counts": {},
                })
        return results


class MediaPipeAnalyzer:
    """MediaPipe 姿态分析器（辅助判断动作幅度）"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or AI_CONFIG["mediapipe"]
        self.enabled = self.config.get("enabled", True)
        self.pose = None

        if self.enabled:
            self._init_pose()

    def _init_pose(self):
        """初始化MediaPipe Pose"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=self.config["min_detection_confidence"],
                min_tracking_confidence=self.config["min_tracking_confidence"],
            )
            logger.info("MediaPipe Pose 初始化成功")
        except ImportError:
            logger.warning("MediaPipe 未安装，动作幅度分析将被跳过")
            self.enabled = False
        except Exception as e:
            logger.warning(f"MediaPipe 初始化失败: {e}")
            self.enabled = False

    def analyze_motion(self, frame_path: str) -> Dict[str, Any]:
        """
        分析单帧中人物的动作幅度

        Returns:
            - motion_score: 动作幅度评分 (0-1)
            - has_motion: 是否有明显动作
            - pose_detected: 是否检测到人体姿态
        """
        if not self.enabled or self.pose is None:
            return {"motion_score": 0.5, "has_motion": False, "pose_detected": False}

        import cv2
        import numpy as np

        img = cv2.imread(frame_path)
        if img is None:
            return {"motion_score": 0, "has_motion": False, "pose_detected": False}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return {"motion_score": 0, "has_motion": False, "pose_detected": False}

        landmarks = results.pose_landmarks.landmark

        # 计算关键肢体部位的伸展程度来评估动作幅度
        # 使用肩、肘、腕、髋、膝、踝等关键点
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

        points = []
        for kp in key_points:
            lm = landmarks[kp.value]
            points.append((lm.x * w, lm.y * h))

        # 计算肢体伸展范围
        if len(points) >= 4:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            spread_x = (max(xs) - min(xs)) / w
            spread_y = (max(ys) - min(ys)) / h
            motion_score = min((spread_x + spread_y) / 1.5, 1.0)
        else:
            motion_score = 0.3

        return {
            "motion_score": round(motion_score, 4),
            "has_motion": motion_score > 0.4,
            "pose_detected": True,
        }

    def close(self):
        """释放资源"""
        if self.pose:
            self.pose.close()


class GPT4oEvaluator:
    """GPT-4o 视觉评估器（构图、质量、场景理解）"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or AI_CONFIG["openai"]
        if not self.config.get("api_key"):
            logger.warning("未设置 OPENAI_API_KEY，GPT-4o 评估将被跳过")
            self.enabled = False
        else:
            self.enabled = True
            self._init_client()

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config.get("base_url"),
                timeout=self.config.get("timeout", 60),
            )
            logger.info("OpenAI 客户端初始化成功")
        except ImportError:
            logger.error("请安装 openai: pip install openai")
            self.enabled = False

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def evaluate_frame(self, frame_path: str) -> Dict[str, Any]:
        """
        使用GPT-4o评估单帧图像

        Returns:
            评估结果字典:
            - scene_type: 场景类型
            - subject_category: 主体类别
            - composition_score: 构图评分 (0-100)
            - background_dynamics: 背景动态感评分 (0-100)
            - subject_clarity: 主体明确度评分 (0-100)
            - description: 场景描述
            - tags: AI生成的标签列表
            - is_suitable_for_bg_replacement: 是否适合背景替换训练
            - suitability_reason: 适合/不适合的原因
        """
        if not self.enabled:
            return self._default_evaluation()

        try:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(frame_path)
            if not mime_type:
                mime_type = "image/jpeg"

            base64_image = self._encode_image(frame_path)

            prompt = """请分析这张视频帧图像，用于"背景替换训练数据"的筛选评估。

请严格按以下JSON格式返回（不要添加任何其他文字）：
{
    "scene_type": "场景类型（如：自然风景/城市街景/室内场景/海洋沙滩/山脉森林/天空云彩/夜景灯光等）",
    "subject_category": "主体类别（如：人物/动物/车辆/建筑/植物/无明确主体等）",
    "composition_score": 构图质量评分(0-100整数，考虑三分法、引导线、层次感等)",
    "background_dynamics": 背景动态感评分(0-100整数，考虑背景是否有运动模糊、光影变化、粒子效果等)",
    "subject_clarity": 主体明确度评分(0-100整数，主体是否清晰可辨、与背景分离度如何)",
    "description": "简要描述画面内容（50字以内）",
    "tags": ["标签1", "标签2", "标签3"],
    "is_suitable_for_bg_replacement": true或false,
    "suitability_reason": "判断理由（30字以内）"
}

重要评估标准：
1. 画面构图要高级，避免随手拍的感觉
2. 背景优先选择动态感强的
3. 主体要明确，动作幅度大
4. 人物不能是正脸（侧面或背影可以）
5. 场景覆盖面要广，避免雷同"""

            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.config["max_tokens"],
            )

            content = response.choices[0].message.content.strip()

            # 提取JSON（处理可能的markdown代码块包裹）
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                # 确保数值类型正确
                for key in ["composition_score", "background_dynamics", "subject_clarity"]:
                    if key in result:
                        result[key] = int(result[key])
                return result
            else:
                logger.warning(f"GPT-4o 返回格式异常: {content[:200]}")
                return self._default_evaluation()

        except Exception as e:
            logger.error(f"GPT-4o 评估失败: {e}")
            return self._default_evaluation()

    def evaluate_batch(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """批量评估帧"""
        results = []
        for fp in frame_paths:
            result = self.evaluate_frame(fp)
            result["frame_path"] = fp
            results.append(result)
        return results

    def _default_evaluation(self) -> Dict[str, Any]:
        """返回默认评估结果（当GPT-4o不可用时）"""
        return {
            "scene_type": "unknown",
            "subject_category": "unknown",
            "composition_score": 50,
            "background_dynamics": 50,
            "subject_clarity": 50,
            "description": "",
            "tags": [],
            "is_suitable_for_bg_replacement": False,
            "suitability_reason": "GPT-4o评估不可用",
        }
