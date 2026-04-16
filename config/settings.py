"""
全局配置文件
支持通过环境变量覆盖默认配置
"""

import os
from pathlib import Path
from typing import List, Optional

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# 输出目录
OUTPUT_DIR = BASE_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
DATASET_DIR = OUTPUT_DIR / "dataset"
LABELS_DIR = OUTPUT_DIR / "labels"
LOGS_DIR = BASE_DIR / "logs"

# 确保目录存在
for d in [OUTPUT_DIR, FRAMES_DIR, DATASET_DIR, LABELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== 数据库配置 ====================
DB_CONFIG = {
    "type": os.getenv("DB_TYPE", "sqlite"),  # sqlite / mysql / postgresql
    "sqlite_path": os.getenv("DB_SQLITE_PATH", str(BASE_DIR / "videos.db")),
    "mysql": {
        "host": os.getenv("DB_MYSQL_HOST", "localhost"),
        "port": int(os.getenv("DB_MYSQL_PORT", "3306")),
        "user": os.getenv("DB_MYSQL_USER", "root"),
        "password": os.getenv("DB_MYSQL_PASSWORD", ""),
        "database": os.getenv("DB_MYSQL_DATABASE", "video_library"),
    },
    "postgresql": {
        "host": os.getenv("DB_PG_HOST", "localhost"),
        "port": int(os.getenv("DB_PG_PORT", "5432")),
        "user": os.getenv("DB_PG_USER", "postgres"),
        "password": os.getenv("DB_PG_PASSWORD", ""),
        "database": os.getenv("DB_PG_DATABASE", "video_library"),
    },
    # 数据库表名（可按需修改）
    "table_name": os.getenv("DB_TABLE_NAME", "videos"),
    # 视频文件路径字段名
    "file_path_column": os.getenv("DB_FILE_PATH_COLUMN", "file_path"),
    # 视频文件存储的根目录（用于拼接完整路径）
    "video_root_dir": os.getenv("VIDEO_ROOT_DIR", ""),
}

# ==================== 视频筛选配置 ====================
FILTER_CONFIG = {
    # 横屏视频的宽高比阈值（width/height > 此值视为横屏）
    "landscape_ratio_threshold": float(os.getenv("LANDSCAPE_RATIO", "1.5")),
    # 最小分辨率
    "min_width": int(os.getenv("MIN_WIDTH", "1280")),
    "min_height": int(os.getenv("MIN_HEIGHT", "720")),
    # 需要排除的tags关键词（视频包含这些tag将被过滤）
    "exclude_tags": [
        "face", "portrait", "selfie", "close-up", "talking head",
        "interview", "news", "text", "subtitle", "watermark",
        "正面", "人脸", "特写", "采访", "字幕",
    ],
    # 需要包含的tags关键词（至少匹配一个）
    "include_tags": [
        "nature", "landscape", "city", "street", "ocean", "mountain",
        "forest", "sky", "sunset", "rain", "snow", "night",
        "animal", "wildlife", "bird", "dog", "cat",
        "dance", "sport", "action", "motion", "dynamic",
        "自然", "风景", "城市", "街道", "海洋", "山",
        "森林", "天空", "日落", "雨", "雪", "夜景",
        "动物", "鸟", "舞蹈", "运动", "动态",
    ],
    # 标签匹配模式：any（任一匹配）或 all（全部匹配）
    "include_mode": "any",
}

# ==================== 抽帧配置 ====================
FRAME_CONFIG = {
    # 抽帧策略：uniform（均匀） / keyframe（关键帧） / scene（场景变化）
    "strategy": os.getenv("FRAME_STRATEGY", "uniform"),
    # 每秒抽取帧数（uniform模式）
    "fps": float(os.getenv("FRAME_FPS", "1.0")),
    # 最大抽帧数量（防止超长视频）
    "max_frames": int(os.getenv("MAX_FRAMES", "30")),
    # 输出帧图像格式
    "image_format": os.getenv("FRAME_FORMAT", "jpg"),
    # 输出帧图像质量（1-100）
    "image_quality": int(os.getenv("FRAME_QUALITY", "85")),
    # 帧图像尺寸（None表示保持原始尺寸）
    "resize": None,
}

# ==================== 质量分析配置 ====================
QUALITY_CONFIG = {
    # 最低清晰度评分（0-1，低于此值将被过滤）
    "min_sharpness": float(os.getenv("MIN_SHARPNESS", "0.3")),
    # 最低亮度评分（0-1）
    "min_brightness": float(os.getenv("MIN_BRIGHTNESS", "0.1")),
    # 最高亮度评分（过曝检测）
    "max_brightness": float(os.getenv("MAX_BRIGHTNESS", "0.95")),
    # 最低对比度评分
    "min_contrast": float(os.getenv("MIN_CONTRAST", "0.1")),
    # 场景变化检测阈值（0-1，越小越敏感）
    "scene_change_threshold": float(os.getenv("SCENE_CHANGE_THRESHOLD", "0.3")),
}

# ==================== AI模型配置 ====================
AI_CONFIG = {
    # YOLO 模型配置（用于目标检测）
    "yolo": {
        "model_name": os.getenv("YOLO_MODEL", "yolov8x.pt"),
        "confidence_threshold": float(os.getenv("YOLO_CONFIDENCE", "0.5")),
        # 需要检测的类别（COCO类别名）
        "target_classes": [
            "person", "cat", "dog", "bird", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "motorcycle",
            "bicycle", "car", "bus", "truck", "boat", "airplane",
        ],
        # 人物正脸检测：如果检测到 person 且面部面积占比较大，标记为正脸
        "face_detection": True,
    },
    # OpenAI GPT-4o 配置（用于内容理解和质量评估）
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
        # 请求超时（秒）
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "60")),
        # 并发请求数
        "max_concurrent": int(os.getenv("OPENAI_CONCURRENT", "5")),
    },
    # MediaPipe 配置（用于人体姿态估计，辅助判断动作幅度）
    "mediapipe": {
        "enabled": bool(os.getenv("MEDIAPIPE_ENABLED", "true").lower() == "true"),
        "min_detection_confidence": float(os.getenv("MP_DET_CONF", "0.5")),
        "min_tracking_confidence": float(os.getenv("MP_TRACK_CONF", "0.5")),
    },
}

# ==================== 筛选评分配置 ====================
SCORING_CONFIG = {
    # 各维度权重
    "weights": {
        "sharpness": 0.15,        # 清晰度
        "composition": 0.25,      # 构图质量
        "background_dynamics": 0.20,  # 背景动态感
        "subject_clarity": 0.20,  # 主体明确度
        "motion_intensity": 0.10, # 动作幅度
        "diversity": 0.10,        # 多样性（类别覆盖）
    },
    # 最低可用性评分（0-100，低于此值不纳入训练集）
    "min_usability_score": float(os.getenv("MIN_USABILITY_SCORE", "60")),
    # 每个类别最大视频数量（避免类别不平衡）
    "max_per_category": int(os.getenv("MAX_PER_CATEGORY", "100")),
}

# ==================== 存储配置 ====================
STORAGE_CONFIG = {
    # 分类存储的目录结构
    "category_structure": "{scene_type}/{subject_category}",
    # 标签文件格式
    "label_format": "json",
    # 是否同时复制视频文件
    "copy_videos": bool(os.getenv("COPY_VIDEOS", "false").lower() == "true"),
    # 是否生成符号链接（节省空间）
    "symlink_videos": bool(os.getenv("SYMLINK_VIDEOS", "true").lower() == "true"),
    # 数据集清单文件名
    "manifest_filename": "dataset_manifest.json",
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": str(LOGS_DIR / "pipeline.log"),
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}
