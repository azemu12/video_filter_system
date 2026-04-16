# 视频数据自动筛选与标签生成系统

用于提高背景替换训练数据质量的自动化系统。针对大规模视频数据，自动完成数据库筛选、视频抽帧、质量分析、AI内容理解、自动筛选、标签生成和分类存储的完整流程。

## 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  数据库筛选  │───▶│  视频抽帧    │───▶│  质量分析    │
│ (tags/宽高) │    │ (关键帧提取) │    │ (清晰度/亮度)│
└─────────────┘    └──────────────┘    └──────┬───────┘
                                             │
                    ┌──────────────┐    ┌─────▼───────┐
                    │  标签生成    │◀───│  AI内容理解  │
                    │  (JSON)     │    │ (YOLO+GPT)  │
                    └──────┬───────┘    └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  分类存储    │
                    │  (数据集输出)│
                    └──────────────┘
```

## 项目结构

```
video_filter_system/
├── main.py                          # CLI入口
├── config/
│   └── settings.py                  # 全局配置（支持环境变量覆盖）
├── src/
│   ├── pipeline.py                  # 主流程管道（串联所有模块）
│   ├── db/
│   │   ├── video_db.py              # 数据库筛选模块
│   │   └── sample_db.py             # 示例数据库初始化
│   ├── frame_extractor/
│   │   └── extractor.py             # 视频抽帧 + 质量分析
│   ├── ai_analysis/
│   │   └── analyzer.py              # AI内容理解（YOLO + MediaPipe + GPT-4o）
│   ├── filter/
│   │   └── scorer.py                # 综合评分 + 标签生成
│   ├── storage/
│   │   └── dataset_storage.py       # 分类存储 + 数据集输出
│   └── utils/
│       └── helpers.py               # 日志、进度条等工具
├── output/                          # 输出目录（自动创建）
│   ├── frames/                      # 抽取的帧图像
│   ├── labels/                      # 生成的JSON标签
│   └── dataset/                     # 最终训练数据集
├── requirements.txt                 # Python依赖
├── .env.example                     # 环境变量模板
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd video_filter_system
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key 和数据库配置
```

### 3. 初始化示例数据库（用于测试）

```bash
python main.py setup-db
```

### 4. 运行完整管道

```bash
# 处理所有视频
python main.py run

# 限制处理数量
python main.py run --limit 50

# 禁用GPT-4o（仅使用YOLO检测）
python main.py run --no-gpt4o

# 强制重新处理（忽略已处理的视频）
python main.py run --force

# 输出处理报告
python main.py run --output-report report.json
```

## 核心模块说明

### 1. 数据库筛选 (`src/db/`)

- **支持数据库**: SQLite / MySQL / PostgreSQL
- **筛选条件**:
  - 横屏视频（宽高比 > 1.5）
  - 最小分辨率 1280x720
  - Tags关键词过滤（排除正脸/特写等，包含风景/动态等）

### 2. 视频抽帧 (`src/frame_extractor/`)

- **抽帧策略**:
  - `uniform`: 均匀抽帧（按FPS）
  - `keyframe`: 基于帧间差异的关键帧提取
  - `scene`: 基于颜色直方图的场景变化检测
- **质量分析**: 清晰度（拉普拉斯算子）、亮度、对比度

### 3. AI内容理解 (`src/ai_analysis/`)

采用**混合模型方案**:

| 模型 | 用途 | 说明 |
|------|------|------|
| YOLOv8 | 目标检测 | 检测人物/动物/车辆等，判断正面人脸 |
| MediaPipe | 姿态估计 | 分析人物动作幅度 |
| GPT-4o | 内容评估 | 构图评分、场景理解、背景动态感评估 |

### 4. 综合评分 (`src/filter/`)

六维度加权评分:

| 维度 | 权重 | 说明 |
|------|------|------|
| 清晰度 | 15% | 拉普拉斯方差 |
| 构图质量 | 25% | GPT-4o评估（三分法/引导线/层次感） |
| 背景动态感 | 20% | GPT-4o评估（运动模糊/光影变化） |
| 主体明确度 | 20% | GPT-4o评估（主体清晰度/背景分离度） |
| 动作幅度 | 10% | MediaPipe姿态伸展范围 |
| 多样性 | 10% | 检测到的类别数量 |

**一票否决规则**:
- 检测到正面人脸 → 直接排除
- 质量通过率 < 50% → 直接排除
- 综合评分 < 60 → 排除

### 5. 标签生成 (`src/filter/`)

输出的JSON标签结构:

```json
{
  "version": "1.0",
  "generated_at": "2026-04-16T10:00:00",
  "video": {
    "id": "1",
    "file_path": "videos/nature/ocean_waves.mp4",
    "width": 1920,
    "height": 1080,
    "tags": "nature, ocean, waves",
    "duration": 30.5
  },
  "scene": {
    "type": "自然风景",
    "description": "海浪拍打礁石，夕阳西下",
    "ai_tags": ["ocean", "sunset", "waves", "dramatic"]
  },
  "subject": {
    "category": "无明确主体",
    "detected_objects": {},
    "has_person": false,
    "has_frontal_face": false
  },
  "quality": {
    "sharpness": 78.5,
    "composition": 85,
    "background_dynamics": 72,
    "subject_clarity": 90,
    "motion_intensity": 65
  },
  "scoring": {
    "total_score": 78.3,
    "is_usable": true,
    "rejection_reasons": []
  },
  "frames": {
    "total_extracted": 30,
    "best_frame_path": "output/frames/1/1_frame_0005.jpg",
    "best_frame_timestamp": 5.0
  }
}
```

### 6. 分类存储 (`src/storage/`)

- 按 `场景类型/主体类别` 组织目录结构
- 支持复制文件或创建符号链接
- 自动生成数据集清单 `dataset_manifest.json`
- 每个类别有数量上限，避免类别不平衡

## 配置说明

所有配置均支持通过环境变量覆盖，详见 `.env.example`。

关键配置项:

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `DB_TYPE` | sqlite | 数据库类型 |
| `LANDSCAPE_RATIO` | 1.5 | 横屏宽高比阈值 |
| `FRAME_STRATEGY` | uniform | 抽帧策略 |
| `YOLO_MODEL` | yolov8x.pt | YOLO模型 |
| `MIN_USABILITY_SCORE` | 60 | 最低可用性评分 |
| `OPENAI_MODEL` | gpt-4o | GPT模型 |

## 数据库表结构

系统需要视频数据库包含以下字段:

```sql
CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,     -- 视频文件路径
    width INTEGER NOT NULL,      -- 视频宽度
    height INTEGER NOT NULL,     -- 视频高度
    duration REAL,               -- 视频时长（秒）
    fps REAL,                    -- 帧率
    tags TEXT                    -- 标签（逗号分隔）
);
```

如果你的数据库表结构不同，请修改 `config/settings.py` 中的 `DB_CONFIG`。

## 扩展指南

- **添加新的筛选条件**: 修改 `src/db/video_db.py` 中的 `_build_tag_filter` 方法
- **自定义评分权重**: 修改 `config/settings.py` 中的 `SCORING_CONFIG.weights`
- **添加新的AI模型**: 在 `src/ai_analysis/analyzer.py` 中添加新的分析器类
- **自定义标签格式**: 修改 `src/filter/scorer.py` 中的 `LabelGenerator.generate_label` 方法
