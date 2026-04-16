"""
示例数据库初始化脚本
创建一个包含示例视频数据的SQLite数据库，用于测试和演示
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_sample_database(db_path: str = "videos.db"):
    """创建示例SQLite数据库"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建视频表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            duration REAL,
            fps REAL,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 插入示例数据
    sample_videos = [
        # 横屏视频 - 适合背景替换
        ("videos/nature/ocean_waves_01.mp4", 1920, 1080, 30.5, 30, "nature, ocean, waves, sunset, landscape, dynamic"),
        ("videos/nature/mountain_forest_02.mp4", 1920, 1080, 45.2, 24, "nature, mountain, forest, landscape, sky"),
        ("videos/city/street_night_03.mp4", 2560, 1440, 20.0, 30, "city, street, night, lights, dynamic, urban"),
        ("videos/animal/bird_flying_04.mp4", 1920, 1080, 15.3, 60, "animal, bird, flying, sky, nature, wildlife"),
        ("videos/nature/rain_city_05.mp4", 3840, 2160, 25.7, 30, "nature, rain, city, street, moody, dynamic"),
        ("videos/sport/surfing_06.mp4", 1920, 1080, 35.0, 60, "sport, surfing, ocean, action, dynamic, motion"),
        ("videos/nature/snow_mountain_07.mp4", 2560, 1440, 40.0, 24, "nature, snow, mountain, winter, landscape"),
        ("videos/animal/horse_running_08.mp4", 1920, 1080, 18.5, 60, "animal, horse, running, nature, action, motion"),
        ("videos/city/traffic_timelapse_09.mp4", 3840, 2160, 60.0, 30, "city, traffic, timelapse, night, lights, dynamic"),
        ("videos/nature/aurora_sky_10.mp4", 4096, 2160, 55.0, 24, "nature, aurora, sky, night, landscape, dynamic"),

        # 更多横屏视频
        ("videos/dance/ballet_11.mp4", 1920, 1080, 28.0, 30, "dance, ballet, motion, artistic, dynamic"),
        ("videos/nature/waterfall_12.mp4", 2560, 1440, 32.0, 30, "nature, waterfall, forest, landscape, dynamic"),
        ("videos/city/skyline_sunset_13.mp4", 3840, 2160, 22.0, 30, "city, skyline, sunset, urban, landscape"),
        ("videos/animal/dolphin_jump_14.mp4", 1920, 1080, 12.5, 60, "animal, dolphin, ocean, jumping, action, wildlife"),
        ("videos/nature/desert_dune_15.mp4", 4096, 2160, 48.0, 24, "nature, desert, dune, landscape, wind"),

        # 竖屏视频 - 应被过滤
        ("videos/portrait/selfie_16.mp4", 1080, 1920, 15.0, 30, "face, portrait, selfie, close-up"),
        ("videos/portrait/interview_17.mp4", 1080, 1920, 120.0, 30, "interview, talking head, face, close-up"),

        # 正脸人物视频 - 应被过滤
        ("videos/people/news_anchor_18.mp4", 1920, 1080, 300.0, 30, "news, face, portrait, talking head, text"),
        ("videos/people/vlog_19.mp4", 1920, 1080, 600.0, 30, "vlog, face, selfie, portrait, close-up"),

        # 低分辨率 - 应被过滤
        ("videos/low_res/old_clip_20.mp4", 640, 480, 10.0, 15, "nature, landscape, old, low quality"),

        # 侧面/背影人物 - 适合背景替换
        ("videos/people/walking_back_21.mp4", 1920, 1080, 20.0, 30, "people, walking, street, urban, back view"),
        ("videos/dance/group_dance_22.mp4", 2560, 1440, 35.0, 60, "dance, group, motion, artistic, dynamic, action"),

        # 更多场景覆盖
        ("videos/nature/volcano_23.mp4", 3840, 2160, 42.0, 30, "nature, volcano, fire, smoke, dramatic, dynamic"),
        ("videos/city/subway_24.mp4", 1920, 1080, 25.0, 30, "city, subway, urban, motion, dynamic"),
        ("videos/nature/coral_reef_25.mp4", 3840, 2160, 38.0, 30, "nature, ocean, coral, fish, underwater, wildlife"),
        ("videos/sport/skateboard_26.mp4", 1920, 1080, 16.0, 60, "sport, skateboard, action, urban, motion, dynamic"),
        ("videos/nature/lightning_storm_27.mp4", 2560, 1440, 55.0, 30, "nature, storm, lightning, sky, dramatic, dynamic"),
        ("videos/animal/eagle soaring_28.mp4", 3840, 2160, 30.0, 60, "animal, eagle, bird, flying, sky, wildlife, nature"),
        ("videos/city/festival_night_29.mp4", 1920, 1080, 45.0, 30, "city, festival, night, lights, crowd, dynamic"),
        ("videos/nature/cherry_blossom_30.mp4", 4096, 2160, 50.0, 24, "nature, cherry blossom, spring, garden, landscape"),
    ]

    cursor.executemany(
        """INSERT OR REPLACE INTO videos (file_path, width, height, duration, fps, tags)
           VALUES (?, ?, ?, ?, ?, ?)""",
        sample_videos,
    )

    conn.commit()
    conn.close()

    logger.info(f"示例数据库已创建: {db_path}")
    logger.info(f"  - 共 {len(sample_videos)} 条视频记录")
    logger.info(f"  - 包含横屏/竖屏/正脸/低分辨率等多种场景")
    print(f"\n✅ 示例数据库已创建: {db_path}")
    print(f"   共 {len(sample_videos)} 条视频记录")
    print(f"   包含横屏/竖屏/正脸/低分辨率等多种场景用于测试筛选逻辑")


if __name__ == "__main__":
    create_sample_database()
