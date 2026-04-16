"""
数据库筛选模块
根据tags、width、height等字段从数据库中筛选符合条件的视频
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import DB_CONFIG, FILTER_CONFIG

logger = logging.getLogger(__name__)


class VideoDatabase:
    """视频数据库查询接口"""

    def __init__(self, db_config: Optional[Dict] = None):
        self.config = db_config or DB_CONFIG
        self.conn = None
        self._connect()

    def _connect(self):
        """建立数据库连接"""
        db_type = self.config["type"]

        if db_type == "sqlite":
            import sqlite3
            db_path = self.config["sqlite_path"]
            logger.info(f"连接 SQLite 数据库: {db_path}")
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row

        elif db_type == "mysql":
            try:
                import pymysql
            except ImportError:
                raise ImportError("请安装 pymysql: pip install pymysql")
            cfg = self.config["mysql"]
            logger.info(f"连接 MySQL 数据库: {cfg['host']}:{cfg['port']}/{cfg['database']}")
            self.conn = pymysql.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"],
                cursorclass=pymysql.cursors.DictCursor,
            )

        elif db_type == "postgresql":
            try:
                import psycopg2
            except ImportError:
                raise ImportError("请安装 psycopg2: pip install psycopg2-binary")
            cfg = self.config["postgresql"]
            logger.info(f"连接 PostgreSQL 数据库: {cfg['host']}:{cfg['port']}/{cfg['database']}")
            self.conn = psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                dbname=cfg["database"],
            )

        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

    def _build_landscape_filter(self) -> str:
        """构建横屏视频过滤条件"""
        ratio = FILTER_CONFIG["landscape_ratio_threshold"]
        min_w = FILTER_CONFIG["min_width"]
        min_h = FILTER_CONFIG["min_height"]
        return f"(width >= {min_w} AND height >= {min_h} AND CAST(width AS FLOAT) / height > {ratio})"

    def _build_tag_filter(self) -> tuple:
        """构建标签过滤条件"""
        exclude_tags = FILTER_CONFIG["exclude_tags"]
        include_tags = FILTER_CONFIG["include_tags"]
        mode = FILTER_CONFIG["include_mode"]

        # 排除条件：tags中不包含任何排除关键词
        exclude_conditions = []
        for tag in exclude_tags:
            exclude_conditions.append(f"tags NOT LIKE '%{tag}%'")
        exclude_clause = " AND ".join(exclude_conditions)

        # 包含条件：tags中至少包含一个包含关键词
        include_conditions = []
        for tag in include_tags:
            include_conditions.append(f"tags LIKE '%{tag}%'")
        if mode == "any":
            include_clause = f"({' OR '.join(include_conditions)})"
        else:
            include_clause = f"({' AND '.join(include_conditions)})"

        return include_clause, exclude_clause

    def filter_videos(
        self,
        extra_where: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        筛选符合条件的视频

        Args:
            extra_where: 额外的WHERE条件
            limit: 返回结果数量限制
            offset: 结果偏移量

        Returns:
            符合条件的视频记录列表
        """
        table = self.config["table_name"]
        landscape_filter = self._build_landscape_filter()
        include_clause, exclude_clause = self._build_tag_filter()

        where_parts = [
            landscape_filter,
            include_clause,
            exclude_clause,
        ]
        if extra_where:
            where_parts.append(f"({extra_where})")

        where_sql = " AND ".join(where_parts)

        sql = f"""
            SELECT * FROM {table}
            WHERE {where_sql}
            ORDER BY width DESC
        """
        if limit:
            sql += f" LIMIT {limit} OFFSET {offset}"

        logger.info(f"执行筛选查询 (limit={limit}, offset={offset})")
        logger.debug(f"SQL: {sql}")

        cursor = self.conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        # 统一转换为字典格式
        if hasattr(rows[0], "keys") if rows else False:
            results = [dict(row) for row in rows]
        else:
            # SQLite Row 对象
            results = [dict(row) for row in rows]

        logger.info(f"筛选到 {len(results)} 个符合条件的视频")
        return results

    def get_video_count(self) -> int:
        """获取符合条件的视频总数"""
        table = self.config["table_name"]
        landscape_filter = self._build_landscape_filter()
        include_clause, exclude_clause = self._build_tag_filter()

        where_sql = f"{landscape_filter} AND {include_clause} AND {exclude_clause}"

        sql = f"SELECT COUNT(*) as cnt FROM {table} WHERE {where_sql}"

        cursor = self.conn.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()

        if hasattr(row, "keys"):
            return dict(row).get("cnt", 0)
        elif isinstance(row, dict):
            return row.get("cnt", 0)
        else:
            return row[0]

    def get_full_path(self, record: Dict[str, Any]) -> str:
        """获取视频文件的完整路径"""
        file_path = record.get(self.config["file_path_column"], "")
        root_dir = self.config.get("video_root_dir", "")
        if root_dir and not Path(file_path).is_absolute():
            return str(Path(root_dir) / file_path)
        return file_path

    def get_all_tags(self) -> List[str]:
        """获取数据库中所有不重复的tags（用于分析）"""
        table = self.config["table_name"]
        sql = f"SELECT DISTINCT tags FROM {table} WHERE tags IS NOT NULL AND tags != ''"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        all_tags = set()
        for row in rows:
            tags_str = dict(row).get("tags", "") if hasattr(row, "keys") else (row[0] if row else "")
            for tag in str(tags_str).replace(",", " ").split():
                all_tags.add(tag.strip().lower())

        return sorted(all_tags)
