
import psycopg
from typing import List, Dict
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB


class KnowledgeRetriever:
    """PostgreSQL知识库检索类（MD文档）"""

    def __init__(self):
        self.conn_params = {
            "host": PG_HOST,
            "port": PG_PORT,
            "user": PG_USER,
            "password": PG_PASSWORD,
            "dbname": PG_DB
        }

    def _get_connection(self):
        """获取数据库连接"""
        try:
            return psycopg.connect(**self.conn_params)
        except psycopg.OperationalError as e:
            raise Exception(f"PostgreSQL 知识库连接失败：{str(e)}")

    def add_knowledge(self, title: str, content: str, keywords: str = ""):
        """添加MD文档到知识库"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_base (title, content, keywords)
                    VALUES (%s, %s, %s)
                    """,
                    (title, content, keywords)
                )
            conn.commit()
        print(f"✅ 知识库文档「{title}」添加成功")

    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        return []
        """
        检索知识库（基于关键词匹配+全文检索）
        :param query: 用户问题（检索关键词）
        :param top_k: 返回最相关的top_k条结果
        :return: 检索结果列表（含title/content）
        """
        # PostgreSQL全文检索语法（适配中文）
        search_sql = """
                     SELECT title, \
                            content,
                            ts_rank(to_tsvector('chinese', content), plainto_tsquery('chinese', %s)) AS rank
                     FROM knowledge_base
                     WHERE to_tsvector('chinese', content) @@ plainto_tsquery('chinese' \
                         , %s)
                        OR to_tsvector('chinese' \
                         , title) @@ plainto_tsquery('chinese' \
                         , %s)
                        OR to_tsvector('chinese' \
                         , keywords) @@ plainto_tsquery('chinese' \
                         , %s)
                     ORDER BY rank DESC
                         LIMIT %s \
                     """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    search_sql,
                    (query, query, query, query, top_k)
                )
                results = [dict(row) for row in cur.fetchall()]

        # 过滤掉rank为0的无关结果
        results = [r for r in results if r["rank"] > 0]
        return results

    def format_knowledge(self, query: str) -> str:
        """
        格式化检索结果（转为AI可读取的文本）
        :param query: 用户问题
        :return: 格式化后的知识库内容
        """
        results = self.search_knowledge(query)
        if not results:
            return "【未检索到相关知识库内容】"

        formatted = "【知识库参考内容】\n"
        for i, res in enumerate(results, 1):
            formatted += f"{i}. 标题：{res['title']}\n内容：{res['content']}\n\n"
        return formatted.strip()