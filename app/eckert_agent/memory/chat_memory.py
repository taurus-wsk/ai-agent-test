
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB, MAX_MEMORY_LEN
from langchain_core.chat_history import BaseChatMessageHistory
import psycopg  # 核心导入，替代 psycopg2
from psycopg import OperationalError, ProgrammingError  # 异常类导入路径不变（兼容）
from psycopg.rows import dict_row
class ChatMemory(BaseChatMessageHistory):
    """ChatMemory 对话记忆封装类"""

    def __init__(self):
        self.conn_params = {
            "host": PG_HOST,
            "port": PG_PORT,
            "user": PG_USER,
            "password": PG_PASSWORD,
            "dbname": PG_DB
        }
        self.table_name="chat_memory"
        self.max_memory_len = MAX_MEMORY_LEN

    def _get_connection(self):
        """获取数据库连接（内部方法）"""
        try:
            return psycopg.connect(**self.conn_params)
        except psycopg.OperationalError as e:
            raise Exception(f"ChatMemory 连接失败：{str(e)}")

    def _init_table(self):
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS chat_memory (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL  ,
            role VARCHAR(20) NOT NULL  ,
            content TEXT NOT NULL ,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP 
        );
        
         
        CREATE INDEX IF NOT EXISTS idx_chat_memory_session_id ON chat_memory (session_id);
        """
        with psycopg.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
    def save_message(self, session_id: str, role: str, content: str):
        """保存单条对话消息到数据库"""
        if role not in ["user", "assistant"]:
            raise ValueError("角色只能是 user 或 assistant")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 插入新消息
                cur.execute(
                    """
                    INSERT INTO chat_memory (session_id, role, content)
                    VALUES (%s, %s, %s)
                    """,
                    (session_id, role, content)
                )
                # 清理超出长度的历史
                cur.execute(
                    """
                    DELETE
                    FROM chat_memory
                    WHERE session_id = %s
                      AND id NOT IN (SELECT id
                                     FROM chat_memory
                                     WHERE session_id = %s
                                     ORDER BY created_at DESC
                        LIMIT %s
                        )
                    """,
                    (session_id, session_id, self.max_memory_len)
                )
            conn.commit()

    def clear(self) -> None:
        delete_sql = f"DELETE FROM {self.table_name} WHERE session_id = %s;"
        with psycopg.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(delete_sql, (self.session_id,))
            conn.commit()
    def get_history(self, session_id: str) -> List[Dict]:
        """获取原始对话历史（role+content），仅返回最新20条"""
        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM chat_memory
                    WHERE session_id = %s
                    ORDER BY created_at DESC  -- 1. 先按时间倒序（最新的记录排在前面）
                    """,
                    (session_id,)
                )
                # 3. 反转结果，恢复为「创建时间正序」（早→晚），保证上下文逻辑连贯
                latest_20_records = cur.fetchall()
                return [dict(row) for row in latest_20_records]

    def get_history_as_messages(self, session_id: str) -> List:
        """将历史转为LangChain消息对象（适配ChatPromptTemplate）"""
        raw_history = self.get_history(session_id)
        messages = []
        for msg in raw_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

# 记忆获取函数，供RunnableWithMessageHistory调用
def get_postgres_memory() -> ChatMemory:
    return ChatMemory()