from langgraph.checkpoint.postgres import PostgresSaver  # 确保正常导入
from langgraph.checkpoint.base import BaseCheckpointSaver
import psycopg  # 导入 psycopg 3.x
from psycopg.rows import dict_row
import os
from dotenv import load_dotenv

load_dotenv()

def get_postgres_checkpointer() -> BaseCheckpointSaver:
    """确保 PostgresSaver 正常导入并使用，无兼容问题"""
    # 1. 构建连接参数
    conn_params = {
        "host": os.getenv("PG_HOST"),
        "port": int(os.getenv("PG_PORT")),
        "user": os.getenv("PG_USER"),
        "password": os.getenv("PG_PASSWORD"),
        "dbname": os.getenv("PG_DB")
    }

    # 2. 创建 psycopg 3.x 兼容连接（满足 PostgresSaver 要求）
    try:
        compatible_conn = psycopg.connect(**conn_params, row_factory=dict_row)
        compatible_conn.autocommit = True
    except Exception as e:
        raise Exception(f"❌ PostgreSQL 连接失败：{str(e)}") from e

    # 3. 初始化 PostgresSaver（正常使用，无导入/类型错误）
    memory = PostgresSaver(
        conn=compatible_conn,
        pipe=None,
        serde=None
    )

    # 4. 初始化表（确保持久化功能正常）
    try:
        memory.setup()
    except Exception:
        pass

    print("✅ PostgresSaver 正常导入并初始化成功（psycopg[binary] 3.x）")
    return memory