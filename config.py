import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# Ollama 配置
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))

# PostgreSQL 配置
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD","123456")
PG_DB = os.getenv("PG_DB", "postgres")
MAX_MEMORY_LEN = int(os.getenv("MAX_MEMORY_LEN", 10))

# 会话配置
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "default_session")