from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TEMPERATURE


class OllamaModel:
    """Ollama 模型封装类（单例模式）"""
    # 类属性：存储全局唯一实例
    _instance = None
    _llm = None

    def __new__(cls):
        """控制实例创建，确保全局唯一"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 仅在第一次实例化时初始化 llm
            cls._llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=OLLAMA_TEMPERATURE,

            )
        return cls._instance

    # 提供 getter 方法，暴露 llm 实例
    def get_llm(self):
        return self._llm