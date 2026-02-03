import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载环境变量
load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 2. 初始化 ChatOllama（保留你的原始配置，streaming=True）
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,
    timeout=30.0,
    streaming=True,
      # 保留Ollama原生格式，保证模型响应正常
)

# 3. 定义 ChatPromptTemplate（核心：封装系统提示词+用户输入占位符）
# 模板结构：System Message（固定） + Human Message（动态占位符）
# chat_template = ChatPromptTemplate.from_messages([
#     # 固定系统提示词：约束模型行为，可复用
#     ("system", "你是一个数学计算助手，严格遵守：1. 只返回计算结果的数字；2. 不寒暄、不解释、无多余文字；3. 精准计算，不出错。"),
#     # 动态用户输入：用 {user_input} 作为占位符，后续填充具体问题
#     ("human", "{user_input}")
# ])

# ==================== 方式1：单次对话（模板填充+模型调用）====================

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个小说家。"),
    ("human", "{user_input}")
])
# ==================== 方式2：多轮对话（模板复用+内存记忆）====================
# 4. 核心：终端持续交互式对话（流式输出）
def receive_stream_output():
    # 4.1 填充模板，生成合法输入
    filled_messages = chat_template.format_messages(user_input="你好")

    # 4.2 调用 stream()，获取迭代器（Iterator[AIMessageChunk]）
    stream_iterator = llm.stream(filled_messages)

    # 4.3 迭代迭代器，提取有效内容（解决空白回复的关键）
    print("🤖 AI 响应：", end="", flush=True)  # 保持打字机效果，不换行
    full_response = ""  # 可选，用于拼接完整响应结果

    for chunk in stream_iterator:
        # 关键细节 1：提取 chunk.content 字段（这才是有效响应文本）
        valid_content = chunk.content

        # 关键细节 2：过滤空内容块（部分模型会返回空块，避免无效打印）
        if not valid_content:
            continue

        # 关键细节 3：打印有效内容，保持打字机效果
        print(valid_content, end="", flush=True)

        # 可选：拼接完整响应，用于后续存储/复用
        full_response += valid_content

    print(f"\n\n✅ 完整响应结果（可存储）：{full_response}")
if __name__ == "__main__":
    receive_stream_output()