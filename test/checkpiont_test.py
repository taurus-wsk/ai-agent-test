# -*- coding: utf-8 -*-
"""
Checkpoint 临时记忆示例（使用 MemorySaver，纯内存存储，调试专用）
特点：程序运行期间保留记忆，退出后丢失，无需数据库，快速调试
"""
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
# 导入 内存版 checkpoint（临时记忆，调试专用）
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import RunnableConfig


# ===================== 1. 定义 AgentState（对话状态载体） =====================
class AgentState(BaseModel):
    """对话状态，包含完整的消息上下文"""
    messages: List[BaseMessage] = Field(default_factory=list, description="所有轮次的对话消息列表")


# ===================== 2. 配置（无需数据库，仅配置模型） =====================
OLLAMA_CONFIG = {
    "model": "qwen3:4b",  # 替换为你已有的 Ollama 模型
    "base_url": "http://localhost:11434",
    "temperature": 0.3,  # 低温度，回复更稳定，方便调试
    "streaming": True
}

# 固定会话 ID（标记唯一对话，用于关联 checkpoint）
THREAD_ID = "debug_temp_memory_001"


# ===================== 3. 核心节点：处理对话，生成回复 =====================
def chat_node(state: AgentState, llm: ChatOllama) -> AgentState:
    """
    对话处理节点：
    1. 流式生成 AI 回复
    2. 更新对话状态（追加 AI 消息）
    """
    full_response = ""
    print("AI 回复：", end="", flush=True)

    # 流式调用 LLM，生成回复（和你之前的代码逻辑对齐）
    for chunk in llm.stream(state.messages):
        # 提取流式内容（健壮性处理）
        try:
            current_content = chunk.content
        except AttributeError:
            current_content = ""

        if current_content:
            print(current_content, end="", flush=True)
            full_response += current_content

    # 追加 AI 消息到上下文，更新状态（内存中）
    new_messages = state.messages.copy()
    if full_response:
        new_messages.append(AIMessage(content=full_response))

    return AgentState(messages=new_messages)


# ===================== 4. 构建并运行对话流程（带临时 checkpoint） =====================
def main():
    # 步骤 1：初始化 LLM
    llm = ChatOllama(**OLLAMA_CONFIG)

    # 步骤 2：初始化 内存版 checkpoint（临时记忆，调试专用，无需数据库）
    temp_checkpointer = MemorySaver()

    # 步骤 3：构建 LangGraph 流程
    graph_builder = StateGraph(AgentState)
    # 添加对话节点（绑定 LLM）
    graph_builder.add_node("chat", lambda state: chat_node(state, llm))
    # 定义流程：START → chat → END
    graph_builder.add_edge(START, "chat")
    graph_builder.add_edge("chat", END)

    # 步骤 4：编译流程（绑定临时 checkpoint）
    compiled_graph = graph_builder.compile(
        checkpointer=temp_checkpointer,  # 绑定内存 checkpoint
        interrupt_before=[],  # 无需中断，直接运行
        interrupt_after=[]
    )

    # 步骤 5：构造会话配置（关联唯一 thread_id）
    session_config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    print("===== Checkpoint 临时记忆调试示例 =====")
    print("说明：程序运行期间保留记忆，退出后丢失；输入 'exit' 退出程序")
    print("=" * 60 + "\n")

    # 步骤 6：交互式对话（多轮测试，验证 checkpoint 记忆）
    while True:
        # 获取用户输入
        user_input = input("用户：").strip()
        if user_input.lower() in ["exit", "退出", "q"]:
            print("\n程序退出，临时记忆已清除")
            break
        if not user_input:
            continue

        # 构建初始状态（追加用户输入消息）
        # 先从 checkpoint 加载历史状态（如果有）
        history_state = compiled_graph.get_state(session_config)
        if history_state:
            print(history_state)
            # 加载历史 messages
            # history_messages = history_state.values["messages"]
        else:
            history_messages = []

        # 追加本次用户输入消息
        # new_messages = history_messages.copy()

        new_messages=[HumanMessage(content=user_input)]
        initial_state = AgentState(messages=new_messages)

        # 运行流程（自动保存状态到内存 checkpoint）
        print("-" * 60)
        compiled_graph.invoke(
            input=initial_state,
            config=session_config
        )
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()