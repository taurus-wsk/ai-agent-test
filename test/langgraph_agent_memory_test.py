# -*- coding: utf-8 -*-
"""
LangGraph 调试案例：验证历史记忆自动加载到 state.messages 中
核心：添加关键日志，直观看到记忆加载过程
"""
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.agents import AgentStep
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
import sys

from app.eckert_agent.memory.pg_checkpointer import get_postgres_checkpointer
from app.eckert_agent.model.ollama import OllamaModel


# ===================== 1. 定义 AgentState =====================
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list, description="对话历史（含历史记忆）")
    intermediate_steps: List[AgentStep] = Field(default_factory=list, description="工具调用步骤")


# ===================== 2. 配置（替换为你的 PostgreSQL 信息） =====================
OLLAMA_CONFIG = {
    "model": "qwen3:4b",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "streaming": True,
    "format": "ollama"
}

POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "langgraph_db",
    "user": "postgres",
    "password": "123456"  # 替换为你的密码
}

  # 固定会话 ID，用于加载历史记忆
SYSTEM_PROMPT = "你是一个友好的助手，必须牢牢记住用户的所有历史信息，回答简洁。"


# ===================== 3. 核心 Agent 类（带完整调试日志） =====================
class DebugMemoryAgent:
    def __init__(self):
        # 1. 初始化 PostgresSaver（全局 checkpointer）
        # self.checkpointer = PostgresSaver(
        #     conn_info=POSTGRES_CONFIG,
        #     table_name="agent_checkpoints"
        # )
        self.checkpointer = get_postgres_checkpointer()
        THREAD_ID = "debug_test_001"
        self.session_config = {"configurable": {"thread_id": THREAD_ID}}

        # 2. 初始化模型和子 Agent（不配置局部 checkpointer，避免冗余）
        self.llm =  OllamaModel().get_llm()
        self.base_agent = self._build_base_agent()

        # 3. 构建并编译 LangGraph（仅一次编译，绑定全局 checkpointer）
        self.compiled_graph = self._build_and_compile_graph()

    def _build_base_agent(self):
        """构建子 Agent，不配置局部 checkpointer"""
        return create_agent(
            model=self.llm,
            tools=[],
            system_prompt=SYSTEM_PROMPT.strip(),

            debug=False
        )

    def _agent_node(self, state: AgentState) -> AgentState:
        """
        LangGraph 核心节点：带 3 处关键调试日志，验证 state.messages 包含历史记忆
        """
        print("\n" + "=" * 80)
        print("【调试日志 1：进入节点时的 state.messages】")
        print(f"  消息总数：{len(state.messages)} 条")
        for idx, msg in enumerate(state.messages, 1):
            role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
            print(f"  {idx}. {role}：{msg.content}")
        print("=" * 80 + "\n")

        # 流式调用子 Agent，拼接响应
        full_response_content = ""
        print("AI 回复：", end="", flush=True)
        for chunk in self.base_agent.stream(state):
            try:
                current_content = chunk['model']['messages'][0].content
            except (IndexError, KeyError):
                current_content = ""
            if current_content:
                print(current_content, end="", flush=True)
                full_response_content += current_content

        # 追加 AI 回复到 messages，更新状态
        new_messages = state.messages.copy()
        if full_response_content:
            new_messages.append(AIMessage(content=full_response_content))
        state.messages.append(HumanMessage(content="我自己加入的"))
        for chunk in self.base_agent.stream(state):
            try:
                current_content = chunk['model']['messages'][0].content
            except (IndexError, KeyError):
                current_content = ""
            if current_content:
                print(current_content, end="", flush=True)
                full_response_content += current_content

        # 追加 AI 回复到 messages，更新状态
        new_messages = state.messages.copy()
        if full_response_content:
            new_messages.append(AIMessage(content=full_response_content))

        print("\n" + "=" * 80)
        print("【调试日志 2：节点退出时的 new_messages（含本次回复）】")
        print(f"  消息总数：{len(new_messages)} 条")
        for idx, msg in enumerate(new_messages, 1):
            role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
            print(f"  {idx}. {role}：{msg.content}")
        print("=" * 80 + "\n")

        return AgentState(
            messages=new_messages,
            intermediate_steps=state.intermediate_steps
        )

    def _build_and_compile_graph(self):
        """构建并编译 LangGraph，绑定全局 checkpointer"""
        graph = StateGraph(AgentState)
        graph.add_node("core_agent", self._agent_node)
        graph.add_edge(START, "core_agent")
        graph.add_edge("core_agent", END)

        # 编译时绑定全局 checkpointer（核心：实现记忆加载/持久化）
        return graph.compile(checkpointer=self.checkpointer)

    def chat(self):
        """
        对外对话接口：关键！传入「仅含本次用户输入的轻量状态」，让 LangGraph 自动合并历史记忆
        """
        # 1. 仅封装本次用户输入（不传入历史，让 LangGraph 从 checkpointer 自动加载）
        # current_human_msg = HumanMessage(content='我叫何增辉')
        # 🔴 关键：初始状态仅包含本次用户输入，历史记忆由 LangGraph 自动加载合并


        print("\n" + "-" * 80)

        print("-" * 80)
        # self._debug_print_checkpoint_data()
        # 2. 运行编译后的 graph，自动完成「历史记忆加载→合并→生成回复→持久化最新状态」
        checkpoint = self.checkpointer.get(self.session_config)
        # input_state1 = AgentState(messages=[HumanMessage(content='我叫何增辉1')])
        # chunk=self.compiled_graph.invoke(input_state1, config=self.session_config)
        # chunk["messages"][-1].pretty_print()
        # checkpoint = self.checkpointer.get(self.session_config)
        input_state1 = AgentState(messages=[HumanMessage(content='我叫何增辉2')])
        chunk = self.compiled_graph.invoke(input_state1, config=self.session_config)
        chunk["messages"][-1].pretty_print()
        # 3. 调试：运行完成后，手动从 checkpointer 读取最新状态，验证持久化


    def _debug_print_checkpoint_data(self):
        """调试：手动从 checkpointer 读取最新数据，验证记忆已持久化"""
        try:
            # 适配新旧 LangGraph 版本
            try:
                checkpoint = self.checkpointer.get(self.session_config)
            except AttributeError:
                pass
                # checkpoint_id = (THREAD_ID, None, None)
                # checkpoint = self.checkpointer.load_checkpoint(checkpoint_id)

            if checkpoint:
                history_messages = checkpoint.get("values", {}).get("messages", [])
                print("\n" + "=" * 80)
                print("【调试日志 3：从 checkpointer 读取的最新完整记忆】")
                print(f"  消息总数：{len(history_messages)} 条")
                for idx, msg in enumerate(history_messages, 1):
                    role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
                    print(f"  {idx}. {role}：{msg.content}")
                print("=" * 80 + "\n")
            else:
                print("【调试日志 3】：暂无历史 checkpoint 数据")
        except Exception as e:
            print(f"【调试日志 3 错误】：{str(e)}")


# ===================== 4. 测试主函数 =====================
def main():
    print("===== LangGraph 记忆加载调试案例 =====")
    print("  1. 第一次运行：输入「我叫何增辉」，保存记忆")
    print("  2. 第二次运行：输入「你还记得我叫什么吗」，验证记忆加载")
    print("  3. 输入「exit」退出")
    print("=" * 80 + "\n")

    agent = DebugMemoryAgent()

    agent.chat()


if __name__ == "__main__":
    main()