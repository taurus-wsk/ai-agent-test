from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from app.eckert_agent.memory.chat_memory import ChatMemory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# from langgraph.checkpoint import

from app.eckert_agent.memory.chat_memory import get_postgres_memory
from app.eckert_agent.memory.pg_checkpointer import get_postgres_checkpointer
from app.eckert_agent.skills.test_add import AddToolInput
from app.eckert_agent.tool.agent_tools import get_agent_tools
from app.eckert_agent.model.ollama import OllamaModel
# from app.eckert_agent.graph.chat_graph import AgentState
from langchain_core.exceptions import OutputParserException
import os
from dotenv import load_dotenv
# from typing import TypedDict, List
load_dotenv()
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentStep
# from langgraph.checkpoint.sqlite import SqliteSaver
from app.eckert_agent.skills.test_add import add_numbers_tool,load_add_prompt
# ===================== 1. 定义LangGraph状态结构（无修改） =====================
# 4. 定义你的 Agent 状态类（与你 _build_langgraph() 中的 AgentState 保持一致）

# 合规 Agent 状态类（BaseModel 实现，LangGraph 官方推荐）
class AgentState(BaseModel):
    """
    符合 AgentState 限制的状态类（BaseModel 版）
    包含 Agent 必需的 2 个核心字段：messages + intermediate_steps
    """
    # 1. 对话消息列表（核心字段，存储用户/AI/系统消息）
    # 类型：List[BaseMessage]，默认空列表，支持 LangChain 所有消息对象
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="存储对话历史消息，包含 HumanMessage、AIMessage 等"
    )

    # 2. 工具调用中间步骤（核心字段，存储 Agent 调用工具的历史记录）
    # 类型：List[AgentStep]，默认空列表，记录工具调用的输入/输出
    intermediate_steps: List[AgentStep] = Field(
        default_factory=list,
        description="存储 Agent 工具调用的中间步骤，包含工具名称、参数、结果"
    )

    # 3. （可选）自定义扩展字段（按需添加，不违反 AgentState 限制）
    user_id: str = Field(default="default_user", description="用户唯一标识")
    session_id: str = Field(default="default_session", description="会话唯一标识")

# class ToolMetaSchema(BaseModel):
#     """工具元数据 Schema（结构化描述工具信息）"""
#     tool_name: str = Field(..., description="工具唯一标识名称")
#     tool_description: str = Field(..., description="工具功能描述")
#     tool_input_schema: AddToolInput = Field(..., description="工具输入参数格式要求")

class ContextSchema(BaseModel):
    """
    会话上下文配置 Schema（context_schema）
    适配 type[ContextT]：传入类本身，而非实例
    包含：从 md 读取的工具规则（tool_context）、可用工具列表（available_tools）
    """
    # 修正：默认值改为空列表（符合 List[str] 类型），描述明确为「从 md 加载的工具调用规则」
    tool_context: List[str] = Field(default_factory=list, description="从 md 文件加载的工具调用规则，列表格式")
    # available_tools: List[ToolMetaSchema] = Field(default_factory=list, description="Agent 可使用的工具列表")

# 工具调用中间步骤（Agent 必备）
# ===================== 2. 核心整合类（Ollama自定义模型专属适配） =====================
class OllamaAgent:
    def __init__(self):
        # 初始化所有组件：LLM（Ollama自定义）+工具+Checkpointer
        self.llm = OllamaModel().get_llm() # 核心：Ollama自定义模型初始化
        # self.tools = get_agent_tools()
        self.prompt = self._build_ollama_prompt()  # 核心：Ollama轻量化prompt
        # self.checkpointer = get_postgres_checkpointer()

        self.session_config =RunnableConfig(configurable={"thread_id": "debug_test_001"})
        # self.session_config = {"configurable": {"thread_id": "debug_test_001"}}
        # self.checkpointer = MemorySaver()
        self.postgres_memory = ChatMemory()
        # self.graph_config = {
        #     "configurable": {"thread_id": "agent_session_001"}
        # }

        self.context_schema=ContextSchema(
            tool_context=[load_add_prompt()]
        )
        self.base_agent = create_agent(
            model=self.llm,
            system_prompt=self.prompt,
            tools=[add_numbers_tool],
            context_schema=self.context_schema,
            debug=True
        )

        # 构建LangGraph流程（保留核心，可扩展多节点）
        self.graph = self._build_langgraph()



    def _build_ollama_prompt(self) -> SystemMessage:
        """
        核心：构建Ollama专属Prompt（轻量化，适配本地模型推理习惯）
        简化官方模板，去除冗余描述，避免Ollama模型过载
        """
        system_prompt = """
                        你是编程资深咨询助手，保持你的角色：
                        1. 回答简洁专业，可以编写代码，有发散性思维。
                        """

        # Ollama适配：仅保留3个核心占位符，顺序不可变
        return SystemMessage(content=system_prompt)

    from langchain_core.messages import AIMessage, BaseMessage
    from typing import List

    def agent_node(self, state: AgentState) -> AgentState:
        """
        包装 create_agent() 生成的子图，在节点内部实现循环对话
        直到用户输入「exit/退出」，才终止循环并返回最终状态
        """
        # 步骤 1：初始化当前状态（继承节点传入的初始状态，保留外部上下文）
        current_state = state
        print("=== 进入内部循环对话（输入 'exit' 或 '退出' 结束对话）===")
        print("用户：", end="", flush=True)

        # 步骤 2：搭建对话循环（核心：持续交互直到触发退出条件）
        while True:
            # 子步骤 2.1：接收用户输入（循环内的新输入）
            user_input = input().strip()

            # 子步骤 2.2：判断退出条件，触发循环终止
            if user_input.lower() in ["exit", "退出"]:
                print("=== 内部循环对话结束 ===")
                break

            # 子步骤 2.3：封装用户输入为 HumanMessage，追加到当前状态的 messages 中
            new_human_message = HumanMessage(content=user_input)
            updated_messages = current_state.messages.copy()
            updated_messages.append(new_human_message)
            current_state = AgentState(
                messages=updated_messages,
                intermediate_steps=current_state.intermediate_steps
            )

            # 子步骤 2.4：流式调用子图，获取 AI 响应并拼接完整内容
            full_response_content = ""
            final_ai_message = None

            # checkpoint = self.checkpointer.get(self.session_config)
            # print("checkpoint: "+checkpoint, end="", flush=True)

            for chunk in self.base_agent.stream(current_state):
                # 健壮提取 chunk 中的有效内容
                try:
                    current_content = chunk['model']['messages'][0].content
                except (IndexError, KeyError):
                    current_content = ""

                if current_content:
                    print("AI: "+current_content, end="", flush=True)
                    full_response_content += current_content

            # 子步骤 2.5：封装 AI 响应为 AIMessage，更新当前状态（关键：维护上下文）
            if full_response_content:
                final_ai_message = AIMessage(content=full_response_content)
                # 追加 AI 回复到 messages，更新当前状态（为下一轮对话做准备）
                new_messages = current_state.messages.copy()
                new_messages.append(final_ai_message)
                current_state = AgentState(
                    messages=new_messages
                )

            self.postgres_memory.save_message(state.session_id, "user", user_input)
            self.postgres_memory.save_message(state.session_id, "assistant", full_response_content)
            # 子步骤 2.6：准备下一轮输入，保持交互格式整洁
            print("\n")
            print("用户：", end="", flush=True)

        # 步骤 3：循环终止，返回最终更新后的状态（包含完整循环对话历史）
        return AgentState(
            messages=current_state.messages,
            intermediate_steps=current_state.intermediate_steps
        )

    def _build_langgraph(self) -> CompiledStateGraph:
        """构建 LangGraph 流程（保留你的骨架，嵌入 create_agent() 生成的 Agent 核心）"""
        # 1. 保留你的手动状态图初始化
        graph = StateGraph(AgentState)

        # 2. 保留你的核心节点（现在节点内部调用 create_agent() 生成的 Agent 子图）
        graph.add_node("ollama_agent", self.agent_node)

        # 3. 保留你的流程边定义（可扩展多节点，如添加 RAG 节点、校验节点）
        graph.add_edge(START, "ollama_agent")
        graph.add_edge("ollama_agent", END)

        # 4. （可选）若需扩展多节点，可继续添加，例如：
        # graph.add_node("rag_node", self._rag_node)
        # graph.add_edge("ollama_agent", "rag_node")
        # graph.add_edge("rag_node", END)

        compiled_graph = graph.compile()
        return compiled_graph

    # 5. 编译完整图（对外提供入口）
    # def build_compiled_graph(self) -> CompiledStateGraph:
    #     """编译完整的 LangGraph 流程（手动骨架 + Agent 核心）"""
    #     graph = self._build_langgraph()
    #     return graph.compile()

    def run(self) -> str:
        """
        对外统一接口：运行Ollama Agent+LangGraph+自动记忆

        :return: Agent最终回复
        """
        initial_inputs = AgentState(messages=[])
        # graph_config = {
        #     "configurable": {"thread_id": "agent_session_001"}
        # }

        # 执行LangGraph流程，自动加载记忆+流程状态
        print("AI 回复：", end="\n", flush=True)
        res = self.graph.get_state(self.session_config)
        final_state = self.graph.invoke(input=initial_inputs, config=self.session_config)
        # final_response = final_state["messages"][-1].content
        res=self.graph.get_state(self.session_config)
        print(final_state, end="\n")
        return final_state

# ===================== 测试用例（Ollama自定义模型专属） =====================
if __name__ == "__main__":
    # 一键初始化：Ollama Agent+LangGraph+PostgreSQL记忆+Checkpointer
    agent = OllamaAgent()
    agent.run()
