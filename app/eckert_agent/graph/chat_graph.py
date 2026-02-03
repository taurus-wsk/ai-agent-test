from anyio.lowlevel import checkpoint
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, List

from app.eckert_agent.agent.code_agent import OllamaAgent,AgentState
from app.eckert_agent.agent.react_agent import ReActAgent
from app.eckert_agent.memory.knowledge_retriever import KnowledgeRetriever
from app.eckert_agent.model.ollama import OllamaModel
from app.eckert_agent.memory.chat_memory import ChatMemory
from app.eckert_agent.prompts.PromptTemplateManager import PromptTemplateManager
from app.eckert_agent.memory.pg_checkpointer import get_postgres_checkpointer

# 定义LangGraph状态结构（独立在模块内）
# 必要导入
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentStep
#
#
# # 合规 Agent 状态类（BaseModel 实现，LangGraph 官方推荐）
# class AgentState(BaseModel):
#     """
#     符合 AgentState 限制的状态类（BaseModel 版）
#     包含 Agent 必需的 2 个核心字段：messages + intermediate_steps
#     """
#     # 1. 对话消息列表（核心字段，存储用户/AI/系统消息）
#     # 类型：List[BaseMessage]，默认空列表，支持 LangChain 所有消息对象
#     messages: List[BaseMessage] = Field(
#         default_factory=list,
#         description="存储对话历史消息，包含 HumanMessage、AIMessage 等"
#     )
#
#     # 2. 工具调用中间步骤（核心字段，存储 Agent 调用工具的历史记录）
#     # 类型：List[AgentStep]，默认空列表，记录工具调用的输入/输出
#     intermediate_steps: List[AgentStep] = Field(
#         default_factory=list,
#         description="存储 Agent 工具调用的中间步骤，包含工具名称、参数、结果"
#     )
#
#     # 3. （可选）自定义扩展字段（按需添加，不违反 AgentState 限制）
#     user_id: str = Field(default="default_user", description="用户唯一标识")
#     session_id: str = Field(default="default_session", description="会话唯一标识")

class ChatGraph:
    """LangGraph 对话图封装类"""
    def __init__(self):
        """
                初始化：支持传入默认角色设定
                :param default_role_prompt: 默认角色设定（100字以内），不传则用基础设定
        """
        # self.ollama_model = OllamaModel()
        self.postgres_memory = ChatMemory()
        self.knowledge_retriever = KnowledgeRetriever()
        # 默认角色设定（兜底）
        self.default_role_prompt =  """你是友好的AI助手，用简洁的中文回复，优先基于知识库内容，结合对话历史回答问题。"""

        # 初始化提示词管理器（传入默认角色）
        self.prompt_manager = PromptTemplateManager(
            default_role_prompt=self.default_role_prompt,
            # 可自定义默认规则，也可用内置默认
            # default_reply_rules=["规则1", "规则2"]
        )
        # 初始化独立ReActAgent
        self.agent = OllamaAgent()
        self.compiled_graph = self._build_graph()

    def _load_memory_node(self, state: AgentState) -> AgentState:
        """节点1：加载数据库记忆"""
        res= self.postgres_memory.get_history_as_messages(state.session_id)
        state.messages.extend(res)
        return state

    def _search_knowledge_node(self, state: AgentState) -> AgentState:
        """新增节点2：检索知识库"""
        # user_input = state.messages[0].content
        # # 检索并格式化知识库内容
        # state["knowledge_context"] = self.knowledge_retriever.format_knowledge(user_input)
        # print(f"\n{state['knowledge_context']}\n")  # 可选：打印检索结果，便于调试
        return state

    def _chat_node(self, state: AgentState) -> AgentState:
        """节点3：结合知识库+对话记忆+角色设定生成回复"""



        # 1. 构建动态提示词模板（传入单轮角色设定）
        prompt_template = self.prompt_manager.build_chat_prompt_template(
            role_prompt=state.get("role_prompt"),
            # 可选：传入临时规则（覆盖默认）
            # custom_rules=["临时规则1", "临时规则2"]
        )

        # 2. 格式化提示词（填充参数）
        formatted_prompt = self.prompt_manager.format_prompt(
            prompt_template=prompt_template,
            chat_history=state["chat_history"],
            user_input=state["user_input"],
            knowledge_context=state["knowledge_context"]
        )

        full_response = ""
        print("AI: ", end="", flush=True)
        for chunk in self.ollama_model.stream(formatted_prompt):
            full_response += chunk
            print(chunk, end="", flush=True)
        print("\n")

        self.postgres_memory.save_message(state["session_id"], "user", state["user_input"])
        self.postgres_memory.save_message(state["session_id"], "assistant", full_response)
        state["assistant_response"] = full_response.strip()
        return state

    def _build_graph(self):
        """构建并编译LangGraph（内部方法）"""
        graph = StateGraph(AgentState)
        # 添加节点
        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("chat", self.agent.agent_node)
        graph.add_node("search_knowledge", self._search_knowledge_node)
        # 定义流程
        graph.add_edge(START, "load_memory")
        # graph.add_edge("load_memory", "chat")
        graph.add_edge("load_memory", "search_knowledge")
        graph.add_edge("search_knowledge", "chat")
        graph.add_edge("chat", END)
        # 编译图
        # memory=get_postgres_checkpointer()
        compiled_graph = graph.compile()
        return compiled_graph

    def run(self) -> AgentState:
        """对外暴露的运行方法"""
        initial_state = AgentState(messages=[])
        initial_state.session_id="eckert_001"
        # session_config = RunnableConfig(configurable={"thread_id": "eckert_001"})
        return self.compiled_graph.invoke(initial_state)

    def add_knowledge_doc(self, title: str, content: str, keywords: str = ""):
        self.knowledge_retriever.add_knowledge(title, content, keywords)