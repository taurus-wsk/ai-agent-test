from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.tools import BaseTool
from typing import List, Dict, Optional, Callable
from pydantic import BaseModel
import json

class ReActAgentConfig(BaseModel):
    """ReActAgent配置类（参数解耦）"""
    default_role_prompt: str = "你是友好的AI助手，按ReAct规则回答问题。"
    enforce_tool_use: bool = True  # 强制工具调用
    handle_parsing_errors: bool = True  # 容错处理
    max_iterations: int = 5  # 最大工具调用次数

class ReActAgent:
    """
    独立ReActAgent类（适配MCP协议）
    无Graph依赖，可单独调用或接入任意LangGraph
    """
    def __init__(
        self,
        llm: Runnable,  # 任意LLM（如Ollama/OpenAI）
        tools: List[BaseTool],
        prompt_manager: object,  # 提示词管理器
        config: Optional[ReActAgentConfig] = None
    ):
        """
        初始化独立ReActAgent
        :param llm: LLM模型（需实现stream/invoke）
        :param tools: 工具列表
        :param prompt_manager: 提示词管理器实例
        :param config: Agent配置
        """
        self.llm = llm
        self.tools = tools
        self.prompt_manager = prompt_manager
        self.config = config or ReActAgentConfig()
        # 工具映射（快速查找）
        self.tool_map = {tool.name: tool for tool in tools}
        # 构建ReAct提示词模板
        self.prompt_template = self._build_prompt_template()
        # 4. 创建ReAct智能体（LangGraph原生）
        self.agent = create_react_agent(
            llm=self.llm,
            tools=[],  # 无需记忆工具，checkpointer自动加载状态
            prompt=self.prompt,
            enforce_tool_use=False,  # 无需强制工具调用（checkpointer已管理记忆）
            handle_parsing_errors=True
        )

        # 5. 编译Graph并绑定Checkpointer（核心：开启状态持久化）
        self.compiled_agent = self.agent.compile(
            checkpoint=CheckpointConfig(
                checkpointer=self.checkpointer,
                id_key="session_id"  # 用session_id作为检查点ID（区分不同会话）
            )
        )

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """构建ReAct提示词模板（适配MCP）"""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""【ReAct执行规则】：
1. 你拥有以下工具：{tool_descriptions}
2. 执行步骤：
   a. 思考：分析用户问题，判断是否需要调用工具
   b. 行动：如需调用，指定工具名称（必须是已提供的工具）
   c. 行动输入：工具入参（JSON格式，需匹配工具入参要求）
   d. 观察：工具返回结果
   e. 最终答案：基于工具结果回答用户问题
3. 输出格式（严格遵守）：
   思考：[你的思考]
   行动：[工具名称/None]
   行动输入：[JSON字符串/None]
   观察：[工具返回结果/None]
   最终答案：[给用户的回复]"""),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="{user_input}")
        ])

    def _format_tool_descriptions(self) -> str:
        """格式化工具描述（供提示词使用）"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}：{tool.description}")
        return "\n".join(descriptions)

    def _parse_agent_output(self, output: str) -> Dict:
        """解析Agent输出（提取思考/行动/输入）"""
        parsed = {
            "thought": "",
            "action": None,
            "action_input": None,
            "observation": None,
            "final_answer": None
        }
        lines = output.strip().split("\n")
        for line in lines:
            if line.startswith("思考："):
                parsed["thought"] = line[3:].strip()
            elif line.startswith("行动："):
                action = line[3:].strip()
                parsed["action"] = action if action != "None" else None
            elif line.startswith("行动输入："):
                input_str = line[5:].strip()
                parsed["action_input"] = json.loads(input_str) if input_str != "None" else None
            elif line.startswith("观察："):
                parsed["observation"] = line[3:].strip()
            elif line.startswith("最终答案："):
                parsed["final_answer"] = line[5:].strip()
        return parsed

    def _run_tool(self, tool_name: str, tool_input: Dict) -> str:
        """执行工具调用"""
        if tool_name not in self.tool_map:
            return f"错误：工具{tool_name}不存在"
        try:
            tool = self.tool_map[tool_name]
            # 适配工具入参（支持单参数/多参数）
            if isinstance(tool_input, dict):
                result = tool.func(**tool_input)
            else:
                result = tool.func(tool_input)
            return str(result)
        except Exception as e:
            return f"工具调用失败：{str(e)}"

    def invoke(self, inputs: Dict, config: Optional[RunnableConfig] = None) -> Dict:
        """
        同步调用Agent（核心接口）
        :param inputs: 输入参数 {
            "user_input":  用户问题,
            "messages":    对话消息列表,
            "role_prompt": 角色设定（可选）,
            "session_id":  会话ID（可选）
        }
        :param config: 运行配置
        :return: {
            "final_answer": 最终回复,
            "thought":      思考过程,
            "actions":      工具调用记录,
            "messages":     更新后的消息列表
        }
        """
        # 1. 初始化参数
        user_input = inputs.get("user_input")
        messages = inputs.get("messages", [])
        role_prompt = inputs.get("role_prompt") or self.config.default_role_prompt
        session_id = inputs.get("session_id", "default")
        actions = []  # 记录工具调用历史

        # 2. 构建提示词
        prompt = self.prompt_template.format(
            tool_descriptions=self._format_tool_descriptions(),
            messages=messages,
            user_input=user_input,
            role_prompt=role_prompt[:100]
        )

        # 3. ReAct循环（思考→行动→观察）
        for _ in range(self.config.max_iterations):
            # 调用LLM生成思考/行动
            llm_output = self.llm.invoke(prompt)
            parsed_output = self._parse_agent_output(llm_output)

            # 记录思考过程
            actions.append({
                "thought": parsed_output["thought"],
                "action": parsed_output["action"],
                "action_input": parsed_output["action_input"]
            })

            # 有最终答案则退出循环
            if parsed_output["final_answer"]:
                return {
                    "final_answer": parsed_output["final_answer"],
                    "thought": parsed_output["thought"],
                    "actions": actions,
                    "messages": messages + [AIMessage(content=parsed_output["final_answer"])]
                }

            # 调用工具
            if parsed_output["action"]:
                # 补充session_id到工具入参
                tool_input = parsed_output["action_input"] or {}
                if "session_id" not in tool_input:
                    tool_input["session_id"] = session_id
                # 执行工具
                observation = self._run_tool(parsed_output["action"], tool_input)
                parsed_output["observation"] = observation
                # 更新提示词（追加工具调用结果）
                prompt += f"\n观察：{observation}"
            else:
                # 无工具调用，直接返回
                final_answer = parsed_output["thought"] or "暂无有效回答"
                return {
                    "final_answer": final_answer,
                    "thought": parsed_output["thought"],
                    "actions": actions,
                    "messages": messages + [AIMessage(content=final_answer)]
                }

        # 超过最大迭代次数
        return {
            "final_answer": "思考超时，请简化问题重试",
            "thought": "超过最大迭代次数",
            "actions": actions,
            "messages": messages
        }

    def stream(self, inputs: Dict, config: Optional[RunnableConfig] = None):
        """
        流式调用Agent（核心接口）
        :yield: 逐块返回思考/行动/最终答案
        """
        result = self.invoke(inputs, config)
        # 模拟流式输出（可根据需要改为实时流式）
        yield {"type": "thought", "content": result["thought"]}
        for action in result["actions"]:
            yield {"type": "action", "content": f"调用工具：{action['action']}，入参：{action['action_input']}"}
        yield {"type": "final_answer", "content": result["final_answer"]}

    # ========== 扩展接口（便于集成到Graph） ==========
    def as_node(self) -> Callable:
        """返回适配LangGraph的节点函数"""
        def node_fn(state: Dict) -> Dict:
            result = self.invoke(state)
            state["assistant_response"] = result["final_answer"]
            state["thought"] = result["thought"]
            state["actions"] = result["actions"]
            state["messages"] = result["messages"]
            return state
        return node_fn