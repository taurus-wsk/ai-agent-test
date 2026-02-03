import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool  # 工具定义必备
from langchain.agents import create_agent  # 导入智能代理构建函数

from langchain_core.messages import SystemMessage, HumanMessage

# 1. 加载环境变量
load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 2. 初始化 ChatOllama（保留 streaming=True，兼容流式输出）
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7,

    # 补全你之前遗漏的 Ollama 原生格式配置
)


# 3. 步骤1：定义一个演示工具（计算工具，符合 BaseTool 规范，代理可调用）
# 用 @tool 装饰器快速定义工具，简化 BaseTool 子类实现
@tool
def calculate_math(expression: str) -> str:
    """
    用于解决数学计算问题的工具，支持加减乘除四则运算。
    参数 expression：字符串格式的数学表达式，例如 "100+200"、"500*3-100"
    """
    try:
        # 简单实现：使用 eval 计算（仅用于演示，生产环境需替换为安全计算逻辑）
        result = eval(expression)
        return f"数学表达式 '{expression}' 的计算结果为：{result}"
    except Exception as e:
        return f"计算失败，错误原因：{str(e)}"


# 4. 步骤2：定义代理的提示词模板（复用原有 ChatPromptTemplate 结构，整合工具相关内容）
# 代理提示词需要包含工具使用说明、scratchpad（思考过程）
def build_agent_prompt_template():
    # 系统提示词：整合小说家身份 + 工具使用说明
    system_prompt = """
"""

    # 构建 ChatPromptTemplate，包含代理必需的字段
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("human", "{user_input}"),

    ])

    return prompt


# 5. 步骤3：使用 create_agent() 构建智能代理
def build_novelist_agent():
    # 5.1 获取提示词模板
    agent_prompt = build_agent_prompt_template()


    # 5.5 调用 create_agent() 构建完整代理（核心步骤）
    agent = create_agent(
        model=llm,  # 传入绑定了工具的模型实例
        # tools=tools,  # 传入可用工具列表
        system_prompt=agent_prompt.messages[0].content,  # 传入系统提示词
        debug=True  # 关闭调试模式，如需排查问题可改为 True
    )

    return agent


# 6. 步骤4：核心：代理流式输出（保留原有打字机效果，解决空白回复）
def agent_stream_output(user_question: str = "你好，1加1等于几"):
    # 6.1 构建代理配置（固定 thread_id，用于会话持久化）
    agent_config = {"configurable": {"thread_id": "novelist_agent_001"}}

    # 6.2 构建代理输入
    # agent_input = {
    #     "user_input": user_question,
    #     "intermediate_steps": []  # 初始化工具调用中间步骤
    # }

    # 6.3 获取构建好的代理
    novelist_agent = build_novelist_agent()

    # 6.4 流式接收代理输出（保留原有打字机效果）
    print("🤖 代理响应（流式输出）：", end="", flush=True)
    full_response = ""

    print("===== stream_mode='updates' 演示 =====")
    # res = novelist_agent.invoke(input={"messages": [("user", "你好3+2等于几")]}, config=agent_config)
#     input=[{"messages": [
#     # ("system", "你是计算助手"),
#     HumanMessage(content="你好3+2等于几")
# ]}]
    stream_iterator = novelist_agent.stream(
        input={"messages": [
            # 系统消息：定义代理身份/规则
            SystemMessage(content="你是一个严谨的计算助手，只返回纯数字计算结果，不添加其他内容。"),
            # 用户消息：实际查询内容
            HumanMessage(content=user_question)
        ]},
        config=agent_config
    )
    for step_data in stream_iterator:
        print(f"✅ 步骤数据（仅更新内容）：{step_data['model']['messages'][0].content}\n")

    print(f"\n\n✅ 代理完整响应结果：{full_response}")


# 7. 运行测试
if __name__ == "__main__":
    # 测试：同时包含问候（小说家身份）和数学计算（工具调用）
    agent_stream_output(user_question="你好，1加1等于几")