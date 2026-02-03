import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
# 导入你贴出的这个新版 create_agent
from langchain.agents import create_agent
# 导入你的 PostgresSaver（检查点，用于持久化）
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
# 1. 加载环境变量
load_dotenv()

# 2. 初始化 ChatOllama（已验证可用，直接复用）
def init_ollama_llm():
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.7,
        timeout=30.0,

        streaming=True
    )

# 3. 初始化 PostgresSaver（检查点，传入 create_agent 实现持久化）
def init_postgres_checkpointer():
    try:
        # 读取 PostgreSQL 配置（从 .env 中加载，或直接硬编码测试）
        conn_params = {
            "host": os.getenv("PG_HOST", "localhost"),
            "port": int(os.getenv("PG_PORT", 5432)),
            "user": os.getenv("PG_USER", "postgres"),
            "password": os.getenv("PG_PASSWORD", "postgres"),
            "dbname": os.getenv("PG_DB", "langgraph_db"),
            # "connection_factory": extras.DictConnection,
            # 核心：添加 UTF-8 编码配置，解决中文乱码
            "options": "-c client_encoding=utf8"
        }

        pg_conn = psycopg.connect(**conn_params)
        pg_conn.autocommit = True  # 自动提交事务，确保表创建/数据插入生效

        # 3. 可选：手动修改数据库默认编码（一次性执行，确保数据库本身支持 UTF-8）
        # 3. 仅保留有效的 client_encoding 配置（删除无效的 encoding 配置）
        with pg_conn.cursor() as cur:
            db_name = os.getenv("PG_DB", "langgraph_db")
            # 只执行 client_encoding 配置（运行时有效，支持中文存储/显示）
            cur.execute(f"ALTER DATABASE {db_name} SET client_encoding = 'utf8';")
            print(f"✅ 数据库 {db_name} 已设置客户端编码为 UTF-8（有效配置）")

        # 4. 初始化 PostgresSaver（适配新版 psycopg 连接，参数不变）
        checkpointer = PostgresSaver(
            conn=pg_conn,
            pipe=None,
            serde=None
        )

        # 初始化表（首次运行创建所需表）
        checkpointer.setup()
        print("✅ PostgresSaver 初始化成功（检查点就绪）")
        return checkpointer

    except Exception as e:
        print(f"❌ PostgresSaver 初始化失败：{str(e)}")
        return None

# 4. 定义工具（简单计算工具，传入 create_agent）
@tool
def calculate(num1: float, num2: float, operation: str = "+") -> str:
    """
    用于执行简单的数学运算，支持 +、-、*、/ 四种操作。
    参数：
    - num1: 第一个数字
    - num2: 第二个数字
    - operation: 运算符号，可选值：+、-、*、/，默认是 +
    """
    operations = {
        "+": num1 + num2,
        "-": num1 - num2,
        "*": num1 * num2,
        "/": num1 / num2 if num2 != 0 else "错误：除数不能为 0"
    }
    result = operations.get(operation, "错误：不支持的运算符号，仅支持 +、-、*、/")
    text= f"{num1} {operation} {num2} = {result}"
    return text

tools = [calculate]

# 5. 适配新版 create_agent，创建智能体（LangGraph 整合版）
def test_new_create_agent():
    try:
        # 步骤 1：初始化核心组件
        llm = init_ollama_llm()
        checkpointer = init_postgres_checkpointer()
        system_prompt = """不寒暄、不解释、不发表情、无多余文字"""
        print(f"✅ 已初始化核心组件（模型：{llm.model}）")

        # 步骤 2：调用新版 create_agent（完全匹配你贴出的参数格式）
        agent_graph = create_agent(
            model=llm,  # 必传：已验证的 ChatOllama 实例
            tools=[],  # 推荐传：计算工具列表
            system_prompt=system_prompt,  # 推荐传：系统提示词（字符串即可）
            checkpointer=checkpointer,  # 可选传：PostgresSaver 检查点（持久化）
            debug=True,  # 关闭调试模式，如需排查可改为 True
            name="math_agent"  # 智能体名称（可选）
        )
        print("✅ 新版 create_agent 调用成功，返回 LangGraph 状态图")

        # 步骤 3：运行智能体（传入简单输入，无需手动定义 AgentState）
        # print("\n✅ 正在运行智能体，计算 100 + 200...")
        # 输入格式：直接传入字典，key 为 "input" 即可（内部已封装 AgentState）
        input_data = {
            "input": "",
        }
        # 配置：传入 thread_id，实现会话持久化（对应 PostgresSaver 的检查点）
        config = {
            "configurable": {
                "thread_id": "test_thread_001"  # 自定义会话 ID，用于区分不同会话
            }
        }
        # 运行 LangGraph 状态图（invoke 同步调用）
        # 步骤 4：第一次调用 - 初始化会话，计算 100 + 200
        print("=== 第一次调用智能体（初始化会话，psycopg v3+）===")
        input1 = {
            "input": "我叫何增辉，请计算 100 + 200 的结果，记住这个结果（"
        }
        response1 = agent_graph.invoke(input1, config=config)
        final1 = [msg for msg in response1['messages'] if isinstance(msg, AIMessage)][-1].content
        print(f"📌 第一次调用结果：{final1}")

        # 步骤 5：第二次调用 - 验证记忆，用上一轮结果加 500
        print("\n=== 第二次调用智能体（验证记忆，psycopg v3+）===")
        input2 = {
            "input": "还记得我叫名字吗"
        }
        response2 = agent_graph.invoke(input2, config=config)
        final2 = [msg for msg in response2['messages'] if isinstance(msg, AIMessage)][-1].content
        print(f"📌 第二次调用结果：{final2}")


        return True

    except Exception as e:
        print("\n❌ 新版 create_agent 调用/运行失败！")
        print(f"📌 错误详情：{str(e)}")
        print(f"📌 错误类型：{type(e).__name__}")
        return False

# 6. 运行验证
if __name__ == "__main__":
    test_new_create_agent()