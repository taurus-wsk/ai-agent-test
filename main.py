from app.eckert_agent.graph.chat_graph import ChatGraph
# from app.eckert_agent.graph.chat_graph import ChatState  # 导入状态结构
from config import DEFAULT_SESSION_ID

if __name__ == "__main__":
    chat_graph = ChatGraph()
    res=chat_graph.run()
    print(res)