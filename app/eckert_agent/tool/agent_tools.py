from langchain_core.tools import Tool

def search_digital_media_knowledge(query: str) -> str:
    """
    数字媒体专业知识库检索工具（Ollama调用无格式要求，输入为纯文本）
    适配场景：数字媒体专业的核心技能、课程、就业、工具使用等问题
    """
    knowledge_map = {
        "核心技能": "数字媒体专业核心技能：1.3D建模（Blender/C4D）；2.交互设计（Figma/AXURE）；3.影视剪辑（PR/AE）；4.AIGC工具（Midjourney/Runway）；5.游戏美术基础",
        "核心课程": "数字媒体专业核心课程：素描、色彩构成、数字图像处理、三维建模、影视编导、交互设计原理、游戏概论、新媒体运营",
        "就业方向": "数字媒体专业就业方向：创意类（剪辑师/3D设计师）、设计类（UI/UX/游戏美术）、运营类（新媒体/短视频编导）、稳定类（政企宣传/教育讲师）",
        "常用工具": "剪辑：PR/AE/剪映；建模：Blender/C4D；设计：Figma/PS/AI；AIGC：Midjourney/Runway；游戏引擎：Unity"
    }
    for key, value in knowledge_map.items():
        if key in query:
            return f"【知识库】{value}"
    return f"【知识库】未找到{query}相关内容，可询问核心技能、核心课程、就业方向、常用工具等问题。"

# 工具列表：Ollama模型自动识别，新增工具仅需添加此处
def get_agent_tools():
    return [
        Tool(
            name="SearchDigitalMediaKnowledge",
            func=search_digital_media_knowledge,
            description="用于检索数字媒体专业的核心技能、核心课程、就业方向、常用工具等专业信息，输入为用户问题的纯文本关键词"
        )
    ]