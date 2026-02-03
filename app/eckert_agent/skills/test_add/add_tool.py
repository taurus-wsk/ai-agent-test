# -*- coding: utf-8 -*-
"""
加法工具实现（单独文件，模块化封装，可直接导入使用）
"""
from pydantic import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool

from app.utils.utils import clean_md_content


def load_add_prompt(md_path: str = "app/eckert_agent/skills/test_add/skill.md") -> str:
    """从 md 文件中加载 system_prompt 内容"""
    with open(md_path, "r", encoding="utf-8") as f:
        raw_md = f.read()

        # 步骤 2：清洗 md 内容，去除格式冗余
    clean_content = clean_md_content(raw_md)
    return clean_content

# 1. 定义加法工具的输入参数 Schema（结构化，用于参数校验）
class AddToolInput(BaseModel):
    """加法工具输入参数模型（约束两个相加的数字）"""
    a: float = Field(..., description="需要相加的第一个数字（整数/小数均可）")
    b: float = Field(..., description="需要相加的第二个数字（整数/小数均可）")

# 2. 定义加法工具的核心实现函数
def add_numbers(a: float, b: float) -> str:
    """
    加法工具核心逻辑：计算两个数字的和并返回结构化结果
    :param a: 第一个数字
    :param b: 第二个数字
    :return: 加法计算结果字符串
    """
    print(f"调用了add_numbers,a:{a},b:{b}")
    try:
        result = a + b
        return f"加法计算成功：{a} + {b} = {result}"
    except Exception as e:
        return f"加法计算失败：{str(e)}"

# 3. 封装为标准 LangChain Tool（可直接传入 create_agent() 的 tools 参数）
add_numbers_tool = StructuredTool.from_function(
    name="add_numbers_tool",
    description="用于完成两个数字的加法运算，支持整数、小数相加，返回精确计算结果",
    func=add_numbers,
    args_schema=AddToolInput  # 绑定输入参数 Schema，自动校验
)