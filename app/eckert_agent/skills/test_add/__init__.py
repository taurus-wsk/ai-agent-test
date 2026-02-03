# -*- coding: utf-8 -*-
"""
工具包初始化：暴露所有可用工具，方便外部导入
"""
# 从 add_tool.py 中导入加法工具，暴露给外部
from .add_tool import add_numbers_tool, AddToolInput,load_add_prompt

# 定义公开接口（规范导入，避免冗余）
__all__ = ["add_numbers_tool", "AddToolInput","load_add_prompt"]