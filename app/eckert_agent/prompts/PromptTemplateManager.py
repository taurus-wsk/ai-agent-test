from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Optional


class PromptTemplateManager:
    """提示词模板管理类：封装模板构建、角色切换、规则扩展"""

    def __init__(
            self,
            default_role_prompt: str = None,
            default_reply_rules: List[str] = None,
            max_role_length: int = 100,
            max_rule_length: int = 50
    ):
        """
        初始化提示词管理器
        :param default_role_prompt: 默认角色设定（100字以内）
        :param default_reply_rules: 默认回复规则列表（每条50字以内）
        :param max_role_length: 角色设定最大长度限制
        :param max_rule_length: 单条规则最大长度限制
        """
        # 默认角色设定（兜底）
        self.default_role_prompt = default_role_prompt or "你是友好的AI助手，用简洁的中文回复用户问题。"
        # 默认回复规则（可自定义扩展）
        self.default_reply_rules = default_reply_rules or [
            "优先基于【知识库参考内容】回答问题；",
            "结合【对话历史】补充上下文；",
            "用简洁的中文回复，不啰嗦；",
            "如果知识库没有相关内容，仅基于对话历史回答。"
        ]
        # 长度限制（防滥用）
        self.max_role_length = max_role_length
        self.max_rule_length = max_rule_length

    def _format_role_prompt(self, role_prompt: Optional[str]) -> str:
        """格式化角色设定（长度限制+兜底）"""
        # 优先使用传入的角色，无则用默认
        final_role = role_prompt or self.default_role_prompt
        # 长度限制
        return final_role[:self.max_role_length].strip()

    def _format_reply_rules(self, custom_rules: Optional[List[str]] = None) -> str:
        """格式化回复规则（合并默认+自定义，长度限制）"""
        # 合并规则：自定义规则优先，无则用默认
        rules = custom_rules or self.default_reply_rules
        # 每条规则长度限制 + 拼接成文本
        formatted_rules = []
        for idx, rule in enumerate(rules, 1):
            rule = rule[:self.max_rule_length].strip()
            if rule:  # 过滤空规则
                formatted_rules.append(f"{idx}. {rule}")
        return "\n".join(formatted_rules)

    def build_system_message(self, role_prompt: Optional[str] = None,
                             custom_rules: Optional[List[str]] = None) -> SystemMessage:
        """构建系统消息（角色设定+回复规则）"""
        # 格式化角色和规则
        formatted_role = self._format_role_prompt(role_prompt)
        formatted_rules = self._format_reply_rules(custom_rules)
        tool_descriptions=""
        # 系统消息模板
        system_content = f"""【角色设定】：{formatted_role}
             ReAct执行规则】：
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
               最终答案：[给用户的回复]
            {formatted_rules}"""
        return SystemMessage(content=system_content)

    def build_human_message_template(self) -> HumanMessage:
        """构建用户消息模板（包含用户问题+知识库参考）"""
        human_content = """【用户问题】：{user_input}
        【知识库参考】：{knowledge_context}"""
        return HumanMessage(content=human_content)

    def build_chat_prompt_template(
            self,
            role_prompt: Optional[str] = None,
            custom_rules: Optional[List[str]] = None
    ) -> ChatPromptTemplate:
        """
        构建完整的ChatPromptTemplate
        :param role_prompt: 临时角色设定（覆盖默认）
        :param custom_rules: 临时回复规则（覆盖默认）
        :return: 编译好的ChatPromptTemplate
        """
        # 组装模板组件
        messages = [
            # 系统消息（动态角色+规则）
            self.build_system_message(role_prompt, custom_rules),
            # 对话历史占位符
            MessagesPlaceholder(variable_name="chat_history"),
            # 用户消息模板
            self.build_human_message_template()
        ]
        # 创建ChatPromptTemplate
        return ChatPromptTemplate.from_messages(messages)

    def format_prompt(
            self,
            prompt_template: ChatPromptTemplate,
            chat_history: List,
            user_input: str,
            knowledge_context: str
    ) -> str:
        """
        填充模板参数，生成最终提示词
        :param prompt_template: 构建好的ChatPromptTemplate
        :param chat_history: 对话历史（消息对象列表）
        :param user_input: 用户输入
        :param knowledge_context: 知识库检索结果
        :return: 格式化后的纯文本提示词
        """
        return prompt_template.format(
            chat_history=chat_history,
            user_input=user_input,
            knowledge_context=knowledge_context
        )

    # ========== 扩展方法：支持模板定制 ==========
    def update_default_role(self, new_role: str):
        """更新全局默认角色设定"""
        self.default_role_prompt = new_role[:self.max_role_length].strip()

    def update_default_rules(self, new_rules: List[str]):
        """更新全局默认回复规则"""
        self.default_reply_rules = [rule[:self.max_rule_length].strip() for rule in new_rules if rule.strip()]

    def add_custom_rule(self, new_rule: str):
        """新增一条默认回复规则"""
        if new_rule.strip():
            self.default_reply_rules.append(new_rule[:self.max_rule_length].strip())