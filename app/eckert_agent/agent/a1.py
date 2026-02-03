def _agent_node(self, state: AgentState) -> AgentState:
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

        print("AI: ", end="", flush=True)
        for chunk in self.base_agent.stream(current_state):
            # 健壮提取 chunk 中的有效内容
            try:
                current_content = chunk['model']['messages'][0].content
            except (IndexError, KeyError):
                current_content = ""

            if current_content:
                print(current_content, end="", flush=True)
                full_response_content += current_content

        # 子步骤 2.5：封装 AI 响应为 AIMessage，更新当前状态（关键：维护上下文）
        if full_response_content:
            final_ai_message = AIMessage(content=full_response_content)
            # 追加 AI 回复到 messages，更新当前状态（为下一轮对话做准备）
            new_messages = current_state.messages.copy()
            new_messages.append(final_ai_message)
            current_state = AgentState(
                messages=new_messages,
                intermediate_steps=current_state.intermediate_steps
            )

        # 子步骤 2.6：准备下一轮输入，保持交互格式整洁
        print("\n")
        print("用户：", end="", flush=True)

    # 步骤 3：循环终止，返回最终更新后的状态（包含完整循环对话历史）
    return AgentState(
        messages=current_state.messages,
        intermediate_steps=current_state.intermediate_steps
    )



