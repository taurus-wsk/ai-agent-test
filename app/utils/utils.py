import re


def clean_md_content(md_content: str) -> str:
    """
    清洗 md 内容，去除格式冗余，返回干净的纯文本规则
    :param md_content: 从 md 文件读取的原始内容
    :return: 清洗后的纯文本内容，无换行符、md 语法残留
    """
    # 步骤 1：去除 md 分隔线（---）
    # content = re.sub(r"---+", "", md_content)

    # 步骤 2：去除 md 反引号（`xxx` → xxx）
    content = re.sub(r"`(.*?)`", r"\1", md_content)

    # 步骤 3：去除 md 标题标记（# 、## 等，保留标题内容）
    content = re.sub(r"#{1,6}\s*", "", content)

    # 步骤 4：去除多余换行符（多个 \n 转为单个空格，避免规则拆分）
    content = re.sub(r"\n+", " ", content)

    # 步骤 5：去除多余空格（连续空格、首尾空格，保留规则内必要空格）
    content = re.sub(r"\s+", " ", content).strip()

    # 步骤 6：去除无意义的空字符串（可选）
    if not content:
        return "默认规则：使用 add_numbers_tool 完成加法计算，参数格式 {\"a\": 数字1, \"b\": 数字2}"

    return content