from core.config import SMART_MODEL
from core.llm_client import generate_structured_data
from core.schemas import WritingRule


def anatomize_fiction_snippet(
    text: str,
    element_target: str,
    model: str = SMART_MODEL,
) -> WritingRule:
    system_prompt = (
        "你是小说质量分析师，负责把高质量小说片段解剖为可执行写作法则。"
        "只输出严格匹配 WritingRule 的结构化结果。"
        "category 必须设置为 Elements。"
        "positive_instructions 与 negative_constraints 必须都是可直接执行的短句列表。"
    )
    user_prompt = (
        f"解剖目标要素：{element_target}\n"
        f"原始文本：\n{text}\n"
        "请提炼一条高价值写作法则。"
    )
    return generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=WritingRule,
        model=model,
        temperature=0.3,
    )


def distill_tutorial_to_rule(
    text: str,
    category_target: str,
    model: str = SMART_MODEL,
) -> WritingRule:
    system_prompt = (
        "你是写作法则蒸馏师，负责把教程、避毒指南和考据资料转成可执行指令。"
        "只输出严格匹配 WritingRule 的结构化结果。"
        "category 必须与用户提供的目标分类一致，仅可使用 Elements/Theories/Taboos/Formatting/Lore/Tropes。"
        "positive_instructions 必须是 Do 列表，negative_constraints 必须是 Don't 列表。"
    )
    user_prompt = (
        f"目标分类：{category_target}\n"
        f"原始资料：\n{text}\n"
        "请提炼一条严苛写作法则。"
    )
    return generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=WritingRule,
        model=model,
        temperature=0.3,
    )
