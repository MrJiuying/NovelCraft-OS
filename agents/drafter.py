from core.config import SMART_MODEL
from core.llm_client import generate_text
from core.schemas import ChapterBeatTemplate, NLPBaseTraits, WritingRule


def draft_chapter(
    chapter_number: int,
    beats: ChapterBeatTemplate,
    character_traits: NLPBaseTraits,
    mounted_rules: list[WritingRule] | None = None,
    checker_feedback: str = "",
    model: str = SMART_MODEL,
) -> str:
    style_guidelines: list[str] = []
    negative_constraints: list[str] = []
    for rule in mounted_rules or []:
        for instruction in rule.positive_instructions:
            style_guidelines.append(str(instruction))
        for constraint in rule.negative_constraints:
            negative_constraints.append(str(constraint))
    style_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(style_guidelines))
    negative_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(negative_constraints))
    traits_text = character_traits.model_dump_json(indent=2)
    beats_text = beats.model_dump_json(indent=2)

    system_prompt = (
        "你是一个顶级的网文主笔。请严格按照传入的 4 个剧情节拍，以及主角的心理设定，"
        "将其扩写为完整的章节正文。必须遵循动态挂载法则中的正向指令，并绝对遵守负向约束。\n"
        "动态法则正向指令：\n"
        f"{style_text}\n"
        "动态法则负向约束：\n"
        f"{negative_text}\n"
        "硬性要求：\n"
        "1) 必须覆盖四个节拍，且推进顺序不可错乱。\n"
        "2) 角色行为与心理变化必须与角色设定一致。\n"
        "3) 叙事以可读性和爽点密度优先，避免空泛总结。\n"
        "4) 输出仅为章节正文，不要添加解释、标题或备注。"
    )

    user_prompt = (
        f"章节号：{chapter_number}\n"
        "四拍节奏模板(JSON)：\n"
        f"{beats_text}\n"
        "主角当前六层人物设定(JSON)：\n"
        f"{traits_text}\n"
        f"裁判反馈：{checker_feedback or '无'}"
    )

    return generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.8,
    )
