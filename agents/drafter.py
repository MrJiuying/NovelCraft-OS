from core.config import SMART_MODEL
from core.config_manager import ConfigManager
from core.llm_client import generate_text
from core.schemas import ChapterBeatTemplate, NLPBaseTraits


def draft_chapter(
    chapter_number: int,
    beats: ChapterBeatTemplate,
    character_traits: NLPBaseTraits,
    platform: str = "番茄小说",
    checker_feedback: str = "",
    model: str = SMART_MODEL,
) -> str:
    # 初始化配置管理器并加载平台配置，用于控制文风与违禁词约束。
    config_manager = ConfigManager()
    platform_config = config_manager.load_platform_config(platform)

    # 读取文风指南与违禁词库，并格式化为可读文本注入提示词。
    style_guidelines = platform_config.get("style_guidelines", [])
    banned_words = platform_config.get("banned_words", [])
    style_text = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(style_guidelines))
    banned_text = "、".join(str(word) for word in banned_words)
    traits_text = character_traits.model_dump_json(indent=2)
    beats_text = beats.model_dump_json(indent=2)

    # system_prompt 强制模型执行“按四拍扩写 + 按人设落地 + 严守文风与违禁词”的复合约束。
    system_prompt = (
        "你是一个顶级的网文主笔。请严格按照传入的 4 个剧情节拍，以及主角的心理设定，"
        "将其扩写为完整的章节正文。必须遵循文风指南，且绝对不允许使用违禁词库中的词汇。\n"
        f"平台名称：{platform_config.get('platform_name', platform)}\n"
        "文风指南：\n"
        f"{style_text}\n"
        "违禁词库：\n"
        f"{banned_text}\n"
        "硬性要求：\n"
        "1) 必须覆盖四个节拍，且推进顺序不可错乱。\n"
        "2) 角色行为与心理变化必须与角色设定一致。\n"
        "3) 叙事以可读性和爽点密度优先，避免空泛总结。\n"
        "4) 输出仅为章节正文，不要添加解释、标题或备注。"
    )

    # user_prompt 注入章节号、节拍模板与角色画像，保证模型在固定上下文中稳定创作。
    user_prompt = (
        f"章节号：{chapter_number}\n"
        "四拍节奏模板(JSON)：\n"
        f"{beats_text}\n"
        "主角当前六层人物设定(JSON)：\n"
        f"{traits_text}\n"
        f"裁判反馈：{checker_feedback or '无'}"
    )

    # 使用文本生成通道扩写正文，温度设为 0.8 以释放词汇创造力。
    return generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.8,
    )
