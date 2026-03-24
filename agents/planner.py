from core.config import SMART_MODEL
from core.config_manager import ConfigManager
from core.llm_client import generate_structured_data
from core.schemas import ChapterBeatTemplate, NLPBaseTraits


class BeatPlannerAgent:
    # 节拍规划代理：根据平台规则与角色设定，为单章生成 4 段式节拍模板。
    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        # 允许外部注入配置管理器，便于测试时替换或扩展配置来源。
        self.config_manager = config_manager or ConfigManager()

    # 按章节脑洞与角色画像生成结构化节拍，输出严格受 ChapterBeatTemplate 约束。
    def plan_chapter_beats(
        self,
        chapter_number: int,
        chapter_idea: str,
        character_traits: NLPBaseTraits,
        platform: str = "番茄小说",
        model: str = SMART_MODEL,
    ) -> ChapterBeatTemplate:
        # 加载平台节奏规则，确保节拍设计符合渠道读者预期。
        platform_config = self.config_manager.load_platform_config(platform)
        pacing_rules = platform_config.get("pacing_rules", [])
        pacing_text = "\n".join(f"{index + 1}. {rule}" for index, rule in enumerate(pacing_rules))
        traits_text = character_traits.model_dump_json(indent=2)

        # system_prompt 采用强约束指令，要求模型严格输出四拍结构并绑定指定章节号。
        system_prompt = (
            "你是一个网文节拍规划师，必须严格按照以下平台节奏规则，结合主角当前的性格设定，"
            "将本章拆解为 4 个具体的节拍指令。\n"
            f"平台名称：{platform_config.get('platform_name', platform)}\n"
            "平台节奏规则：\n"
            f"{pacing_text}\n"
            "强制要求：\n"
            "1) 输出必须精准匹配 ChapterBeatTemplate 字段。\n"
            "2) chapter_number 必须等于用户给定章节号。\n"
            "3) beat_1_setup 聚焦前 300 字铺垫与起因。\n"
            "4) beat_2_conflict 聚焦中间 800 字冲突与打压。\n"
            "5) beat_3_climax 聚焦中间 700 字高潮与反击。\n"
            "6) beat_4_cliffhanger 聚焦末尾 200 字悬念钩子。\n"
            "7) 节拍必须可执行、可写作，禁止空话与抽象口号。"
        )

        # user_prompt 注入章节脑洞与角色六层画像，为模型提供可落地的创作上下文。
        user_prompt = (
            f"章节号：{chapter_number}\n"
            f"本章脑洞：{chapter_idea}\n"
            "主角当前六层人物设定(JSON)：\n"
            f"{traits_text}"
        )

        result = generate_structured_data(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ChapterBeatTemplate,
            model=model,
            temperature=0.7,
        )

        # 二次兜底修正章节号，避免极端情况下模型偏离输入章节编号。
        if result.chapter_number != chapter_number:
            return result.model_copy(update={"chapter_number": chapter_number})
        return result


# 对外提供同名核心函数，便于在工作流中直接函数式调用。
def plan_chapter_beats(
    chapter_number: int,
    chapter_idea: str,
    character_traits: NLPBaseTraits,
    platform: str = "番茄小说",
    model: str = SMART_MODEL,
) -> ChapterBeatTemplate:
    planner = BeatPlannerAgent()
    return planner.plan_chapter_beats(
        chapter_number=chapter_number,
        chapter_idea=chapter_idea,
        character_traits=character_traits,
        platform=platform,
        model=model,
    )
