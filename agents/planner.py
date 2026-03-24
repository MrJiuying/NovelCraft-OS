from core.config import SMART_MODEL
from core.llm_client import generate_structured_data
from core.schemas import ChapterBeatTemplate, NLPBaseTraits, WritingRule


class BeatPlannerAgent:
    def __init__(self) -> None:
        pass

    def plan_chapter_beats(
        self,
        chapter_number: int,
        chapter_idea: str,
        character_traits: NLPBaseTraits,
        mounted_rules: list[WritingRule] | None = None,
        model: str = SMART_MODEL,
    ) -> ChapterBeatTemplate:
        pacing_rules = []
        for rule in mounted_rules or []:
            for instruction in rule.positive_instructions:
                pacing_rules.append(str(instruction))
        pacing_text = "\n".join(f"{index + 1}. {rule}" for index, rule in enumerate(pacing_rules))
        traits_text = character_traits.model_dump_json(indent=2)

        system_prompt = (
            "你是一个网文节拍规划师，必须严格按照以下动态挂载写作法则，结合主角当前的性格设定，"
            "将本章拆解为 4 个具体的节拍指令。\n"
            "动态法则指令：\n"
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

        if result.chapter_number != chapter_number:
            return result.model_copy(update={"chapter_number": chapter_number})
        return result


# 对外提供同名核心函数，便于在工作流中直接函数式调用。
def plan_chapter_beats(
    chapter_number: int,
    chapter_idea: str,
    character_traits: NLPBaseTraits,
    mounted_rules: list[WritingRule] | None = None,
    model: str = SMART_MODEL,
) -> ChapterBeatTemplate:
    planner = BeatPlannerAgent()
    return planner.plan_chapter_beats(
        chapter_number=chapter_number,
        chapter_idea=chapter_idea,
        character_traits=character_traits,
        mounted_rules=mounted_rules,
        model=model,
    )
