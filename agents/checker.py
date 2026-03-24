from pydantic import BaseModel, Field

from core.config import FAST_MODEL
from core.llm_client import generate_structured_data
from core.schemas import NLPBaseTraits


class CheckResult(BaseModel):
    is_passed: bool = Field(
        ...,
        description="是否通过一致性校验，True 表示正文符合人设与章节脑洞要求。",
    )
    feedback: str = Field(
        ...,
        description="当校验未通过时给出一针见血的修改意见；通过时可返回简短通过说明。",
    )


def check_consistency(
    draft_text: str,
    character_traits: NLPBaseTraits,
    chapter_idea: str,
    model: str = FAST_MODEL,
) -> CheckResult:
    system_prompt = (
        "你是网文一致性裁判。请严格检查章节正文是否符合角色设定与本章脑洞。"
        "若不符合，必须指出最关键的问题并给出可执行修改建议。"
        "输出必须严格匹配 CheckResult。"
    )
    user_prompt = (
        f"本章脑洞：{chapter_idea}\n"
        "角色设定(JSON)：\n"
        f"{character_traits.model_dump_json(indent=2)}\n"
        "正文草稿：\n"
        f"{draft_text}"
    )
    return generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=CheckResult,
        model=model,
        temperature=0.0,
    )
