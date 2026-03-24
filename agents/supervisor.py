from pydantic import BaseModel, Field

from core.config import SMART_MODEL
from core.llm_client import generate_structured_data
from core.schemas import BookOutline, ChapterOutlineList, ConceptProposal, NLPBaseTraits, VolumeOutline


class BrainstormResult(BaseModel):
    assistant_reply: str = Field(
        ...,
        description="AI 对当前灵感的回应文本，包含追问或方向建议。",
    )
    quick_options: list[str] = Field(
        ...,
        description="可供点击的快捷建议列表，建议 2-4 条。",
    )
    proposal: ConceptProposal = Field(
        ...,
        description="当前阶段的策划案草稿，会随着对话逐步收敛。",
    )


def brainstorm_ideas(
    current_idea: str,
    chat_history: list,
    model: str = SMART_MODEL,
) -> dict:
    system_prompt = (
        "你是网文项目主编，正在与执行编剧进行创作沟通。"
        "你现在不能写正式大纲，只能做头脑风暴：提出高价值问题或给出可选剧情走向。"
        "输出必须严格匹配 BrainstormResult。"
    )
    user_prompt = (
        f"当前脑洞：{current_idea}\n"
        f"历史对话：{chat_history}\n"
        "请基于上下文输出一段主编回复、2-4个快捷选项，并同步更新策划案草稿。"
    )
    result = generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=BrainstormResult,
        model=model,
        temperature=0.7,
    )
    return result.model_dump()


def finalize_proposal(
    user_responses: list,
    model: str = SMART_MODEL,
) -> ConceptProposal:
    system_prompt = (
        "你是网文总编，负责把对话记录收敛成可执行的故事核心策划案。"
        "输出必须严格匹配 ConceptProposal，内容要具体、可落地。"
    )
    user_prompt = f"用户回应列表：{user_responses}"
    return generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ConceptProposal,
        model=model,
        temperature=0.7,
    )


def generate_book_outline(
    user_idea: str,
    model: str = SMART_MODEL,
    concept_proposal: ConceptProposal | None = None,
) -> BookOutline:
    system_prompt = (
        "你是网文白金作家兼资深主编。你的目标是把用户脑洞推演成可商业连载的重度追更型全书总纲。"
        "输出必须避免空泛口号，必须给出可执行的剧情推进设计、明确字数规划与节奏爆点分布。"
        "请确保主线冲突可持续升级、力量体系可递进扩展、分卷布局可落地执行、结局方向可兑现。"
        "target_word_count 必须给出明确体量；pacing_design 必须说明前中后期高潮节点与节奏控制策略。"
        "【强制字数与章节数学逻辑】"
        "1) 绝对基准：必须严格遵循 1章=2000字。"
        "2) 卷字数对应：若某卷为20万字，则该卷必须对应100章。"
        "3) 节点对应：若某剧情节点设在10万字爆发，则对应第50章，禁止出现与数学关系冲突的章位。"
        "4) 总数对应：若全书预计300万字，则大结局应在第1500章左右。"
        "5) pacing_design 输出时，所有关键节点必须标注“（约X万字，第Y章）”，且 X 与 Y 必须满足 1万字=5章（即 1:50）。"
    )
    proposal_text = (
        concept_proposal.model_dump_json(indent=2) if concept_proposal else "未提供"
    )
    user_prompt = (
        f"用户核心脑洞：{user_idea}\n"
        f"策划案草稿：{proposal_text}\n"
        "请基于该脑洞输出全书总纲，并严格遵守 BookOutline 的字段结构。"
    )
    return generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=BookOutline,
        model=model,
        temperature=0.7,
    )


def generate_volume_outline(
    book_outline: BookOutline,
    target_volume_num: int,
    model: str = SMART_MODEL,
) -> VolumeOutline:
    system_prompt = (
        "你是网文白金主编，负责制定单卷连载作战方案。"
        "请基于全书总纲为目标卷号输出极其详实、节奏明确、商业化可执行的分卷大纲。"
        "必须明确本卷核心冲突、新增势力、预计字数与核心支线，杜绝假大空描述。"
        "estimated_word_count 需给出明确规模；key_subplots 需给出可独立推动读者留存的支线条目。"
        "【强制字数与章节数学逻辑】"
        "1) 必须严格遵循 1章=2000字。"
        "2) 本卷 estimated_word_count 必须可换算为明确章节量，并与支线规划章位一致。"
        "3) key_subplots 每个条目都要写出章节区间与字数体量，例如："
        "“支线A：网吧冲突（第1-15章，约3万字）”。"
        "4) 章节区间与字数必须满足 1:50 的万字-章数关系，禁止数学冲突。"
    )
    user_prompt = (
        f"目标卷号：{target_volume_num}\n"
        "全书总纲(JSON)：\n"
        f"{book_outline.model_dump_json(indent=2)}\n"
        "请输出 VolumeOutline，并确保 volume_number 与目标卷号一致。"
    )
    result = generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=VolumeOutline,
        model=model,
        temperature=0.7,
    )
    if result.volume_number != target_volume_num:
        return result.model_copy(update={"volume_number": target_volume_num})
    return result


def generate_chapter_ideas(
    volume_outline: VolumeOutline,
    chapter_count: int,
    model: str = SMART_MODEL,
) -> ChapterOutlineList:
    system_prompt = (
        "你是网文章节连载编排师。请根据分卷核心冲突，把剧情拆解为连续可追更的单章核心事件。"
        "章节事件必须递进升级、前后因果紧密，并服务于本卷冲突闭环。"
        "输出必须严格遵守 ChapterOutlineList 结构。"
    )
    user_prompt = (
        f"目标章节数：{chapter_count}\n"
        "分卷大纲(JSON)：\n"
        f"{volume_outline.model_dump_json(indent=2)}\n"
        "请生成 chapters 列表，列表长度必须等于目标章节数，"
        "每个元素必须包含 chapter_number 与 core_event。"
    )
    result = generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ChapterOutlineList,
        model=model,
        temperature=0.7,
    )
    normalized = result.model_copy(update={"volume_number": volume_outline.volume_number})
    if len(normalized.chapters) != chapter_count:
        raise ValueError(f"生成章节数量不匹配，期望 {chapter_count}，实际 {len(normalized.chapters)}")
    return normalized


def derive_initial_traits(
    user_idea: str,
    book_outline: BookOutline,
    concept_proposal: ConceptProposal | None = None,
    model: str = SMART_MODEL,
) -> NLPBaseTraits:
    system_prompt = (
        "你是网文角色总监。请根据脑洞、策划案和全书总纲，提炼主角初始六维人设。"
        "输出必须严格遵守 NLPBaseTraits。内容要具体可执行。"
    )
    proposal_text = concept_proposal.model_dump_json(indent=2) if concept_proposal else "未提供"
    user_prompt = (
        f"用户脑洞：{user_idea}\n"
        f"策划案：{proposal_text}\n"
        f"全书总纲：{book_outline.model_dump_json(indent=2)}\n"
        "请输出主角初始人设。"
    )
    return generate_structured_data(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=NLPBaseTraits,
        model=model,
        temperature=0.7,
    )
