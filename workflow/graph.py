import logging
from time import perf_counter
from typing import Optional, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from agents.checker import check_consistency
from agents.drafter import draft_chapter
from agents.planner import plan_chapter_beats
from core.config import FAST_MODEL, SMART_MODEL
from core.schemas import ChapterBeatTemplate, NLPBaseTraits, WritingRule

logger = logging.getLogger(__name__)


class ChapterState(TypedDict):
    pipeline_id: str
    chapter_num: int
    chapter_idea: str
    mounted_rules: list[WritingRule]
    traits: NLPBaseTraits
    beats: Optional[ChapterBeatTemplate]
    draft: str
    checker_feedback: str
    retry_count: int
    planner_model: str
    drafter_model: str
    checker_model: str


def plan_node(state: ChapterState) -> ChapterState:
    start = perf_counter()
    logger.info(
        "pipeline.%s.plan.start chapter=%s rules=%s model=%s",
        state["pipeline_id"],
        state["chapter_num"],
        len(state["mounted_rules"]),
        state["planner_model"],
    )
    beats = plan_chapter_beats(
        chapter_number=state["chapter_num"],
        chapter_idea=state["chapter_idea"],
        character_traits=state["traits"],
        mounted_rules=state["mounted_rules"],
        model=state["planner_model"],
    )
    duration_ms = int((perf_counter() - start) * 1000)
    logger.info(
        "pipeline.%s.plan.done chapter=%s duration_ms=%s",
        state["pipeline_id"],
        state["chapter_num"],
        duration_ms,
    )
    return {**state, "beats": beats}


def draft_node(state: ChapterState) -> ChapterState:
    start = perf_counter()
    logger.info(
        "pipeline.%s.draft.start chapter=%s retry_count=%s model=%s",
        state["pipeline_id"],
        state["chapter_num"],
        state["retry_count"],
        state["drafter_model"],
    )
    beats = state["beats"]
    if beats is None:
        raise ValueError("beats 为空，无法生成正文。")
    draft = draft_chapter(
        chapter_number=state["chapter_num"],
        beats=beats,
        character_traits=state["traits"],
        mounted_rules=state["mounted_rules"],
        checker_feedback=state["checker_feedback"],
        model=state["drafter_model"],
    )
    duration_ms = int((perf_counter() - start) * 1000)
    logger.info(
        "pipeline.%s.draft.done chapter=%s duration_ms=%s output_chars=%s",
        state["pipeline_id"],
        state["chapter_num"],
        duration_ms,
        len(draft),
    )
    return {**state, "draft": draft}


def check_node(state: ChapterState) -> ChapterState:
    start = perf_counter()
    logger.info(
        "pipeline.%s.check.start chapter=%s retry_count=%s model=%s",
        state["pipeline_id"],
        state["chapter_num"],
        state["retry_count"],
        state["checker_model"],
    )
    result = check_consistency(
        draft_text=state["draft"],
        character_traits=state["traits"],
        chapter_idea=state["chapter_idea"],
        model=state["checker_model"],
    )
    duration_ms = int((perf_counter() - start) * 1000)
    logger.info(
        "pipeline.%s.check.done chapter=%s duration_ms=%s is_passed=%s feedback_chars=%s",
        state["pipeline_id"],
        state["chapter_num"],
        duration_ms,
        result.is_passed,
        len(result.feedback or ""),
    )
    if result.is_passed:
        return {**state, "checker_feedback": ""}
    return {
        **state,
        "checker_feedback": result.feedback,
        "retry_count": state["retry_count"] + 1,
    }


def check_router(state: ChapterState) -> str:
    if state["checker_feedback"] == "" or state["retry_count"] >= 3:
        logger.info(
            "pipeline.%s.route chapter=%s decision=end retry_count=%s feedback_empty=%s",
            state["pipeline_id"],
            state["chapter_num"],
            state["retry_count"],
            state["checker_feedback"] == "",
        )
        return "end"
    logger.info(
        "pipeline.%s.route chapter=%s decision=rewrite retry_count=%s",
        state["pipeline_id"],
        state["chapter_num"],
        state["retry_count"],
    )
    return "rewrite"


def _build_graph():
    graph = StateGraph(ChapterState)
    graph.add_node("plan_node", plan_node)
    graph.add_node("draft_node", draft_node)
    graph.add_node("check_node", check_node)
    graph.set_entry_point("plan_node")
    graph.add_edge("plan_node", "draft_node")
    graph.add_edge("draft_node", "check_node")
    graph.add_conditional_edges(
        "check_node",
        check_router,
        {
            "rewrite": "draft_node",
            "end": END,
        },
    )
    return graph.compile()


chapter_pipeline = _build_graph()


def run_chapter_pipeline(
    chapter_num: int,
    idea: str,
    traits: NLPBaseTraits,
    mounted_rules: list[WritingRule] | None = None,
    planner_model: str = SMART_MODEL,
    drafter_model: str = SMART_MODEL,
    checker_model: str = FAST_MODEL,
) -> str:
    pipeline_id = uuid4().hex[:8]
    start = perf_counter()
    logger.info(
        "pipeline.%s.start chapter=%s rules=%s planner_model=%s drafter_model=%s checker_model=%s",
        pipeline_id,
        chapter_num,
        len(mounted_rules or []),
        planner_model,
        drafter_model,
        checker_model,
    )
    try:
        final_state = chapter_pipeline.invoke(
            {
                "pipeline_id": pipeline_id,
                "chapter_num": chapter_num,
                "chapter_idea": idea,
                "mounted_rules": mounted_rules or [],
                "traits": traits,
                "beats": None,
                "draft": "",
                "checker_feedback": "",
                "retry_count": 0,
                "planner_model": planner_model,
                "drafter_model": drafter_model,
                "checker_model": checker_model,
            }
        )
    except Exception:
        duration_ms = int((perf_counter() - start) * 1000)
        logger.exception(
            "pipeline.%s.error chapter=%s duration_ms=%s",
            pipeline_id,
            chapter_num,
            duration_ms,
        )
        raise
    duration_ms = int((perf_counter() - start) * 1000)
    logger.info(
        "pipeline.%s.done chapter=%s duration_ms=%s retry_count=%s output_chars=%s",
        pipeline_id,
        chapter_num,
        duration_ms,
        final_state["retry_count"],
        len(final_state["draft"]),
    )
    return final_state["draft"]
