from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.checker import check_consistency
from agents.drafter import draft_chapter
from agents.planner import plan_chapter_beats
from core.config import FAST_MODEL, SMART_MODEL
from core.schemas import ChapterBeatTemplate, NLPBaseTraits


class ChapterState(TypedDict):
    chapter_num: int
    chapter_idea: str
    platform: str
    traits: NLPBaseTraits
    beats: Optional[ChapterBeatTemplate]
    draft: str
    checker_feedback: str
    retry_count: int
    planner_model: str
    drafter_model: str
    checker_model: str


def plan_node(state: ChapterState) -> ChapterState:
    beats = plan_chapter_beats(
        chapter_number=state["chapter_num"],
        chapter_idea=state["chapter_idea"],
        character_traits=state["traits"],
        platform=state["platform"],
        model=state["planner_model"],
    )
    return {**state, "beats": beats}


def draft_node(state: ChapterState) -> ChapterState:
    beats = state["beats"]
    if beats is None:
        raise ValueError("beats 为空，无法生成正文。")
    draft = draft_chapter(
        chapter_number=state["chapter_num"],
        beats=beats,
        character_traits=state["traits"],
        platform=state["platform"],
        checker_feedback=state["checker_feedback"],
        model=state["drafter_model"],
    )
    return {**state, "draft": draft}


def check_node(state: ChapterState) -> ChapterState:
    result = check_consistency(
        draft_text=state["draft"],
        character_traits=state["traits"],
        chapter_idea=state["chapter_idea"],
        model=state["checker_model"],
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
        return "end"
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
    platform: str = "番茄小说",
    planner_model: str = SMART_MODEL,
    drafter_model: str = SMART_MODEL,
    checker_model: str = FAST_MODEL,
) -> str:
    final_state = chapter_pipeline.invoke(
        {
            "chapter_num": chapter_num,
            "chapter_idea": idea,
            "platform": platform,
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
    return final_state["draft"]
