import logging
import json
import re
from time import perf_counter
from pathlib import Path

from core.config import SMART_MODEL
from core.llm_client import generate_text
from core.schemas import ChapterBeatTemplate, NLPBaseTraits, WritingRule

logger = logging.getLogger(__name__)


def draft_chapter(
    chapter_number: int,
    beats: ChapterBeatTemplate,
    character_traits: NLPBaseTraits,
    mounted_rules: list[WritingRule] | None = None,
    checker_feedback: str = "",
    model: str = SMART_MODEL,
    project_name: str = "",
    volume_number: int = 1,
) -> str:
    start = perf_counter()
    logger.info(
        "drafter.start chapter=%s rules=%s model=%s feedback_chars=%s",
        chapter_number,
        len(mounted_rules or []),
        model,
        len(checker_feedback),
    )
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
    previous_context = ""
    core_hook = ""
    world_tone = ""
    project_path = Path(__file__).resolve().parent.parent / "data" / "projects" / str(project_name or "")
    if chapter_number > 1 and project_name:
        chapters_dir = project_path / "chapters"
        previous_index = chapter_number - 1
        candidate_paths = [
            chapters_dir / f"vol_{volume_number}_ch_{previous_index}.md",
            chapters_dir / f"v{volume_number}_c{previous_index}.md",
            chapters_dir / f"chapter_{previous_index}.md",
            chapters_dir / f"ch_{previous_index}.md",
        ]
        target: Path | None = next((path for path in candidate_paths if path.exists()), None)
        if target is None and chapters_dir.exists():
            for path in sorted(chapters_dir.glob("*.md")):
                name = path.name
                matched_v2 = re.match(r"^vol_(\d+)_ch_(\d+)\.md$", name)
                matched_legacy = re.match(r"^v(\d+)_c(\d+)\.md$", name)
                if matched_v2 and int(matched_v2.group(1)) == volume_number and int(matched_v2.group(2)) == previous_index:
                    target = path
                    break
                if matched_legacy and int(matched_legacy.group(1)) == volume_number and int(matched_legacy.group(2)) == previous_index:
                    target = path
                    break
        if target is not None and target.exists():
            try:
                previous_text = target.read_text(encoding="utf-8")
                previous_context = previous_text[-800:]
                print(f"✅ 成功加载上一章内容：{target}，截取最后 800 字...")
            except Exception:
                previous_context = ""
                print("⚠️ 警告：未找到上一章文件，本次为无前文独立生成！")
        else:
            print("⚠️ 警告：未找到上一章文件，本次为无前文独立生成！")
    if project_name:
        meta_file = project_path / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    core_hook = str(meta.get("core_hook", ""))
                    world_tone = str(meta.get("world_tone", ""))
            except Exception:
                core_hook = ""
                world_tone = ""

    system_prompt = (
        "你是一个顶级的网文主笔。请严格按照传入的 4 个剧情节拍，以及主角的心理设定，"
        "将其扩写为完整的章节正文。必须遵循动态挂载法则中的正向指令，并绝对遵守负向约束。"
        "禁止重复上一章已出现的开场和设定说明。\n"
        "[你的任务]\n"
        f"你现在正在撰写本小说的第 {chapter_number} 章。\n"
        "[前文硬链接](极其重要！)\n"
        "上一章的结尾是这样的：\n"
        f"\"{previous_context or '无'}\"\n"
        "[强制规则]\n"
        "1. 你必须紧接着上一章结尾的最后一句话往下写，做到无缝衔接。\n"
        "2. 绝对不要重复上一章已经描写过的场景、设定介绍或主角刚苏醒/发现重生的心理活动。\n"
        "3. 必须直接推进当前章大纲，不得离题。\n"
        "[全书基调约束]\n"
        f"核心卖点：{core_hook or '无'}\n"
        f"世界基调：{world_tone or '无'}\n"
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
        "【你的任务】\n"
        f"你现在正在撰写本小说的第 {chapter_number} 章。\n"
        "【前文硬链接】(极其重要！)\n"
        "上一章的结尾是这样的：\n"
        f"\"{previous_context or '无'}\"\n"
        "【强制规则】\n"
        "1. 你必须紧接着上一章结尾的最后一句话往下写，做到无缝衔接。\n"
        "2. 绝对不要重复上一章已经描写过的场景、设定介绍或主角刚苏醒/发现重生的心理活动。\n"
        "3. 直接推进当前章的大纲内容，禁止回炉重写开头。\n"
        "【当前章_大纲】\n"
        f"{beats_text}\n"
        "【全书基调约束】\n"
        f"核心卖点：{core_hook or '无'}\n"
        f"世界基调：{world_tone or '无'}\n"
        "【战术背包法则】\n"
        f"正向指令：\n{style_text}\n"
        f"负向约束：\n{negative_text}\n"
        "主角当前六层人物设定(JSON)：\n"
        f"{traits_text}\n"
        f"裁判反馈：{checker_feedback or '无'}"
    )
    print(f"🧾 Drafter User Prompt 预览(前500字符)：{user_prompt[:500]}")

    text = generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.8,
    )
    duration_ms = int((perf_counter() - start) * 1000)
    logger.info(
        "drafter.done chapter=%s duration_ms=%s output_chars=%s",
        chapter_number,
        duration_ms,
        len(text),
    )
    return text
