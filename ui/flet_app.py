import json
import logging
import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import flet as ft


# 将项目根目录手动加入搜索路径
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agents.supervisor import (
    brainstorm_ideas,
    derive_initial_traits,
    finalize_proposal,
    generate_book_outline,
    generate_chapter_ideas,
    generate_volume_outline,
)
from agents.quality_extractor import anatomize_fiction_snippet, distill_tutorial_to_rule
from core.config import FAST_MODEL, SMART_MODEL
from core.database import DB_PATH
from core.database import init_character_traits_table
from core.database import load_default_traits
from core.database import save_active_traits
from core.llm_client import generate_structured_data, generate_text
from core.schemas import (
    BookOutline,
    ChapterOutlineList,
    ConceptProposal,
    NLPBaseTraits,
    VolumeOutline,
    WritingRule,
)
from workflow.graph import run_chapter_pipeline

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
logger = logging.getLogger(__name__)

STAGE_CONCEPT = "CONCEPT"
STAGE_OUTLINE = "OUTLINE"
STAGE_CHAPTERS = "CHAPTERS"


class AppState:
    def __init__(self) -> None:
        self.project_name: str = "未命名脑洞"
        self.active_traits: NLPBaseTraits | None = None
        self.concept_proposal: ConceptProposal | None = None
        self.book_outline: BookOutline | None = None
        self.volumes: dict[int, VolumeOutline] = {}
        self.chapter_outlines: dict[int, ChapterOutlineList] = {}
        self.selected_volume_num: int | None = None
        self.volume_outline: VolumeOutline | None = None
        self.chapter_outline: ChapterOutlineList | None = None
        self.chapter_state: dict[str, Any] = {}
        self.chapter_texts: dict[str, str] = {}
        self.chapter_files: dict[str, str] = {}
        self.current_stage: str = STAGE_CONCEPT
        self.chat_history: list[dict[str, str]] = []
        self.quick_options: list[str] = []
        self.characters: dict[str, dict[str, Any]] = {}
        self.world_settings: dict[str, dict[str, Any]] = {}
        self.factions: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.timeline_events: dict[str, dict[str, Any]] = {}
        self.mounted_rules: list[WritingRule] = []
        self.model_entries: list[dict[str, str]] = []
        self.agent_aliases: dict[str, str] = {}
        self.models = {
            "supervisor": SMART_MODEL,
            "planner": SMART_MODEL,
            "drafter": SMART_MODEL,
            "checker": FAST_MODEL,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_path() -> Path:
    return _project_root() / ".env"


def _model_config_path() -> Path:
    return _project_root() / "model_manager.json"


def _data_dir() -> Path:
    return _project_root() / "data"


def _chat_cache_path() -> Path:
    return _data_dir() / "chat_cache.json"


def _projects_dir() -> Path:
    return _data_dir() / "projects"


def _rules_dir() -> Path:
    return _data_dir() / "knowledge_base" / "rules"


def _default_model_entries() -> list[dict[str, str]]:
    return [
        {
            "alias": "smart-default",
            "base_url": "",
            "model_name": SMART_MODEL,
        },
        {
            "alias": "fast-default",
            "base_url": "",
            "model_name": FAST_MODEL,
        },
    ]


def _normalize_model_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in entries:
        normalized.append(
            {
                "alias": str(item.get("alias", "")),
                "base_url": str(item.get("base_url", "")),
                "model_name": str(item.get("model_name", "")),
            }
        )
    return normalized


def _default_agent_aliases() -> dict[str, str]:
    return {
        "supervisor": "smart-default",
        "planner": "smart-default",
        "drafter": "smart-default",
        "checker": "fast-default",
    }


def _load_model_config() -> tuple[list[dict[str, str]], dict[str, str]]:
    path = _model_config_path()
    if not path.exists():
        return _default_model_entries(), _default_agent_aliases()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = raw.get("models", _default_model_entries())
    entries = _normalize_model_entries(raw_entries)
    aliases = raw.get("agent_mapping", _default_agent_aliases())
    if entries != raw_entries:
        _save_model_config(entries, aliases)
    return entries, aliases


def _save_model_config(entries: list[dict[str, str]], aliases: dict[str, str]) -> None:
    normalized_entries = _normalize_model_entries(entries)
    _model_config_path().write_text(
        json.dumps({"models": normalized_entries, "agent_mapping": aliases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_agent_models(entries: list[dict[str, str]], aliases: dict[str, str]) -> dict[str, str]:
    alias_to_model = {item.get("alias", ""): item.get("model_name", "") for item in entries}
    return {
        "supervisor": alias_to_model.get(aliases.get("supervisor", ""), SMART_MODEL) or SMART_MODEL,
        "planner": alias_to_model.get(aliases.get("planner", ""), SMART_MODEL) or SMART_MODEL,
        "drafter": alias_to_model.get(aliases.get("drafter", ""), SMART_MODEL) or SMART_MODEL,
        "checker": alias_to_model.get(aliases.get("checker", ""), FAST_MODEL) or FAST_MODEL,
    }


def _load_env_map() -> dict[str, str]:
    env = {}
    path = _env_path()
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _save_env_map(env: dict[str, str]) -> None:
    content = "\n".join(f"{k}={v}" for k, v in env.items()) + "\n"
    _env_path().write_text(content, encoding="utf-8")


def _ensure_entity_tables() -> None:
    def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str) -> None:
        existing = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_names = {row[1] for row in existing}
        if column_name not in existing_names:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")

    init_character_traits_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS world_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_name TEXT NOT NULL,
                tech_level TEXT NOT NULL,
                power_structure TEXT NOT NULL,
                hidden_rules TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS item_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                origin TEXT NOT NULL,
                current_owner TEXT NOT NULL,
                hidden_power TEXT NOT NULL,
                item_function TEXT NOT NULL DEFAULT '',
                story_hook TEXT NOT NULL DEFAULT ''
            )
            """
        )
        ensure_column(conn, "item_cards", "item_function", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "item_cards", "story_hook", "TEXT NOT NULL DEFAULT ''")
        conn.commit()


def _load_entity_data() -> dict[str, list[dict[str, Any]]]:
    _ensure_entity_tables()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        characters = conn.execute(
            """
            SELECT c.id, c.entity_id, c.version_chapter, c.traits_json, c.update_reason
            FROM character_traits c
            JOIN (
                SELECT entity_id, MAX(version_chapter) AS max_chapter
                FROM character_traits
                GROUP BY entity_id
            ) latest
            ON c.entity_id = latest.entity_id AND c.version_chapter = latest.max_chapter
            ORDER BY c.entity_id
            """
        ).fetchall()
        worlds = conn.execute(
            "SELECT id, region_name, tech_level, power_structure, hidden_rules FROM world_cards ORDER BY id"
        ).fetchall()
        items = conn.execute(
            """
            SELECT id, item_name, origin, current_owner, hidden_power, item_function, story_hook
            FROM item_cards
            ORDER BY id
            """
        ).fetchall()
    parsed_characters: list[dict[str, Any]] = []
    for row in characters:
        traits = json.loads(row["traits_json"])
        parsed_characters.append(
            {
                "id": row["id"],
                "entity_id": row["entity_id"],
                "version_chapter": row["version_chapter"],
                "update_reason": row["update_reason"],
                **traits,
            }
        )
    return {
        "characters": parsed_characters,
        "worlds": [dict(row) for row in worlds],
        "items": [dict(row) for row in items],
    }


def main(page: ft.Page) -> None:
    state = AppState()
    _data_dir().mkdir(parents=True, exist_ok=True)
    page.title = "NovelCraft OS - 赛博编辑部"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE_400)
    page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE_400)
    page.padding = 16
    page.window_width = 1500
    page.window_height = 940
    state.active_traits = load_default_traits()

    outlining_status = ft.Text("状态：等待指令...")
    pipeline_status = ft.Text("状态：等待指令...")
    status_label = ft.Text("系统状态：就绪", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400)
    theme_toggle_btn = ft.IconButton(icon_color=ft.Colors.BLUE_400)
    outline_loading = ft.Row(
        controls=[
            ft.ProgressRing(color=ft.Colors.BLUE_400),
            ft.Text("AI 正在基于策划案推演宏观架构...", color=ft.Colors.BLUE_400),
        ],
        visible=False,
        spacing=12,
    )

    stage_badge_concept = ft.Container(
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
        border_radius=16,
        content=ft.Text("CONCEPT"),
    )
    stage_badge_outline = ft.Container(
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
        border_radius=16,
        content=ft.Text("OUTLINE"),
    )
    stage_badge_chapters = ft.Container(
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
        border_radius=16,
        content=ft.Text("CHAPTERS"),
    )
    stage_indicator = ft.Row(
        controls=[stage_badge_concept, stage_badge_outline, stage_badge_chapters],
        spacing=10,
    )

    volume_num_input = ft.TextField(label="目标卷号", value="1")
    chapter_count_input = ft.TextField(label="目标章节数", value="20")
    chat_input = ft.TextField(
        label="与执行编剧对话",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True, height=470)
    outline_chat_preview = ft.TextField(label="对话缩略", multiline=True, read_only=True, min_lines=10, max_lines=16)
    spark_row = ft.Row(wrap=True, spacing=8)
    quick_option_row = ft.Row(wrap=True, spacing=8)

    proposal_core_hook = ft.TextField(label="核心卖点", multiline=True, min_lines=3, max_lines=5, visible=False, animate_opacity=200)
    proposal_golden_finger = ft.TextField(label="金手指设定", multiline=True, min_lines=3, max_lines=5, visible=False, animate_opacity=200)
    proposal_world_tone = ft.TextField(label="世界基调", multiline=True, min_lines=3, max_lines=5, visible=False, animate_opacity=200)

    book_outline_box = ft.TextField(
        label="全书总纲",
        multiline=True,
        min_lines=15,
        max_lines=30,
        visible=False,
        animate_opacity=200,
        border_color=ft.Colors.BLUE_700,
        focused_border_color=ft.Colors.BLUE_300,
        expand=True,
    )
    volume_outline_box = ft.TextField(
        label="分卷大纲",
        multiline=True,
        min_lines=15,
        max_lines=30,
        visible=False,
        animate_opacity=200,
        border_color=ft.Colors.BLUE_700,
        focused_border_color=ft.Colors.BLUE_300,
        expand=True,
    )
    chapter_outline_box = ft.TextField(label="单章脑洞列表", multiline=True, min_lines=12, max_lines=18, visible=False, animate_opacity=200)
    book_outline_context_box = ft.TextField(
        label="全书总纲上下文（只读）",
        multiline=True,
        min_lines=8,
        max_lines=14,
        read_only=True,
        border_color=ft.Colors.BLUE_700,
        focused_border_color=ft.Colors.BLUE_300,
        expand=True,
    )
    book_outline_mount_status = ft.Text("总纲挂载状态：未挂载", color=ft.Colors.RED_300)
    selected_volume_title = ft.Text("当前未选中分卷", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400)
    volume_cards_view = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    chapter_cards_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    chapter_num_input = ft.TextField(label="当前章节号", value="1")
    chapter_idea_input = ft.TextField(
        label="本章核心脑洞",
        multiline=True,
        min_lines=8,
        max_lines=12,
        hint_text="例如：李四在垃圾场捡到戒指...",
    )
    workshop_volume_picker = ft.Dropdown(label="选择分卷", options=[])
    chapter_picker = ft.Dropdown(label="从章纲选中章节", options=[])
    active_environment_input = ft.TextField(
        label="环境",
        value=state.active_traits.environment if state.active_traits else "",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    active_behavior_input = ft.TextField(
        label="行为",
        value=state.active_traits.behavior if state.active_traits else "",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    active_capability_input = ft.TextField(
        label="能力",
        value=state.active_traits.capability if state.active_traits else "",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    active_values_input = ft.TextField(
        label="价值观",
        value=state.active_traits.values if state.active_traits else "",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    active_identity_input = ft.TextField(
        label="身份",
        value=state.active_traits.identity if state.active_traits else "",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    active_vision_input = ft.TextField(
        label="愿景",
        value=state.active_traits.vision if state.active_traits else "",
        multiline=True,
        min_lines=2,
        max_lines=4,
    )
    entity_category_picker = ft.Dropdown(
        label="分类",
        value="characters",
        options=[
            ft.dropdown.Option("characters", "👥 人物"),
            ft.dropdown.Option("world_settings", "🌍 世界"),
            ft.dropdown.Option("factions", "🏢 势力"),
            ft.dropdown.Option("items", "💎 资产"),
            ft.dropdown.Option("timeline_events", "📅 时间线"),
        ],
    )
    entity_name_input = ft.TextField(label="名称")
    pipeline_output = ft.TextField(
        label="章节正文（Markdown）",
        multiline=True,
        min_lines=24,
        border=ft.InputBorder.NONE,
        content_padding=12,
        expand=True,
    )
    pipeline_progress = ft.ProgressBar(visible=False, color=ft.Colors.BLUE_400)
    workshop_word_count = ft.Text(value="字数: 0", text_align=ft.TextAlign.RIGHT)

    character_cards_box = ft.Column(spacing=10)
    world_cards_box = ft.Column(spacing=10)
    item_cards_box = ft.Column(spacing=10)
    entity_list_view = ft.ListView(expand=True, spacing=6, auto_scroll=True)
    entity_detail_box = ft.Container(expand=True)
    selected_entity_ref: dict[str, str | None] = {"key": None}
    entity_form_field_refs: dict[str, ft.TextField] = {}
    tactical_rules_box = ft.Column(spacing=6)
    rule_positive_box = ft.Column(spacing=6)
    rule_negative_box = ft.Column(spacing=6)
    model_manager_column = ft.Column(spacing=8)
    rule_source_input = ft.TextField(
        label="粘贴爆款神级切片 / 官方干货教程 / 时代背景资料",
        multiline=True,
        min_lines=8,
        max_lines=14,
    )
    rule_category_dropdown = ft.Dropdown(
        label="选择提取/蒸馏的法则分类",
        value="Elements",
        options=[
            ft.dropdown.Option("Elements", "要素解剖"),
            ft.dropdown.Option("Theories", "理论蒸馏"),
            ft.dropdown.Option("Taboos", "避毒红线"),
            ft.dropdown.Option("Formatting", "排版语感"),
            ft.dropdown.Option("Lore", "时代考据"),
            ft.dropdown.Option("Tropes", "专属桥段"),
        ],
    )
    rule_group_default_options = ["通用法则", "都市重生类", "科幻修真类", "《网吧AI》专属"]
    rule_group_input = ft.TextField(label="🏷️ 所属分组 (Group)", value="通用法则 (General)")
    element_target_input = ft.TextField(label="要素解剖目标（仅 Elements 使用）", value="人物")
    rule_name_preview = ft.Text("法则名称：-", color=ft.Colors.BLUE_300)
    save_rule_btn = ft.Button("💾 保存到本地知识库", disabled=True)

    env_map = _load_env_map()
    loaded_entries, loaded_aliases = _load_model_config()
    state.model_entries = loaded_entries
    state.agent_aliases = loaded_aliases
    state.models = _resolve_agent_models(state.model_entries, state.agent_aliases)
    api_key_input = ft.TextField(
        label="DEEPSEEK_API_KEY",
        value=env_map.get("DEEPSEEK_API_KEY", ""),
        password=True,
        can_reveal_password=True,
    )
    supervisor_mapping_dropdown = ft.Dropdown(label="Supervisor 映射")
    planner_mapping_dropdown = ft.Dropdown(label="Planner 映射")
    drafter_mapping_dropdown = ft.Dropdown(label="Drafter 映射")
    checker_mapping_dropdown = ft.Dropdown(label="Checker 映射")
    settings_status = ft.Text("状态：等待保存...")
    local_rules_cache: list[WritingRule] = []
    latest_rule_ref: dict[str, WritingRule | None] = {"rule": None}
    stage_jump_ref: dict[str, ft.Dropdown | None] = {"dropdown": None}
    stage_nav_ref: dict[str, ft.Button | None] = {"concept": None, "outline": None, "chapters": None}
    archive_filter_ref: dict[str, ft.Dropdown | None] = {"dropdown": None}
    archive_option_map: dict[str, str] = {}
    loading_text = ft.Text("正在处理...", color=ft.Colors.BLUE_300)
    loading_dialog = ft.AlertDialog(
        modal=True,
        content=ft.Row(
            controls=[ft.ProgressRing(width=22, height=22, color=ft.Colors.BLUE_400), loading_text],
            spacing=12,
            tight=True,
        ),
    )

    def set_status(message: str) -> None:
        status_label.value = f"系统状态：{message}"

    def mounted_rules_suffix() -> str:
        count = len(state.mounted_rules)
        return f"（已挂载 {count} 条战术法则）" if count > 0 else ""

    def with_rules_hint(message: str) -> str:
        suffix = mounted_rules_suffix()
        return f"{message}{suffix}" if suffix else message

    def show_loading(current_page: ft.Page, message: str = "正在处理...") -> None:
        loading_text.value = message
        if loading_dialog not in current_page.overlay:
            current_page.overlay.append(loading_dialog)
        loading_dialog.open = True
        current_page.update()

    def hide_loading(current_page: ft.Page) -> None:
        loading_dialog.open = False
        current_page.update()

    def show_info_dialog(message: str, title: str = "提示") -> None:
        def on_close_dialog(_: ft.ControlEvent) -> None:
            dialog.open = False
            page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Text(message),
            actions=[ft.TextButton("知道了", on_click=on_close_dialog)],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    def rule_category_label(category: str) -> str:
        mapping = {
            "Elements": "要素解剖",
            "Theories": "理论蒸馏",
            "Taboos": "避毒红线",
            "Formatting": "排版语感",
            "Lore": "时代考据",
            "Tropes": "专属桥段",
        }
        return mapping.get(category, category)

    def rule_category_icon(category: str) -> str:
        mapping = {
            "Elements": "🧩",
            "Theories": "📘",
            "Taboos": "⛔",
            "Formatting": "✍️",
            "Lore": "📜",
            "Tropes": "🎭",
        }
        return mapping.get(category, "📌")

    def render_rule_result(rule: WritingRule | None) -> None:
        rule_positive_box.controls.clear()
        rule_negative_box.controls.clear()
        if rule is None:
            rule_name_preview.value = "法则名称：-"
            save_rule_btn.disabled = True
            return
        rule_group_input.value = rule.group or "通用法则 (General)"
        rule_name_preview.value = f"法则名称：{rule.rule_name}（{rule_category_label(rule.category)}）"
        rule_positive_box.controls.extend(ft.Text(f"✅ {item}", color=ft.Colors.GREEN_300) for item in rule.positive_instructions)
        rule_negative_box.controls.extend(ft.Text(f"⛔ {item}", color=ft.Colors.RED_300) for item in rule.negative_constraints)
        save_rule_btn.disabled = False

    def refresh_tactical_backpack() -> None:
        tactical_rules_box.controls.clear()
        mounted_ids = {item.rule_id for item in state.mounted_rules}
        grouped_rules: dict[str, list[WritingRule]] = {}
        for rule in local_rules_cache:
            group_name = (rule.group or "通用法则 (General)").strip() or "通用法则 (General)"
            grouped_rules.setdefault(group_name, []).append(rule)
        for group_name in sorted(grouped_rules.keys()):
            controls: list[ft.Control] = []
            for rule in grouped_rules[group_name]:
                def on_toggle(event: ft.ControlEvent, item: WritingRule = rule) -> None:
                    checked = bool(event.control.value)
                    existing = {entry.rule_id: entry for entry in state.mounted_rules}
                    if checked:
                        existing[item.rule_id] = item
                    elif item.rule_id in existing:
                        existing.pop(item.rule_id)
                    state.mounted_rules = list(existing.values())
                    save_full_archive()
                    set_status(f"战术背包已更新，共挂载 {len(state.mounted_rules)} 条法则。")
                    page.update()

                controls.append(
                    ft.Checkbox(
                        value=rule.rule_id in mounted_ids,
                        label=(
                            f"{rule_category_icon(rule.category)} {rule.rule_name} "
                            f"（{rule_category_label(rule.category)} / {rule.applicable_stage}）"
                        ),
                        on_change=on_toggle,
                    )
                )
            tactical_rules_box.controls.append(
                ft.ExpansionTile(
                    title=ft.Text(f"📁 【{group_name}】（包含 {len(controls)} 条法则）"),
                    controls=controls,
                )
            )
        if not tactical_rules_box.controls:
            tactical_rules_box.controls.append(ft.Text("暂无法则，请先在法则炼金炉中提炼并保存。"))

    def reload_local_rules() -> None:
        nonlocal local_rules_cache
        local_rules_cache = load_local_rules()
        mounted_ids = {item.rule_id for item in state.mounted_rules}
        local_map = {item.rule_id: item for item in local_rules_cache}
        state.mounted_rules = [local_map[item] for item in mounted_ids if item in local_map]
        refresh_tactical_backpack()

    def on_open_rule_mount_panel(_: ft.ControlEvent) -> None:
        reload_local_rules()

        def on_refresh_mount_rules(_: ft.ControlEvent) -> None:
            reload_local_rules()
            page.update()

        mount_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("🎒 配置战术背包（挂载法则）"),
            content=ft.Container(
                width=980,
                height=650,
                content=ft.Column(
                    controls=[
                        ft.Row([ft.OutlinedButton("刷新本地法则", on_click=on_refresh_mount_rules)], alignment=ft.MainAxisAlignment.END),
                        tactical_rules_box,
                    ],
                    spacing=10,
                    expand=True,
                    scroll=ft.ScrollMode.AUTO,
                ),
            ),
            actions=[ft.TextButton("关闭", on_click=lambda _: setattr(mount_dialog, "open", False))],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(mount_dialog)
        mount_dialog.open = True
        page.update()

    def on_extract_rule(_: ft.ControlEvent) -> None:
        source_text = (rule_source_input.value or "").strip()
        if not source_text:
            show_info_dialog("请先粘贴要提炼的文本素材。")
            return
        target_category = rule_category_dropdown.value or "Elements"
        show_loading(page, with_rules_hint("正在深度提炼写作法则..."))
        try:
            if target_category == "Elements":
                result = anatomize_fiction_snippet(
                    text=source_text,
                    element_target=(element_target_input.value or "综合").strip(),
                    model=state.models["supervisor"],
                )
            else:
                result = distill_tutorial_to_rule(
                    text=source_text,
                    category_target=target_category,
                    model=state.models["supervisor"],
                )
            latest_rule_ref["rule"] = result
            render_rule_result(result)
            set_status("法则提炼完成。")
        except Exception as exc:
            latest_rule_ref["rule"] = None
            render_rule_result(None)
            show_info_dialog(f"法则提炼失败：{exc}")
        finally:
            hide_loading(page)
            page.update()

    def on_save_rule(_: ft.ControlEvent) -> None:
        rule = latest_rule_ref.get("rule")
        if rule is None:
            show_info_dialog("暂无可保存法则。")
            return
        if not rule.rule_id.strip():
            rule = rule.model_copy(update={"rule_id": datetime.now().strftime("rule_%Y%m%d_%H%M%S")})
        group_value = (rule_group_input.value or "").strip() or "通用法则 (General)"
        rule = rule.model_copy(update={"group": group_value})
        save_writing_rule(rule)
        latest_rule_ref["rule"] = rule
        reload_local_rules()
        set_status(f"法则已保存：{rule.rule_name}")
        page.update()

    def on_select_rule_group(_: ft.ControlEvent, group_name: str) -> None:
        rule_group_input.value = group_name
        page.update()

    def get_project_name() -> str:
        if state.project_name.strip():
            return state.project_name.strip()
        if state.book_outline and state.book_outline.book_title.strip():
            return state.book_outline.book_title.strip()
        return "未命名脑洞"

    def _project_dir_path(project_name: str | None = None) -> Path:
        _projects_dir().mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_archive_filename(project_name or get_project_name())
        return _projects_dir() / safe_name

    def _project_meta_path(project_name: str | None = None) -> Path:
        return _project_dir_path(project_name) / "meta.json"

    def _project_chapters_dir(project_name: str | None = None) -> Path:
        return _project_dir_path(project_name) / "chapters"

    def _chapter_markdown_filename(volume_num: int, chapter_num: int) -> str:
        return f"vol_{volume_num}_ch_{chapter_num}.md"

    def _legacy_chapter_markdown_filename(volume_num: int, chapter_num: int) -> str:
        return f"v{volume_num}_c{chapter_num}.md"

    def _chapter_relative_path(volume_num: int, chapter_num: int) -> str:
        return f"chapters/{_chapter_markdown_filename(volume_num, chapter_num)}"

    def _legacy_chapter_relative_path(volume_num: int, chapter_num: int) -> str:
        return f"chapters/{_legacy_chapter_markdown_filename(volume_num, chapter_num)}"

    def _write_chapter_markdown(volume_num: int, chapter_num: int, text: str) -> str:
        chapters_dir = _project_chapters_dir(state.project_name)
        chapters_dir.mkdir(parents=True, exist_ok=True)
        filename = _chapter_markdown_filename(volume_num, chapter_num)
        chapter_path = chapters_dir / filename
        chapter_path.write_text(text, encoding="utf-8")
        return _chapter_relative_path(volume_num, chapter_num)

    def _read_chapter_markdown(relative_path: str) -> str:
        path = _project_dir_path(state.project_name) / relative_path
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _parse_chapter_text_key(key: str) -> tuple[int, int] | None:
        parts = str(key).split(":")
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return None

    def save_chapter_text(text_key: str, text: str) -> None:
        parsed = _parse_chapter_text_key(text_key)
        if parsed is None:
            return
        volume_num, chapter_num = parsed
        relative_path = _write_chapter_markdown(volume_num, chapter_num, text)
        state.chapter_files[text_key] = relative_path
        state.chapter_texts[text_key] = text

    def _legacy_payload_to_meta_payload(payload: dict[str, Any], project_name: str) -> dict[str, Any]:
        snapshot = payload.get("app_state") if isinstance(payload.get("app_state"), dict) else payload
        app_state = dict(snapshot) if isinstance(snapshot, dict) else {}
        chapter_texts_data = app_state.get("chapter_texts")
        chapter_files: dict[str, str] = {}
        if isinstance(chapter_texts_data, dict):
            for raw_key, raw_text in chapter_texts_data.items():
                key = str(raw_key)
                text = str(raw_text)
                parsed = _parse_chapter_text_key(key)
                if parsed is None:
                    continue
                volume_num, chapter_num = parsed
                filename = _chapter_markdown_filename(volume_num, chapter_num)
                chapter_files[key] = f"chapters/{filename}"
                chapter_path = _project_chapters_dir(project_name) / filename
                chapter_path.parent.mkdir(parents=True, exist_ok=True)
                chapter_path.write_text(text, encoding="utf-8")
        app_state["chapter_files"] = chapter_files
        app_state.pop("chapter_texts", None)
        return {
            "app_state": app_state,
            "project_name": str(payload.get("project_name") or project_name),
            "save_stage": str(payload.get("save_stage") or app_state.get("save_stage") or STAGE_CONCEPT),
            "saved_at": str(payload.get("saved_at") or datetime.now().isoformat()),
        }

    def migrate_legacy_project_json(legacy_json_path: Path) -> Path:
        try:
            payload = json.loads(legacy_json_path.read_text(encoding="utf-8"))
        except Exception:
            return legacy_json_path
        project_name = str(payload.get("project_name") or legacy_json_path.stem)
        project_dir = _project_dir_path(project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        meta_path = project_dir / "meta.json"
        migrated_payload = _legacy_payload_to_meta_payload(payload, project_name)
        meta_path.write_text(json.dumps(migrated_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            legacy_json_path.unlink()
        except Exception:
            pass
        return meta_path

    def resolve_project_meta_path(project_ref: Path) -> Path:
        if project_ref.is_dir():
            return project_ref / "meta.json"
        if project_ref.suffix.lower() == ".json":
            return migrate_legacy_project_json(project_ref)
        return project_ref

    async def focus_control(control: ft.Control) -> None:
        await control.focus()

    def request_focus(control: ft.Control) -> None:
        page.run_task(focus_control, control)

    def save_chat_to_cache() -> None:
        _data_dir().mkdir(parents=True, exist_ok=True)
        _chat_cache_path().write_text(
            json.dumps(state.chat_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def sanitize_archive_filename(name: str) -> str:
        cleaned = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in name).strip()
        return cleaned or "未命名作品"

    def save_full_archive() -> Path:
        state.project_name = get_project_name()
        project_dir = _project_dir_path(state.project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        _project_chapters_dir(state.project_name).mkdir(parents=True, exist_ok=True)
        project_path = _project_meta_path(state.project_name)
        volumes_data = {str(num): item.model_dump() for num, item in state.volumes.items()}
        chapters_data = {str(num): item.model_dump() for num, item in state.chapter_outlines.items()}
        app_state_snapshot = {
            "chat_history": state.chat_history,
            "quick_options": state.quick_options,
            "current_concept": state.concept_proposal.model_dump() if state.concept_proposal else None,
            "book_outline": state.book_outline.model_dump() if state.book_outline else None,
            "volumes": volumes_data,
            "chapter_outlines": chapters_data,
            "chapter_files": state.chapter_files,
            "selected_volume_num": state.selected_volume_num,
            "active_traits": state.active_traits.model_dump() if state.active_traits else None,
            "characters": state.characters,
            "world_settings": state.world_settings,
            "factions": state.factions,
            "items": state.items,
            "timeline_events": state.timeline_events,
            "mounted_rule_ids": [item.rule_id for item in state.mounted_rules],
        }
        payload = {
            "app_state": app_state_snapshot,
            "project_name": state.project_name,
            "save_stage": state.current_stage,
            "chat_history": app_state_snapshot["chat_history"],
            "quick_options": app_state_snapshot["quick_options"],
            "current_concept": app_state_snapshot["current_concept"],
            "book_outline": app_state_snapshot["book_outline"],
            "chapter_outlines": app_state_snapshot["chapter_outlines"],
            "chapter_files": app_state_snapshot["chapter_files"],
            "volumes": app_state_snapshot["volumes"],
            "active_traits": app_state_snapshot["active_traits"],
            "characters": app_state_snapshot["characters"],
            "world_settings": app_state_snapshot["world_settings"],
            "factions": app_state_snapshot["factions"],
            "items": app_state_snapshot["items"],
            "timeline_events": app_state_snapshot["timeline_events"],
            "mounted_rule_ids": app_state_snapshot["mounted_rule_ids"],
            "saved_at": datetime.now().isoformat(),
        }
        project_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return project_path

    def list_projects() -> list[tuple[Path, str, str]]:
        _projects_dir().mkdir(parents=True, exist_ok=True)
        result: list[tuple[Path, str, str]] = []
        entries: list[Path] = list(_projects_dir().iterdir())
        for path in sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True):
            if path.is_dir():
                meta_path = path / "meta.json"
                if not meta_path.exists():
                    continue
            elif path.suffix.lower() == ".json":
                meta_path = migrate_legacy_project_json(path)
                path = meta_path.parent
            else:
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            project_name = str(payload.get("project_name") or path.name)
            saved_at = str(payload.get("saved_at") or datetime.fromtimestamp(meta_path.stat().st_mtime).isoformat())
            result.append((path, project_name, saved_at))
        return result

    def save_writing_rule(rule: WritingRule) -> Path:
        _rules_dir().mkdir(parents=True, exist_ok=True)
        file_name = f"{sanitize_archive_filename(rule.rule_id)}.json"
        path = _rules_dir() / file_name
        path.write_text(json.dumps(rule.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_local_rules() -> list[WritingRule]:
        _rules_dir().mkdir(parents=True, exist_ok=True)
        rules: list[WritingRule] = []
        for path in sorted(_rules_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and (not str(payload.get("group", "")).strip()):
                    payload["group"] = "通用法则 (General)"
                rules.append(WritingRule.model_validate(payload))
            except Exception:
                continue
        return rules

    def load_chat_from_cache() -> bool:
        path = _chat_cache_path()
        if not path.exists():
            return False
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, list):
            return False
        loaded: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                loaded.append({"role": role, "content": content})
        state.chat_history = loaded
        return bool(state.chat_history)

    def save_chat_markdown() -> Path:
        _data_dir().mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _data_dir() / f"history_log_{timestamp}.md"
        lines = ["# 灵感沙龙对话记录", ""]
        for item in state.chat_history:
            speaker = "主编" if item["role"] == "assistant" else "你"
            lines.append(f"## {speaker}")
            lines.append("")
            lines.append(item["content"])
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def scan_archives() -> list[tuple[str, str, str]]:
        _projects_dir().mkdir(parents=True, exist_ok=True)
        result: list[tuple[str, str, str]] = []
        for project_path, _, _ in list_projects():
            meta_path = project_path / "meta.json"
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            project_name = str(payload.get("project_name") or "未命名脑洞")
            saved_at = str(payload.get("saved_at") or "")
            raw_stage = str(payload.get("save_stage") or "CONCEPT").upper()
            stage = raw_stage if raw_stage in {STAGE_CONCEPT, STAGE_OUTLINE, STAGE_CHAPTERS} else STAGE_CONCEPT
            time_text = saved_at[0:19].replace("T", " ") if saved_at else "未知时间"
            label = f"【{project_name}】 - {time_text} - {stage}"
            result.append((project_path.name, label, stage))
        return result

    def sync_theme_toggle() -> None:
        if page.theme_mode == ft.ThemeMode.DARK:
            theme_toggle_btn.icon = ft.Icons.LIGHT_MODE_OUTLINED
            theme_toggle_btn.tooltip = "切换为明亮模式"
        else:
            theme_toggle_btn.icon = ft.Icons.DARK_MODE_OUTLINED
            theme_toggle_btn.tooltip = "切换为暗黑模式"

    def on_toggle_theme(_: ft.ControlEvent) -> None:
        page.theme_mode = ft.ThemeMode.LIGHT if page.theme_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK
        sync_theme_toggle()
        page.update()

    theme_toggle_btn.on_click = on_toggle_theme

    def sync_mapping_dropdowns() -> None:
        alias_options = [ft.dropdown.Option(item.get("alias", "")) for item in state.model_entries if item.get("alias")]
        supervisor_mapping_dropdown.options = alias_options
        planner_mapping_dropdown.options = alias_options
        drafter_mapping_dropdown.options = alias_options
        checker_mapping_dropdown.options = alias_options
        supervisor_mapping_dropdown.value = state.agent_aliases.get("supervisor")
        planner_mapping_dropdown.value = state.agent_aliases.get("planner")
        drafter_mapping_dropdown.value = state.agent_aliases.get("drafter")
        checker_mapping_dropdown.value = state.agent_aliases.get("checker")

    def render_model_manager() -> None:
        controls: list[ft.Control] = []
        for idx, entry in enumerate(state.model_entries):
            alias_field = ft.TextField(label="模型别名", value=entry.get("alias", ""))
            base_url_field = ft.TextField(label="Base URL", value=entry.get("base_url", ""))
            model_name_field = ft.TextField(label="模型名称", value=entry.get("model_name", ""))

            def sync_entry(
                _: ft.ControlEvent,
                index: int = idx,
                alias_ctrl: ft.TextField = alias_field,
                base_ctrl: ft.TextField = base_url_field,
                model_ctrl: ft.TextField = model_name_field,
            ) -> None:
                state.model_entries[index] = {
                    "alias": alias_ctrl.value or "",
                    "base_url": base_ctrl.value or "",
                    "model_name": model_ctrl.value or "",
                }
                _save_model_config(state.model_entries, state.agent_aliases)
                state.models = _resolve_agent_models(state.model_entries, state.agent_aliases)
                sync_mapping_dropdowns()
                settings_status.value = "状态：模型列表已实时保存。"
                page.update()

            alias_field.on_blur = sync_entry
            base_url_field.on_blur = sync_entry
            model_name_field.on_blur = sync_entry

            controls.append(
                ft.Container(
                    border=ft.Border.all(1, ft.Colors.BLUE_GREY_700),
                    border_radius=8,
                    padding=10,
                    content=ft.Column(
                        controls=[
                            ft.Row(
                                [
                                    alias_field,
                                    base_url_field,
                                    model_name_field,
                                    ft.IconButton(
                                        icon=ft.Icons.DELETE,
                                        on_click=lambda _, index=idx: on_delete_model_entry(index),
                                        icon_color=ft.Colors.RED_300,
                                    ),
                                ]
                            )
                        ],
                        spacing=8,
                    ),
                )
            )
        model_manager_column.controls = controls

    def on_delete_model_entry(index: int) -> None:
        if len(state.model_entries) <= 1:
            settings_status.value = "状态：至少保留一个模型配置。"
            page.update()
            return
        if 0 <= index < len(state.model_entries):
            removed_alias = state.model_entries[index].get("alias", "")
            state.model_entries.pop(index)
            for agent, alias in list(state.agent_aliases.items()):
                if alias == removed_alias:
                    state.agent_aliases[agent] = state.model_entries[0].get("alias", "")
            _save_model_config(state.model_entries, state.agent_aliases)
            state.models = _resolve_agent_models(state.model_entries, state.agent_aliases)
            render_model_manager()
            sync_mapping_dropdowns()
            settings_status.value = "状态：模型配置已删除并保存。"
            page.update()

    def on_add_model_entry(_: ft.ControlEvent) -> None:
        state.model_entries.append(
            {
                "alias": f"model-{len(state.model_entries) + 1}",
                "base_url": "",
                "model_name": SMART_MODEL,
            }
        )
        _save_model_config(state.model_entries, state.agent_aliases)
        render_model_manager()
        sync_mapping_dropdowns()
        settings_status.value = "状态：模型配置已新增。"
        page.update()

    def on_mapping_change(_: ft.ControlEvent) -> None:
        state.agent_aliases["supervisor"] = supervisor_mapping_dropdown.value or ""
        state.agent_aliases["planner"] = planner_mapping_dropdown.value or ""
        state.agent_aliases["drafter"] = drafter_mapping_dropdown.value or ""
        state.agent_aliases["checker"] = checker_mapping_dropdown.value or ""
        _save_model_config(state.model_entries, state.agent_aliases)
        state.models = _resolve_agent_models(state.model_entries, state.agent_aliases)
        settings_status.value = "状态：Agent 映射已保存。"
        page.update()

    def on_api_key_blur(_: ft.ControlEvent) -> None:
        env = _load_env_map()
        env["DEEPSEEK_API_KEY"] = api_key_input.value or ""
        _save_env_map(env)
        os.environ["DEEPSEEK_API_KEY"] = api_key_input.value or ""
        settings_status.value = "状态：.env 中 DEEPSEEK_API_KEY 已实时保存。"
        page.update()

    supervisor_mapping_dropdown.on_change = on_mapping_change
    planner_mapping_dropdown.on_change = on_mapping_change
    drafter_mapping_dropdown.on_change = on_mapping_change
    checker_mapping_dropdown.on_change = on_mapping_change
    api_key_input.on_blur = on_api_key_blur

    def update_workshop_word_count(text: str) -> None:
        workshop_word_count.value = f"字数: {len(text)}"

    def _get_selected_workshop_volume() -> int | None:
        selected = workshop_volume_picker.value
        if not selected:
            return None
        try:
            return int(selected)
        except ValueError:
            return None

    def _get_selected_workshop_chapter() -> int | None:
        selected = chapter_picker.value
        if not selected:
            return None
        try:
            return int(selected)
        except ValueError:
            return None

    def _load_chapter_content_for_workshop(volume_num: int, chapter_num: int) -> str:
        text_key = f"{volume_num}:{chapter_num}"
        deduced_relative = _chapter_relative_path(volume_num, chapter_num)
        loaded_text = _read_chapter_markdown(deduced_relative)
        used_relative = deduced_relative if loaded_text else ""
        if not loaded_text:
            legacy_relative = _legacy_chapter_relative_path(volume_num, chapter_num)
            loaded_text = _read_chapter_markdown(legacy_relative)
            used_relative = legacy_relative if loaded_text else ""
        if not loaded_text:
            mapped_relative = state.chapter_files.get(text_key, "")
            if mapped_relative and mapped_relative not in {deduced_relative, _legacy_chapter_relative_path(volume_num, chapter_num)}:
                loaded_text = _read_chapter_markdown(mapped_relative)
                used_relative = mapped_relative if loaded_text else ""
        if loaded_text:
            state.chapter_texts[text_key] = loaded_text
            state.chapter_files[text_key] = used_relative or deduced_relative
        return loaded_text

    def sync_chapter_picker() -> None:
        volume_options: list[ft.dropdown.Option] = []
        for volume_num in sorted(state.volumes.keys()):
            outline = state.volumes.get(volume_num)
            title = outline.volume_title if outline else ""
            volume_options.append(ft.dropdown.Option(key=str(volume_num), text=f"第{volume_num}卷：{title}"))
        workshop_volume_picker.options = volume_options
        selected_volume = state.selected_volume_num if state.selected_volume_num in state.volumes else None
        workshop_volume_picker.value = str(selected_volume) if selected_volume is not None else None
        options: list[ft.dropdown.Option] = []
        chapter_outline = state.chapter_outlines.get(selected_volume) if selected_volume is not None else None
        if chapter_outline:
            for chapter in chapter_outline.chapters:
                chapter_number = int(chapter.get("chapter_number", 0))
                core_event = str(chapter.get("core_event", ""))
                options.append(ft.dropdown.Option(key=str(chapter_number), text=f"第{chapter_number}章：{core_event}"))
        chapter_picker.options = options
        if chapter_picker.value not in {opt.key for opt in options}:
            chapter_picker.value = None

    def sync_proposal_board() -> None:
        if state.concept_proposal is None:
            proposal_core_hook.value = ""
            proposal_golden_finger.value = ""
            proposal_world_tone.value = ""
            proposal_core_hook.visible = False
            proposal_golden_finger.visible = False
            proposal_world_tone.visible = False
            return
        proposal_core_hook.value = state.concept_proposal.core_hook
        proposal_golden_finger.value = state.concept_proposal.golden_finger
        proposal_world_tone.value = state.concept_proposal.world_tone
        proposal_core_hook.visible = True
        proposal_golden_finger.visible = True
        proposal_world_tone.visible = True

    def set_stage(stage: str) -> None:
        state.current_stage = stage
        active_bg = ft.Colors.BLUE_400
        active_color = ft.Colors.WHITE
        idle_bg = ft.Colors.BLUE_GREY_900
        idle_color = ft.Colors.WHITE
        stage_badge_concept.bgcolor = active_bg if stage == STAGE_CONCEPT else idle_bg
        stage_badge_concept.content.color = active_color if stage == STAGE_CONCEPT else idle_color
        stage_badge_outline.bgcolor = active_bg if stage == STAGE_OUTLINE else idle_bg
        stage_badge_outline.content.color = active_color if stage == STAGE_OUTLINE else idle_color
        stage_badge_chapters.bgcolor = active_bg if stage == STAGE_CHAPTERS else idle_bg
        stage_badge_chapters.content.color = active_color if stage == STAGE_CHAPTERS else idle_color
        concept_panel.visible = stage == STAGE_CONCEPT
        outline_panel.visible = stage == STAGE_OUTLINE
        chapters_panel.visible = stage == STAGE_CHAPTERS
        for key, btn in stage_nav_ref.items():
            if btn is None:
                continue
            active = (key == "concept" and stage == STAGE_CONCEPT) or (key == "outline" and stage == STAGE_OUTLINE) or (
                key == "chapters" and stage == STAGE_CHAPTERS
            )
            btn.style = ft.ButtonStyle(
                bgcolor=ft.Colors.BLUE_700 if active else ft.Colors.BLUE_GREY_900,
                color=ft.Colors.WHITE,
            )
        if stage_jump_ref["dropdown"] is not None:
            stage_jump_ref["dropdown"].value = stage
        if archive_filter_ref["dropdown"] is not None:
            archive_filter_ref["dropdown"].value = stage
            refresh_archive_options()
        if stage == STAGE_OUTLINE:
            book_outline_box.visible = True
            volume_outline_box.visible = True

    def on_select_volume(volume_num: int, refresh_page: bool = True) -> None:
        state.selected_volume_num = volume_num
        state.volume_outline = state.volumes.get(volume_num)
        state.chapter_outline = state.chapter_outlines.get(volume_num)
        render_volume_cards()
        render_outline_boards()
        chapter_outline_box.value = state.chapter_outline.model_dump_json(indent=2) if state.chapter_outline else ""
        sync_chapter_picker()
        render_chapter_cards()
        if refresh_page:
            page.update()

    def render_volume_cards() -> None:
        cards: list[ft.Control] = []
        for volume_num in sorted(state.volumes.keys()):
            outline = state.volumes[volume_num]
            is_selected = state.selected_volume_num == volume_num

            def on_pick(_: ft.ControlEvent, number: int = volume_num) -> None:
                on_select_volume(number)

            cards.append(
                ft.Card(
                    content=ft.Container(
                        bgcolor=ft.Colors.BLUE_GREY_800 if is_selected else ft.Colors.BLUE_GREY_900,
                        padding=10,
                        content=ft.Row(
                            controls=[
                                ft.Text(f"第 {volume_num} 卷：{outline.volume_title}", weight=ft.FontWeight.BOLD),
                                ft.IconButton(icon=ft.Icons.VISIBILITY, tooltip="查看/编辑该卷", on_click=on_pick),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                    )
                )
            )
        volume_cards_view.controls = cards

    def render_chapter_cards() -> None:
        if state.chapter_outline is None:
            chapter_cards_view.controls = []
            return
        cards: list[ft.Control] = []
        for chapter in state.chapter_outline.chapters:
            chapter_number = int(chapter.get("chapter_number", 0))
            core_event = str(chapter.get("core_event", ""))

            def on_send_to_workshop(_: ft.ControlEvent, number: int = chapter_number, idea: str = core_event) -> None:
                chapter_num_input.value = str(number)
                chapter_idea_input.value = idea
                if state.selected_volume_num is not None:
                    workshop_volume_picker.value = str(state.selected_volume_num)
                sync_chapter_picker()
                chapter_picker.value = str(number)
                on_chapter_pick(_)
                pipeline_status.value = f"状态：已从章节 {number} 发送到写作工坊。"
                logger.info("ui.send_to_workshop chapter=%s idea_chars=%s", number, len(idea))
                if tabs_ref["tabs"] is not None:
                    tabs_ref["tabs"].selected_index = 2
                page.update()

            cards.append(
                ft.Card(
                    content=ft.Container(
                        padding=12,
                        content=ft.Column(
                            controls=[
                                ft.Row(
                                    controls=[
                                        ft.Text(f"第 {chapter_number} 章", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                                        ft.IconButton(
                                            icon=ft.Icons.EDIT,
                                            tooltip="发送到写作工坊",
                                            on_click=on_send_to_workshop,
                                            icon_color=ft.Colors.BLUE_400,
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                ),
                                ft.Text(core_event),
                            ],
                            spacing=8,
                        ),
                    )
                )
            )
        chapter_cards_view.controls = cards

    def sync_outline_visibility() -> None:
        book_outline_box.visible = bool(book_outline_box.value.strip())
        volume_outline_box.visible = bool(volume_outline_box.value.strip())
        chapter_outline_box.visible = bool(chapter_outline_box.value.strip())

    def format_book_outline(outline: BookOutline) -> str:
        volumes = "\n".join(f"  - {item}" for item in outline.planned_volumes) or "  - （待补充）"
        return (
            f"📚 【书名】：《{outline.book_title}》 (预计：{outline.target_word_count})\n\n"
            f"✨ 【一句话简介】：{outline.logline}\n\n"
            f"⚙️ 【核心力量体系】：{outline.core_power_system}\n\n"
            f"🧭 【主线推进】：{outline.main_storyline}\n\n"
            f"🎯 【结局愿景】：{outline.ending_vision}\n\n"
            f"📦 【分卷规划】：\n{volumes}\n\n"
            "🧮 【字数-章节基准】：1章=2000字（1万字≈5章）\n\n"
            f"📈 【节奏设计（含字数/章位锚点）】：\n{outline.pacing_design}"
        )

    def format_volume_outline(outline: VolumeOutline) -> str:
        factions = "\n".join(f"  - {item}" for item in outline.new_factions) or "  - （待补充）"
        subplots = "\n".join(f"  - {item}" for item in outline.key_subplots) or "  - （待补充）"
        return (
            f"📘 【卷号】第{outline.volume_number}卷\n"
            f"🏷️ 【卷名】：{outline.volume_title}\n"
            f"📏 【预计字数】：{outline.estimated_word_count}\n\n"
            f"🔥 【核心冲突】：{outline.core_conflict}\n\n"
            f"🛡️ 【新增势力】：\n{factions}\n\n"
            "🧮 【字数-章节基准】：1章=2000字（1万字≈5章）\n\n"
            f"🧩 【核心支线（含章节区间与字数）】：\n{subplots}"
        )

    def render_outline_boards() -> None:
        book_outline_box.value = format_book_outline(state.book_outline) if state.book_outline else ""
        volume_outline_box.value = format_volume_outline(state.volume_outline) if state.volume_outline else ""
        book_outline_context_box.value = format_book_outline(state.book_outline) if state.book_outline else ""
        if state.volume_outline is None:
            selected_volume_title.value = "当前未选中分卷"
        else:
            selected_volume_title.value = f"当前分卷：第 {state.volume_outline.volume_number} 卷《{state.volume_outline.volume_title}》"

    def refresh_all_views(current_page: ft.Page, app_state: AppState) -> None:
        refresh_chat_view(persist=False)
        render_quick_options()
        sync_proposal_board()
        sync_active_traits_inputs()
        if app_state.volumes:
            if app_state.selected_volume_num is None or app_state.selected_volume_num not in app_state.volumes:
                app_state.selected_volume_num = sorted(app_state.volumes.keys())[-1]
            app_state.volume_outline = app_state.volumes.get(app_state.selected_volume_num)
            app_state.chapter_outline = app_state.chapter_outlines.get(app_state.selected_volume_num)
        else:
            app_state.selected_volume_num = None
            app_state.volume_outline = None
            app_state.chapter_outline = None
        render_volume_cards()
        render_outline_boards()
        if app_state.book_outline is None:
            book_outline_mount_status.value = "总纲挂载状态：未挂载（请先生成或加载全书总纲）"
            book_outline_mount_status.color = ft.Colors.RED_300
            ai_volume_outline_btn.disabled = True
        else:
            title = app_state.book_outline.book_title.strip() or "未命名作品"
            book_outline_mount_status.value = f"总纲挂载状态：已挂载《{title}》"
            book_outline_mount_status.color = ft.Colors.GREEN_300
            ai_volume_outline_btn.disabled = False
        chapter_outline_box.value = app_state.chapter_outline.model_dump_json(indent=2) if app_state.chapter_outline else ""
        sync_outline_visibility()
        sync_chapter_picker()
        render_chapter_cards()

    def build_bubble(role: str, text: str) -> ft.Control:
        is_user = role == "user"
        return ft.Container(
            content=ft.Text(text),
            bgcolor=ft.Colors.BLUE_GREY_700 if is_user else ft.Colors.BLUE_GREY_900,
            padding=10,
            border_radius=8,
            alignment=ft.Alignment(1, 0) if is_user else ft.Alignment(-1, 0),
        )

    def refresh_chat_view(persist: bool = True) -> None:
        chat_list.controls = [build_bubble(item["role"], item["content"]) for item in state.chat_history]
        outline_chat_preview.value = "\n\n".join(
            f"{'你' if item['role'] == 'user' else '主编'}：{item['content']}" for item in state.chat_history
        )
        if persist:
            save_chat_to_cache()

    def render_quick_options() -> None:
        quick_option_row.controls = [
            ft.Button(
                option,
                style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
                on_click=lambda _, value=option: on_pick_quick_option(value),
            )
            for option in state.quick_options
        ]

    def update_proposal_from_board() -> None:
        if not any([proposal_core_hook.value, proposal_golden_finger.value, proposal_world_tone.value]):
            state.concept_proposal = None
            return
        state.concept_proposal = ConceptProposal(
            core_hook=proposal_core_hook.value or "",
            golden_finger=proposal_golden_finger.value or "",
            world_tone=proposal_world_tone.value or "",
        )

    def sync_active_traits_inputs() -> None:
        if state.active_traits is None:
            state.active_traits = load_default_traits()
        active_environment_input.value = state.active_traits.environment
        active_behavior_input.value = state.active_traits.behavior
        active_capability_input.value = state.active_traits.capability
        active_values_input.value = state.active_traits.values
        active_identity_input.value = state.active_traits.identity
        active_vision_input.value = state.active_traits.vision

    def update_active_traits_from_inputs() -> NLPBaseTraits:
        state.active_traits = NLPBaseTraits(
            environment=active_environment_input.value or "",
            behavior=active_behavior_input.value or "",
            capability=active_capability_input.value or "",
            values=active_values_input.value or "",
            identity=active_identity_input.value or "",
            vision=active_vision_input.value or "",
        )
        return state.active_traits

    def on_save_active_traits(_: ft.ControlEvent) -> None:
        try:
            traits = update_active_traits_from_inputs()
            save_active_traits(traits, update_reason="UI档案保存")
            refresh_entity_view()
            save_full_archive()
            set_status("已保存到档案库。")
        except Exception as exc:
            set_status(f"档案保存失败：{exc}")
        page.update()

    def run_brainstorm_cycle(current_idea: str) -> None:
        result = brainstorm_ideas(
            current_idea=current_idea,
            chat_history=state.chat_history,
            model=state.models["supervisor"],
        )
        assistant_reply = str(result.get("assistant_reply", "")).strip()
        if assistant_reply:
            state.chat_history.append({"role": "assistant", "content": assistant_reply})
        options = result.get("quick_options", [])
        state.quick_options = [str(item).strip() for item in options if str(item).strip()]
        proposal_data = result.get("proposal", {})
        if isinstance(proposal_data, dict):
            try:
                state.concept_proposal = ConceptProposal.model_validate(proposal_data)
            except Exception:
                state.concept_proposal = None
        sync_proposal_board()
        refresh_chat_view()
        render_quick_options()

    def on_start_brainstorm(_: ft.ControlEvent) -> None:
        start_brainstorm_btn.disabled = True
        start_brainstorm_btn.text = "正在处理..."
        start_brainstorm_ring.visible = True
        set_status("正在连接 DeepSeek 脑回路...")
        page.update()
        initial = (chat_input.value or "").strip()
        if not initial:
            outlining_status.value = "状态：请先输入第一条灵感。"
            start_brainstorm_btn.disabled = False
            start_brainstorm_btn.text = "🧠 开始头脑风暴"
            start_brainstorm_ring.visible = False
            page.update()
            return
        state.chat_history = [{"role": "user", "content": initial}]
        chat_input.value = ""
        state.quick_options = []
        set_stage(STAGE_CONCEPT)
        outlining_status.value = "状态：主编正在组织头脑风暴..."
        show_loading(page, with_rules_hint("主编正在头脑风暴..."))
        page.update()
        try:
            run_brainstorm_cycle(initial)
            outlining_status.value = "状态：头脑风暴进行中。"
            set_status("主编 Agent 正在提炼方向建议...")
        except Exception as exc:
            outlining_status.value = f"状态：头脑风暴失败 - {exc}"
            set_status("头脑风暴阶段发生异常。")
        finally:
            start_brainstorm_btn.disabled = False
            start_brainstorm_btn.text = "🧠 开始头脑风暴"
            start_brainstorm_ring.visible = False
            hide_loading(page)
        page.update()

    def on_send_sandbox(_: ft.ControlEvent) -> None:
        message = (chat_input.value or "").strip()
        if not message:
            return
        if not state.chat_history:
            state.chat_history = [{"role": "user", "content": message}]
            chat_input.value = ""
            state.quick_options = []
            set_stage(STAGE_CONCEPT)
            outlining_status.value = "状态：主编正在组织头脑风暴..."
            show_loading(page, with_rules_hint("主编正在头脑风暴..."))
            page.update()
            try:
                run_brainstorm_cycle(message)
                outlining_status.value = "状态：头脑风暴进行中。"
            except Exception as exc:
                outlining_status.value = f"状态：头脑风暴失败 - {exc}"
            finally:
                hide_loading(page)
            page.update()
            return
        state.chat_history.append({"role": "user", "content": message})
        chat_input.value = ""
        refresh_chat_view()
        outlining_status.value = "状态：主编正在回应..."
        show_loading(page, with_rules_hint("主编正在回应..."))
        page.update()
        try:
            run_brainstorm_cycle(message)
            outlining_status.value = "状态：头脑风暴进行中。"
        except Exception as exc:
            outlining_status.value = f"状态：头脑风暴失败 - {exc}"
        finally:
            hide_loading(page)
        page.update()

    def on_pick_quick_option(option: str) -> None:
        state.chat_history.append({"role": "user", "content": option})
        refresh_chat_view()
        outlining_status.value = "状态：主编正在回应..."
        show_loading(page, with_rules_hint("主编正在处理快捷方案..."))
        page.update()
        try:
            run_brainstorm_cycle(option)
            outlining_status.value = "状态：已应用快捷方案。"
        except Exception as exc:
            outlining_status.value = f"状态：快捷方案失败 - {exc}"
        finally:
            hide_loading(page)
        page.update()

    def on_spark_click(text: str) -> None:
        chat_input.value = text
        message = (chat_input.value or "").strip()
        if not message:
            return
        if not state.chat_history:
            state.chat_history = [{"role": "user", "content": message}]
            chat_input.value = ""
            state.quick_options = []
            set_stage(STAGE_CONCEPT)
            outlining_status.value = "状态：主编正在组织头脑风暴..."
            show_loading(page, with_rules_hint("主编正在处理灵感火花..."))
            page.update()
            try:
                run_brainstorm_cycle(message)
                outlining_status.value = "状态：头脑风暴进行中。"
            except Exception as exc:
                outlining_status.value = f"状态：头脑风暴失败 - {exc}"
            finally:
                hide_loading(page)
            page.update()
            return
        state.chat_history.append({"role": "user", "content": message})
        chat_input.value = ""
        refresh_chat_view()
        outlining_status.value = "状态：主编正在回应..."
        show_loading(page, with_rules_hint("主编正在处理灵感火花..."))
        page.update()
        try:
            run_brainstorm_cycle(message)
            outlining_status.value = "状态：已应用灵感火花。"
        except Exception as exc:
            outlining_status.value = f"状态：灵感火花失败 - {exc}"
        finally:
            hide_loading(page)
        page.update()

    def on_save_chat_markdown(_: ft.ControlEvent) -> None:
        if not any([state.chat_history, state.concept_proposal, state.book_outline, state.volumes]):
            set_status("暂无可存档内容。")
            page.update()
            return
        path = save_full_archive()
        refresh_archive_options()
        archive_dropdown.value = path.parent.name
        set_status(f"项目已保存：{path.parent.name}")
        page.update()

    def refresh_archive_options() -> None:
        archive_option_map.clear()
        archive_dropdown.options = []
        stage_filter = archive_filter_dropdown.value or "ALL"
        for file_name, label, stage in scan_archives():
            if stage_filter != "ALL" and stage != stage_filter:
                continue
            archive_option_map[file_name] = label
            archive_dropdown.options.append(ft.dropdown.Option(key=file_name, text=label))
        if archive_dropdown.value not in archive_option_map:
            archive_dropdown.value = None

    def on_refresh_archives(_: ft.ControlEvent) -> None:
        refresh_archive_options()
        set_status("历史存档列表已按分类刷新。")
        page.update()

    def on_archive_filter_change(_: ft.ControlEvent) -> None:
        refresh_archive_options()
        set_status("已切换存档分类。")
        page.update()

    def hydrate_state_from_payload(payload: dict[str, Any], fallback_project_name: str = "未命名脑洞") -> None:
        snapshot = payload.get("app_state") if isinstance(payload.get("app_state"), dict) else payload
        state.project_name = str(payload.get("project_name") or fallback_project_name)
        state.chat_history = []
        state.quick_options = []
        state.concept_proposal = None
        state.book_outline = None
        state.volumes = {}
        state.chapter_outlines = {}
        state.chapter_texts = {}
        state.chapter_files = {}
        state.selected_volume_num = None
        state.volume_outline = None
        state.chapter_outline = None
        state.active_traits = None
        state.characters = {}
        state.world_settings = {}
        state.factions = {}
        state.items = {}
        state.timeline_events = {}
        state.mounted_rules = []
        raw_chat = snapshot.get("chat_history", []) or []
        normalized_chat: list[dict[str, str]] = []
        if isinstance(raw_chat, list):
            for item in raw_chat:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    normalized_chat.append({"role": role, "content": content})
        state.chat_history = normalized_chat
        raw_options = snapshot.get("quick_options", []) or []
        state.quick_options = [str(item).strip() for item in raw_options if str(item).strip()]
        concept_data = snapshot.get("current_concept")
        if not isinstance(concept_data, dict):
            concept_data = snapshot.get("concept_proposal")
        state.concept_proposal = ConceptProposal.model_validate(concept_data) if isinstance(concept_data, dict) else None
        book_data = snapshot.get("book_outline")
        state.book_outline = BookOutline.model_validate(book_data) if isinstance(book_data, dict) else None
        volumes_data = snapshot.get("volumes")
        if isinstance(volumes_data, dict):
            for key, value in volumes_data.items():
                try:
                    state.volumes[int(key)] = VolumeOutline.model_validate(value)
                except Exception:
                    continue
        legacy_volume = snapshot.get("volume_outline")
        if not state.volumes and isinstance(legacy_volume, dict):
            parsed_legacy = VolumeOutline.model_validate(legacy_volume)
            state.volumes[parsed_legacy.volume_number] = parsed_legacy
        chapters_data = snapshot.get("chapter_outlines")
        if isinstance(chapters_data, dict):
            for key, value in chapters_data.items():
                try:
                    state.chapter_outlines[int(key)] = ChapterOutlineList.model_validate(value)
                except Exception:
                    continue
        legacy_chapter = snapshot.get("chapter_outline")
        if not state.chapter_outlines and isinstance(legacy_chapter, dict):
            parsed_legacy_chapter = ChapterOutlineList.model_validate(legacy_chapter)
            state.chapter_outlines[parsed_legacy_chapter.volume_number] = parsed_legacy_chapter
        chapter_files_data = snapshot.get("chapter_files")
        if isinstance(chapter_files_data, dict):
            state.chapter_files = {str(k): str(v) for k, v in chapter_files_data.items() if str(k).strip()}
        chapter_texts_data = snapshot.get("chapter_texts")
        if isinstance(chapter_texts_data, dict):
            for raw_key, raw_text in chapter_texts_data.items():
                key = str(raw_key)
                text = str(raw_text)
                state.chapter_texts[key] = text
                parsed = _parse_chapter_text_key(key)
                if parsed is None:
                    continue
                volume_num, chapter_num = parsed
                relative_path = _write_chapter_markdown(volume_num, chapter_num, text)
                state.chapter_files[key] = relative_path
        selected_volume = snapshot.get("selected_volume_num")
        if isinstance(selected_volume, int):
            state.selected_volume_num = selected_volume
        elif state.volumes:
            state.selected_volume_num = sorted(state.volumes.keys())[-1]
        state.volume_outline = state.volumes.get(state.selected_volume_num) if state.selected_volume_num is not None else None
        state.chapter_outline = (
            state.chapter_outlines.get(state.selected_volume_num) if state.selected_volume_num is not None else None
        )
        traits_data = snapshot.get("active_traits")
        if isinstance(traits_data, dict):
            state.active_traits = NLPBaseTraits.model_validate(traits_data)
        else:
            state.active_traits = load_default_traits()
        chars = snapshot.get("characters")
        worlds = snapshot.get("world_settings")
        factions = snapshot.get("factions")
        items = snapshot.get("items")
        timeline_events = snapshot.get("timeline_events")
        state.characters = chars if isinstance(chars, dict) else (chars if isinstance(chars, list) else {})
        state.world_settings = worlds if isinstance(worlds, dict) else (worlds if isinstance(worlds, list) else {})
        state.factions = factions if isinstance(factions, dict) else {}
        state.items = items if isinstance(items, dict) else (items if isinstance(items, list) else {})
        state.timeline_events = timeline_events if isinstance(timeline_events, dict) else {}
        mounted_ids = snapshot.get("mounted_rule_ids")
        if isinstance(mounted_ids, list):
            local_map = {item.rule_id: item for item in load_local_rules()}
            state.mounted_rules = [local_map[item] for item in mounted_ids if isinstance(item, str) and item in local_map]
        raw_stage = str(payload.get("save_stage") or snapshot.get("save_stage") or "").upper()
        stage_from_archive = raw_stage if raw_stage in {STAGE_CONCEPT, STAGE_OUTLINE, STAGE_CHAPTERS} else ""
        if stage_from_archive:
            state.current_stage = stage_from_archive
        elif state.chapter_outline or state.volumes:
            state.current_stage = STAGE_CHAPTERS
        elif state.book_outline:
            state.current_stage = STAGE_OUTLINE
        else:
            state.current_stage = STAGE_CONCEPT

    def on_load_selected_archive(_: ft.ControlEvent) -> None:
        file_name = archive_dropdown.value or ""
        if not file_name:
            set_status("请先选择历史存档。")
            page.update()
            return
        project_ref = _projects_dir() / file_name
        if not project_ref.exists():
            set_status("存档文件不存在。")
            page.update()
            return
        path = resolve_project_meta_path(project_ref)
        if not path.exists():
            set_status("存档元数据不存在。")
            page.update()
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            set_status(f"存档解析失败：{exc}")
            page.update()
            return
        hydrate_state_from_payload(payload, fallback_project_name=path.parent.name)
        refresh_all_views(page, state)
        set_stage(state.current_stage)
        set_status(f"存档 {file_name} 加载成功。")
        page.update()

    def on_clear_chat(_: ft.ControlEvent) -> None:
        def on_confirm_clear(_: ft.ControlEvent) -> None:
            cache_path = _chat_cache_path()
            if cache_path.exists():
                cache_path.unlink()
            state.chat_history = []
            state.quick_options = []
            state.concept_proposal = None
            state.project_name = "未命名脑洞"
            state.book_outline = None
            state.volumes = {}
            state.chapter_outlines = {}
            state.chapter_texts = {}
            state.chapter_files = {}
            state.characters = {}
            state.world_settings = {}
            state.factions = {}
            state.items = {}
            state.timeline_events = {}
            state.mounted_rules = []
            state.selected_volume_num = None
            state.volume_outline = None
            state.chapter_outline = None
            chat_input.value = ""
            book_outline_box.value = ""
            volume_outline_box.value = ""
            chapter_outline_box.value = ""
            chapter_num_input.value = "1"
            chapter_idea_input.value = ""
            refresh_chat_view(persist=False)
            render_quick_options()
            sync_proposal_board()
            render_volume_cards()
            render_outline_boards()
            sync_outline_visibility()
            render_chapter_cards()
            set_stage(STAGE_CONCEPT)
            archive_dropdown.value = None
            dialog.open = False
            set_status("已清空当前项目。")
            page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("确认清空当前项目"),
            content=ft.Text("确认后将清空当前对话与大纲状态，并删除本地缓存。"),
            actions=[
                ft.TextButton("取消", on_click=lambda _: setattr(dialog, "open", False)),
                ft.TextButton("确认清除", on_click=on_confirm_clear),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    def on_quick_jump_stage(_: ft.ControlEvent) -> None:
        target = stage_jump_dropdown.value or STAGE_CONCEPT
        if target == STAGE_CHAPTERS:
            refresh_all_views(page, state)
        set_stage(target)
        set_status(f"已切换到 {target} 阶段。")
        page.update()

    def on_nav_to_concept(_: ft.ControlEvent) -> None:
        set_stage(STAGE_CONCEPT)
        set_status("已切换到创世沙盒。")
        page.update()

    def on_nav_to_outline(_: ft.ControlEvent) -> None:
        refresh_all_views(page, state)
        set_stage(STAGE_OUTLINE)
        set_status("已切换到全书总纲。")
        page.update()

    def on_nav_to_chapters(_: ft.ControlEvent) -> None:
        refresh_all_views(page, state)
        set_stage(STAGE_CHAPTERS)
        set_status("已切换到分卷与章节。")
        page.update()

    def on_finalize_proposal(_: ft.ControlEvent) -> None:
        finalize_proposal_btn.disabled = True
        finalize_proposal_btn.text = "正在处理..."
        finalize_proposal_ring.visible = True
        user_responses = [item["content"] for item in state.chat_history if item["role"] == "user"]
        if not user_responses:
            outlining_status.value = "状态：请先完成至少一轮沟通。"
            finalize_proposal_btn.disabled = False
            finalize_proposal_btn.text = "✅ 定稿策划案"
            finalize_proposal_ring.visible = False
            page.update()
            return
        outlining_status.value = "状态：正在定稿策划案..."
        set_status("主编 Agent 正在收敛故事核心策划案...")
        show_loading(page, with_rules_hint("正在定稿策划案..."))
        page.update()
        try:
            proposal = finalize_proposal(user_responses=user_responses, model=state.models["supervisor"])
            state.concept_proposal = proposal
            sync_proposal_board()
            set_stage(STAGE_OUTLINE)
            outlining_status.value = "状态：策划案已定稿，可生成总纲。"
            request_focus(gen_outline_btn)
            set_status("策划案已定稿，可进入大纲推演。")
        except Exception as exc:
            outlining_status.value = f"状态：策划案定稿失败 - {exc}"
            set_status("策划案定稿失败。")
        finally:
            finalize_proposal_btn.disabled = False
            finalize_proposal_btn.text = "✅ 定稿策划案"
            finalize_proposal_ring.visible = False
            hide_loading(page)
        page.update()

    def on_chapter_pick(_: ft.ControlEvent) -> None:
        selected_volume = _get_selected_workshop_volume()
        selected_chapter = _get_selected_workshop_chapter()
        if selected_volume is None or selected_chapter is None:
            return
        chapter_outline = state.chapter_outlines.get(selected_volume)
        if chapter_outline is None:
            pipeline_output.value = ""
            update_workshop_word_count("")
            page.update()
            return
        matched = None
        for chapter in chapter_outline.chapters:
            if int(chapter.get("chapter_number", 0)) == selected_chapter:
                matched = chapter
                break
        if matched is None:
            return
        state.selected_volume_num = selected_volume
        state.volume_outline = state.volumes.get(selected_volume)
        state.chapter_outline = chapter_outline
        chapter_num_input.value = str(selected_chapter)
        chapter_idea_input.value = str(matched.get("core_event", ""))
        loaded_text = _load_chapter_content_for_workshop(selected_volume, selected_chapter)
        pipeline_output.value = loaded_text
        update_workshop_word_count(loaded_text)
        page.update()

    def on_workshop_volume_pick(_: ft.ControlEvent) -> None:
        selected_volume = _get_selected_workshop_volume()
        if selected_volume is None:
            return
        state.selected_volume_num = selected_volume
        state.volume_outline = state.volumes.get(selected_volume)
        state.chapter_outline = state.chapter_outlines.get(selected_volume)
        render_volume_cards()
        render_outline_boards()
        chapter_outline_box.value = state.chapter_outline.model_dump_json(indent=2) if state.chapter_outline else ""
        sync_chapter_picker()
        render_chapter_cards()
        pipeline_output.value = ""
        update_workshop_word_count("")
        page.update()

    def on_workshop_text_change(event: ft.ControlEvent) -> None:
        current_text = event.control.value or ""
        update_workshop_word_count(current_text)
        page.update()

    def on_save_current_chapter(_: ft.ControlEvent) -> None:
        selected_volume = _get_selected_workshop_volume()
        selected_chapter = _get_selected_workshop_chapter()
        if selected_volume is None or selected_chapter is None:
            pipeline_status.value = "状态：请先选择分卷与章节。"
            page.update()
            return
        text_key = f"{selected_volume}:{selected_chapter}"
        text = pipeline_output.value or ""
        save_chapter_text(text_key, text)
        save_full_archive()
        pipeline_status.value = f"状态：第{selected_volume}卷 第{selected_chapter}章已保存。"
        set_status("章节正文已保存到独立文件。")
        page.update()

    workshop_volume_picker.on_change = on_workshop_volume_pick
    chapter_picker.on_change = on_chapter_pick
    pipeline_output.on_change = on_workshop_text_change

    entity_schema: dict[str, dict[str, Any]] = {
        "characters": {
            "label": "👥 人物",
            "default_name": "新人物",
            "fields": [
                ("environment", "环境 / Environment"),
                ("behavior", "行为 / Behavior"),
                ("capability", "能力 / Capability"),
                ("values", "价值观 / Values"),
                ("identity", "身份 / Identity"),
                ("vision", "愿景 / Vision"),
            ],
        },
        "world_settings": {
            "label": "🌍 世界",
            "default_name": "新世界设定",
            "fields": [
                ("background_origin", "背景起源"),
                ("core_rules", "核心规则"),
                ("related_figures", "相关势力人物"),
                ("current_status", "当前状态"),
            ],
        },
        "factions": {
            "label": "🏢 势力",
            "default_name": "新势力",
            "fields": [
                ("core_business_assets", "核心业务与资产"),
                ("key_figures", "关键人物"),
                ("relation_to_protagonist", "与主角关系"),
                ("development_vision", "发展愿景"),
            ],
        },
        "items": {
            "label": "💎 资产",
            "default_name": "新资产",
            "fields": [
                ("acquisition_source", "获取来源"),
                ("core_function", "核心功能机制"),
                ("usage_limits_cost", "使用限制与代价"),
                ("strategic_meaning", "剧情战略意义"),
            ],
        },
        "timeline_events": {
            "label": "📅 时间线",
            "default_name": "新时间线事件",
            "fields": [
                ("event_time", "发生时间"),
                ("real_history_event", "真实历史事件"),
                ("intervention_plan", "主角干预计划"),
                ("butterfly_effect", "蝴蝶效应预测"),
            ],
        },
    }

    def _get_current_entity_category() -> str:
        category = entity_category_picker.value or "characters"
        return category if category in entity_schema else "characters"

    def _entity_bucket(category: str) -> dict[str, dict[str, Any]]:
        _normalize_legacy_entity_data()
        mapping = {
            "characters": state.characters,
            "world_settings": state.world_settings,
            "factions": state.factions,
            "items": state.items,
            "timeline_events": state.timeline_events,
        }
        bucket = mapping.get(category, {})
        if isinstance(bucket, dict):
            return bucket
        if category == "characters":
            state.characters = {}
            return state.characters
        if category == "world_settings":
            state.world_settings = {}
            return state.world_settings
        if category == "factions":
            state.factions = {}
            return state.factions
        if category == "items":
            state.items = {}
            return state.items
        state.timeline_events = {}
        return state.timeline_events

    def _entity_name_of(row: dict[str, Any], default_name: str) -> str:
        value = str(row.get("name", "")).strip()
        return value if value else default_name

    def _normalize_legacy_entity_data() -> None:
        if isinstance(state.characters, list):
            legacy = state.characters
            state.characters = {}
            for idx, row in enumerate(legacy):
                if not isinstance(row, dict):
                    continue
                key = str(row.get("id") or uuid4().hex)
                state.characters[key] = {
                    "name": str(row.get("entity_id") or f"角色{idx + 1}"),
                    "environment": str(row.get("environment", "")),
                    "behavior": str(row.get("behavior", "")),
                    "capability": str(row.get("capability", "")),
                    "values": str(row.get("values", "")),
                    "identity": str(row.get("identity", "")),
                    "vision": str(row.get("vision", "")),
                }
        if isinstance(state.world_settings, list):
            legacy = state.world_settings
            state.world_settings = {}
            for idx, row in enumerate(legacy):
                if not isinstance(row, dict):
                    continue
                key = str(row.get("id") or uuid4().hex)
                state.world_settings[key] = {
                    "name": str(row.get("region_name") or f"世界设定{idx + 1}"),
                    "background_origin": str(row.get("region_name", "")),
                    "core_rules": str(row.get("hidden_rules", "")),
                    "related_figures": str(row.get("power_structure", "")),
                    "current_status": str(row.get("tech_level", "")),
                }
        if isinstance(state.items, list):
            legacy = state.items
            state.items = {}
            for idx, row in enumerate(legacy):
                if not isinstance(row, dict):
                    continue
                key = str(row.get("id") or uuid4().hex)
                state.items[key] = {
                    "name": str(row.get("item_name") or f"资产{idx + 1}"),
                    "acquisition_source": str(row.get("origin", "")),
                    "core_function": str(row.get("item_function", "")),
                    "usage_limits_cost": str(row.get("hidden_power", "")),
                    "strategic_meaning": str(row.get("story_hook", "")),
                }
        if not isinstance(state.factions, dict):
            state.factions = {}
        if not isinstance(state.timeline_events, dict):
            state.timeline_events = {}

    def render_entity_list() -> None:
        category = _get_current_entity_category()
        bucket = _entity_bucket(category)
        entity_list_view.controls.clear()
        selected_key = selected_entity_ref["key"]
        if not bucket:
            selected_entity_ref["key"] = None
            entity_list_view.controls.append(ft.Text("暂无档案。", color=ft.Colors.BLUE_GREY_300))
            return
        if selected_key not in bucket:
            selected_entity_ref["key"] = None
            selected_key = None
        schema = entity_schema[category]
        for key, row in bucket.items():
            name = _entity_name_of(row, schema["default_name"])
            is_selected = selected_key == key
            entity_list_view.controls.append(
                ft.Container(
                    bgcolor=ft.Colors.BLUE_GREY_800 if is_selected else ft.Colors.BLUE_GREY_900,
                    border_radius=8,
                    content=ft.TextButton(
                        name,
                        on_click=lambda event, entity_key=key: on_select_entity(event, entity_key),
                        style=ft.ButtonStyle(color=ft.Colors.WHITE),
                    ),
                )
            )

    def render_entity_detail_panel() -> None:
        category = _get_current_entity_category()
        bucket = _entity_bucket(category)
        selected_key = selected_entity_ref["key"]
        if selected_key is None or selected_key not in bucket:
            entity_detail_box.content = ft.Container(
                expand=True,
                alignment=ft.Alignment(0, 0),
                content=ft.Text("请在左侧选择或新建实体", color=ft.Colors.BLUE_GREY_300),
            )
            return
        schema = entity_schema[category]
        row = bucket[selected_key]
        entity_name_input.value = _entity_name_of(row, schema["default_name"])
        entity_form_field_refs.clear()
        field_controls: list[ft.Control] = [entity_name_input]
        for field_key, field_label in schema["fields"]:
            field = ft.TextField(label=field_label, multiline=True, min_lines=4, value=str(row.get(field_key, "")))
            entity_form_field_refs[field_key] = field
            field_controls.append(field)
        field_controls.append(
            ft.Row(
                controls=[
                    ft.Button(
                        "🗑️ 删除该档案",
                        on_click=on_delete_selected_entity,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE),
                    ),
                    ft.Button(
                        "💾 保存修改",
                        on_click=on_save_selected_entity,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
                    ),
                ],
                spacing=10,
            )
        )
        entity_detail_box.content = ft.Column(
            controls=field_controls,
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def on_select_entity(_: ft.ControlEvent, entity_key: str) -> None:
        selected_entity_ref["key"] = entity_key
        render_entity_list()
        render_entity_detail_panel()
        page.update()

    def refresh_entity_view(_: ft.ControlEvent | None = None) -> None:
        _normalize_legacy_entity_data()
        render_entity_list()
        render_entity_detail_panel()
        page.update()

    def on_entity_category_change(_: ft.ControlEvent) -> None:
        selected_entity_ref["key"] = None
        render_entity_list()
        render_entity_detail_panel()
        page.update()

    def on_add_entity(_: ft.ControlEvent) -> None:
        category = _get_current_entity_category()
        schema = entity_schema[category]
        bucket = _entity_bucket(category)
        key = uuid4().hex
        payload: dict[str, Any] = {"name": schema["default_name"]}
        for field_key, _ in schema["fields"]:
            payload[field_key] = ""
        bucket[key] = payload
        selected_entity_ref["key"] = key
        save_full_archive()
        render_entity_list()
        render_entity_detail_panel()
        page.update()

    def on_save_selected_entity(_: ft.ControlEvent) -> None:
        category = _get_current_entity_category()
        bucket = _entity_bucket(category)
        selected_key = selected_entity_ref["key"]
        if selected_key is None or selected_key not in bucket:
            show_info_dialog("请先在左侧选择实体。")
            return
        schema = entity_schema[category]
        updated: dict[str, Any] = {"name": (entity_name_input.value or schema["default_name"]).strip()}
        for field_key, _ in schema["fields"]:
            field = entity_form_field_refs.get(field_key)
            updated[field_key] = (field.value if field else "") or ""
        bucket[selected_key] = updated
        save_full_archive()
        render_entity_list()
        render_entity_detail_panel()
        set_status("档案已保存。")
        page.update()

    def on_delete_selected_entity(_: ft.ControlEvent) -> None:
        category = _get_current_entity_category()
        bucket = _entity_bucket(category)
        selected_key = selected_entity_ref["key"]
        if selected_key is None or selected_key not in bucket:
            show_info_dialog("请先在左侧选择实体。")
            return

        def on_cancel(_: ft.ControlEvent) -> None:
            dialog.open = False
            page.update()

        def on_confirm(_: ft.ControlEvent) -> None:
            bucket.pop(selected_key, None)
            selected_entity_ref["key"] = None
            dialog.open = False
            save_full_archive()
            render_entity_list()
            render_entity_detail_panel()
            page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("确认删除"),
            content=ft.Text("删除后不可恢复，是否继续？"),
            actions=[
                ft.TextButton("取消", on_click=on_cancel),
                ft.TextButton("确认删除", on_click=on_confirm),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    def on_ai_generate_entity(_: ft.ControlEvent) -> None:
        category = _get_current_entity_category()
        schema = entity_schema[category]
        requirement_input = ft.TextField(
            label="生成要求",
            hint_text="例如：生成一个 2003 年的 SP 业务皮包公司",
            multiline=True,
            min_lines=2,
            max_lines=4,
            autofocus=True,
        )

        def on_cancel(_: ft.ControlEvent) -> None:
            dialog.open = False
            page.update()

        def on_confirm(_: ft.ControlEvent) -> None:
            requirement = (requirement_input.value or "").strip()
            if not requirement:
                show_info_dialog("请输入要求后再生成。")
                return
            dialog.open = False
            page.update()
            show_loading(page, with_rules_hint("AI 正在定向生成档案..."))
            try:
                fields = [field_key for field_key, _ in schema["fields"]]
                response = generate_text(
                    system_prompt=(
                        "你是设定数据库生成助手。请仅输出 JSON 对象，不要输出任何解释。"
                        "必须包含 name 字段与要求的字段。"
                    ),
                    user_prompt=(
                        f"分类：{schema['label']}\n"
                        f"用户要求：{requirement}\n"
                        f"必须输出字段：name, {', '.join(fields)}\n"
                        "输出格式：严格 JSON 对象。"
                    ),
                    model=state.models["supervisor"],
                    temperature=0.7,
                )
                parsed = json.loads(response)
                if not isinstance(parsed, dict):
                    raise ValueError("AI 返回不是 JSON 对象")
                payload: dict[str, Any] = {"name": str(parsed.get("name") or schema["default_name"])}
                for field_key, _ in schema["fields"]:
                    payload[field_key] = str(parsed.get(field_key, ""))
                bucket = _entity_bucket(category)
                key = uuid4().hex
                bucket[key] = payload
                selected_entity_ref["key"] = key
                save_full_archive()
                render_entity_list()
                render_entity_detail_panel()
                page.update()
            except Exception as exc:
                show_info_dialog(f"AI 定向生成失败：{exc}")
            finally:
                hide_loading(page)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("🤖 AI 定向生成"),
            content=requirement_input,
            actions=[
                ft.TextButton("取消", on_click=on_cancel),
                ft.TextButton("生成", on_click=on_confirm),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    def render_character_cards() -> None:
        render_entity_list()

    entity_category_picker.on_change = on_entity_category_change

    def on_generate_book_outline_only(_: ft.ControlEvent) -> None:
        idea = ""
        for item in state.chat_history:
            if item["role"] == "user":
                idea = item["content"].strip()
                if idea:
                    break
        if not idea:
            idea = (chat_input.value or "").strip()
        if not idea:
            outlining_status.value = "状态：请先在对话区输入脑洞并开始头脑风暴。"
            page.update()
            return
        update_proposal_from_board()
        ai_book_outline_btn.disabled = True
        ai_book_outline_ring.visible = True
        outline_loading.visible = True
        set_status("AI 正在疯狂码字推演大纲中...")
        show_loading(page, with_rules_hint("正在推演全书总纲..."))
        page.update()
        try:
            book = generate_book_outline(
                idea,
                model=state.models["supervisor"],
                concept_proposal=state.concept_proposal,
                mounted_rules=state.mounted_rules,
            )
            state.book_outline = book
            if state.project_name == "未命名脑洞" and book.book_title.strip():
                state.project_name = book.book_title.strip()
            refresh_all_views(page, state)
            save_full_archive()
            book_outline_box.visible = True
            set_stage(STAGE_OUTLINE)
            outlining_status.value = "状态：全书总纲推演完成。"
            request_focus(ai_volume_outline_btn)
        except Exception as exc:
            outlining_status.value = f"状态：全书总纲推演失败 - {exc}"
        finally:
            ai_book_outline_btn.disabled = False
            ai_book_outline_ring.visible = False
            outline_loading.visible = False
            hide_loading(page)
            page.update()

    def on_generate_volume_outline_only(_: ft.ControlEvent) -> None:
        if state.book_outline is None:
            show_info_dialog("请先完成或加载全书总纲")
            set_status("请先完成或加载全书总纲。")
            return
        try:
            target_volume_num = int(volume_num_input.value or "1")
        except Exception as exc:
            outlining_status.value = f"状态：总纲或卷号无效 - {exc}"
            page.update()
            return
        ai_volume_outline_btn.disabled = True
        ai_volume_outline_ring.visible = True
        outline_loading.visible = True
        set_status("AI 正在疯狂码字推演大纲中...")
        show_loading(page, with_rules_hint("正在推演分卷大纲..."))
        page.update()
        try:
            volume = generate_volume_outline(
                book_outline=state.book_outline,
                target_volume_num=target_volume_num,
                model=state.models["supervisor"],
                mounted_rules=state.mounted_rules,
            )
            state.volumes[target_volume_num] = volume
            on_select_volume(target_volume_num, refresh_page=False)
            refresh_all_views(page, state)
            save_full_archive()
            set_stage(STAGE_CHAPTERS)
            outlining_status.value = "状态：分卷大纲推演完成。"
            set_status(f"已完成第 {target_volume_num} 卷推演并自动选中。")
        except Exception as exc:
            outlining_status.value = f"状态：分卷大纲推演失败 - {exc}"
            show_info_dialog(f"分卷推演失败：{exc}")
        finally:
            ai_volume_outline_btn.disabled = False
            ai_volume_outline_ring.visible = False
            outline_loading.visible = False
            hide_loading(page)
            page.update()

    def on_generate_chapters_and_enter(_: ft.ControlEvent) -> None:
        if state.volume_outline is None and state.volumes:
            state.selected_volume_num = sorted(state.volumes.keys())[0]
            on_select_volume(state.selected_volume_num)
        if state.volume_outline is None:
            outlining_status.value = "状态：请先推演分卷大纲。"
            page.update()
            return
        try:
            chapter_count = int(chapter_count_input.value or "20")
        except ValueError:
            outlining_status.value = "状态：目标章节数必须为整数。"
            page.update()
            return
        ai_chapters_btn.disabled = True
        ai_chapters_ring.visible = True
        outline_loading.visible = True
        set_status("AI 正在疯狂码字推演大纲中...")
        show_loading(page, with_rules_hint("正在推演单章脑洞列表..."))
        page.update()
        try:
            chapters = generate_chapter_ideas(
                volume_outline=state.volume_outline,
                chapter_count=chapter_count,
                model=state.models["supervisor"],
                mounted_rules=state.mounted_rules,
            )
            state.chapter_outline = chapters
            if state.selected_volume_num is not None:
                state.chapter_outlines[state.selected_volume_num] = chapters
            chapter_outline_box.value = chapters.model_dump_json(indent=2)
            sync_chapter_picker()
            sync_outline_visibility()
            render_chapter_cards()
            save_full_archive()
            set_stage(STAGE_CHAPTERS)
            outlining_status.value = "状态：已进入 CHAPTERS 阶段。"
            set_status("已生成单章脑洞并切换到 CHAPTERS 阶段。")
        except Exception as exc:
            outlining_status.value = f"状态：单章脑洞推演失败 - {exc}"
        finally:
            ai_chapters_btn.disabled = False
            ai_chapters_ring.visible = False
            outline_loading.visible = False
            hide_loading(page)
            page.update()

    def on_generate_outline(_: ft.ControlEvent) -> None:
        idea = ""
        for item in state.chat_history:
            if item["role"] == "user":
                idea = item["content"].strip()
                if idea:
                    break
        if not idea:
            idea = (chat_input.value or "").strip()
        if not idea:
            outlining_status.value = "状态：请先在对话区输入脑洞并开始头脑风暴。"
            page.update()
            return
        update_proposal_from_board()
        gen_outline_btn.disabled = True
        gen_outline_btn.text = "正在处理..."
        gen_outline_ring.visible = True
        outline_loading.visible = True
        set_stage(STAGE_OUTLINE)
        outlining_status.value = "状态：正在生成全书总纲..."
        set_status("正在连接 DeepSeek 脑回路...")
        show_loading(page, with_rules_hint("正在生成全书总纲与人设..."))
        page.update()
        try:
            book = generate_book_outline(
                idea,
                model=state.models["supervisor"],
                concept_proposal=state.concept_proposal,
                mounted_rules=state.mounted_rules,
            )
            state.book_outline = book
            if state.project_name == "未命名脑洞" and book.book_title.strip():
                state.project_name = book.book_title.strip()
            try:
                state.active_traits = derive_initial_traits(
                    user_idea=idea,
                    book_outline=book,
                    concept_proposal=state.concept_proposal,
                    model=state.models["supervisor"],
                )
                save_active_traits(state.active_traits, update_reason="总纲生成后自动提取")
                sync_active_traits_inputs()
            except Exception as traits_exc:
                set_status(f"已生成大纲，但人设自动提取失败：{traits_exc}")
            state.chapter_state = {
                "chapter_num": 1,
                "chapter_idea": "",
                "checker_feedback": "",
                "retry_count": 0,
            }
            render_outline_boards()
            render_volume_cards()
            sync_chapter_picker()
            sync_outline_visibility()
            render_chapter_cards()
            set_stage(STAGE_OUTLINE)
            outlining_status.value = "状态：全书总纲生成完成。"
            set_status("全书总纲生成完成，可前往分卷与章节。")
            save_full_archive()
        except Exception as exc:
            outlining_status.value = f"状态：生成失败 - {exc}"
            set_status("大纲生成失败。")
        finally:
            gen_outline_btn.disabled = False
            gen_outline_btn.text = "🚀 确定，生成总纲"
            gen_outline_ring.visible = False
            outline_loading.visible = False
            hide_loading(page)
            page.update()

    def on_apply_outline_text(_: ft.ControlEvent) -> None:
        try:
            update_proposal_from_board()
            render_outline_boards()
            sync_chapter_picker()
            sync_outline_visibility()
            render_chapter_cards()
            outlining_status.value = "状态：已应用编辑后的大纲。"
        except Exception as exc:
            outlining_status.value = f"状态：应用失败 - {exc}"
        page.update()

    def on_run_pipeline(_: ft.ControlEvent) -> None:
        started_at = perf_counter()
        try:
            chapter_num = int(chapter_num_input.value or "1")
        except ValueError:
            pipeline_status.value = "状态：章节号格式错误，请输入整数。"
            page.update()
            return
        idea = (chapter_idea_input.value or "").strip()
        if not idea:
            pipeline_status.value = "状态：请填写本章核心脑洞。"
            page.update()
            return
        run_pipeline_btn.disabled = True
        run_pipeline_btn.text = "正在处理..."
        run_pipeline_ring.visible = True
        pipeline_progress.visible = True
        pipeline_status.value = "状态：正在加载挂载法则..."
        set_status("正在加载挂载法则...")
        logger.info(
            "ui.pipeline.start chapter=%s idea_chars=%s mounted_rules=%s models=%s/%s/%s",
            chapter_num,
            len(idea),
            len(state.mounted_rules),
            state.models["planner"],
            state.models["drafter"],
            state.models["checker"],
        )
        show_loading(page, with_rules_hint("正在执行章节流水线..."))
        pipeline_output.value = ""
        update_workshop_word_count("")
        page.update()
        try:
            pipeline_status.value = "状态：正在执行章节流水线..."
            set_status("裁判 Agent 正在审稿，请稍候...")
            page.update()
            if state.active_traits is None:
                state.active_traits = load_default_traits()
            final_text = run_chapter_pipeline(
                chapter_num=chapter_num,
                idea=idea,
                traits=state.active_traits,
                mounted_rules=state.mounted_rules,
                planner_model=state.models["planner"],
                drafter_model=state.models["drafter"],
                checker_model=state.models["checker"],
            )
            pipeline_output.value = final_text
            update_workshop_word_count(final_text)
            request_focus(pipeline_output)
            selected_volume = _get_selected_workshop_volume()
            current_volume = selected_volume if selected_volume is not None else (
                state.selected_volume_num if state.selected_volume_num is not None else 1
            )
            text_key = f"{current_volume}:{chapter_num}"
            save_chapter_text(text_key, final_text)
            if current_volume > 0:
                workshop_volume_picker.value = str(current_volume)
            chapter_picker.value = str(chapter_num)
            state.chapter_state = {
                "chapter_num": chapter_num,
                "chapter_idea": idea,
                "checker_feedback": "",
                "retry_count": 0,
            }
            save_full_archive()
            pipeline_status.value = "状态：生成完成。"
            set_status("章节生成完成。")
            logger.info(
                "ui.pipeline.done chapter=%s duration_ms=%s output_chars=%s",
                chapter_num,
                int((perf_counter() - started_at) * 1000),
                len(final_text),
            )
        except Exception as exc:
            pipeline_status.value = f"状态：流水线失败 - {exc}"
            set_status("流水线执行失败。")
            logger.exception(
                "ui.pipeline.error chapter=%s duration_ms=%s error=%s",
                chapter_num_input.value or "",
                int((perf_counter() - started_at) * 1000),
                exc,
            )
        finally:
            run_pipeline_btn.disabled = False
            run_pipeline_btn.text = "🔥 启动流水线"
            run_pipeline_ring.visible = False
            pipeline_progress.visible = False
            hide_loading(page)
            page.update()

    def on_save_settings(_: ft.ControlEvent) -> None:
        save_settings_btn.disabled = True
        save_settings_btn.text = "正在处理..."
        save_settings_ring.visible = True
        state.agent_aliases["supervisor"] = supervisor_mapping_dropdown.value or ""
        state.agent_aliases["planner"] = planner_mapping_dropdown.value or ""
        state.agent_aliases["drafter"] = drafter_mapping_dropdown.value or ""
        state.agent_aliases["checker"] = checker_mapping_dropdown.value or ""
        env = _load_env_map()
        env["DEEPSEEK_API_KEY"] = api_key_input.value or ""
        _save_env_map(env)
        os.environ["DEEPSEEK_API_KEY"] = api_key_input.value or ""
        _save_model_config(state.model_entries, state.agent_aliases)
        state.models = _resolve_agent_models(state.model_entries, state.agent_aliases)
        settings_status.value = "状态：.env、模型列表与映射已保存。"
        set_status("配置已持久化到本地。")
        save_settings_btn.disabled = False
        save_settings_btn.text = "保存 .env 与模型"
        save_settings_ring.visible = False
        page.update()

    start_brainstorm_btn = ft.Button(
        "🧠 开始头脑风暴",
        on_click=on_start_brainstorm,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    send_sandbox_btn = ft.Button(
        "发送",
        on_click=on_send_sandbox,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    finalize_proposal_btn = ft.Button(
        "✅ 定稿策划案",
        on_click=on_finalize_proposal,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    gen_outline_btn = ft.Button(
        "🚀 确定，生成总纲",
        on_click=on_generate_outline,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    apply_outline_btn = ft.OutlinedButton("应用编辑后的大纲", on_click=on_apply_outline_text)
    run_pipeline_btn = ft.Button(
        "🔥 启动流水线",
        on_click=on_run_pipeline,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
        height=52,
    )
    save_chapter_btn = ft.Button(
        "💾 保存本章",
        on_click=on_save_current_chapter,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )
    refresh_entities_btn = ft.OutlinedButton("刷新档案", on_click=refresh_entity_view)
    add_character_btn = ft.Button(
        "➕ 手动新建",
        on_click=on_add_entity,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    ai_character_btn = ft.Button(
        "🤖 AI 定向生成",
        on_click=on_ai_generate_entity,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )
    save_settings_btn = ft.Button(
        "保存 .env 与模型",
        on_click=on_save_settings,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    add_model_btn = ft.Button(
        "新增模型配置",
        on_click=on_add_model_entry,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    save_active_traits_btn = ft.Button(
        "💾 保存到档案库",
        on_click=on_save_active_traits,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    save_chat_btn = ft.Button(
        "💾 保存项目到本地",
        on_click=on_save_chat_markdown,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    ai_book_outline_btn = ft.Button(
        "🤖 1. AI 推演全书总纲",
        on_click=on_generate_book_outline_only,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )
    ai_volume_outline_btn = ft.Button(
        "➕ 新增/推演下一卷",
        on_click=on_generate_volume_outline_only,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )
    ai_chapters_btn = ft.Button(
        "🤖 AI 推演单章脑洞列表",
        on_click=on_generate_chapters_and_enter,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )
    extract_rule_btn = ft.Button(
        "🔥 AI 深度提炼",
        on_click=on_extract_rule,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )
    save_rule_btn.on_click = on_save_rule
    outline_to_chapters_btn = ft.Button(
        "➡️ 前往分卷与章节",
        on_click=on_nav_to_chapters,
        style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE),
    )
    ai_book_outline_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    ai_volume_outline_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    ai_chapters_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    archive_dropdown = ft.Dropdown(
        label="选择历史存档",
        hint_text="选择历史存档",
        options=[],
        expand=True,
    )
    archive_filter_dropdown = ft.Dropdown(
        label="分类",
        value=STAGE_CONCEPT,
        options=[
            ft.dropdown.Option(key="ALL", text="全部"),
            ft.dropdown.Option(key=STAGE_CONCEPT, text="创世沙盒"),
            ft.dropdown.Option(key=STAGE_OUTLINE, text="全书总纲"),
            ft.dropdown.Option(key=STAGE_CHAPTERS, text="分卷与章节"),
        ],
        width=140,
    )
    archive_filter_dropdown.on_change = on_archive_filter_change
    archive_filter_ref["dropdown"] = archive_filter_dropdown
    refresh_archives_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        on_click=on_refresh_archives,
        icon_color=ft.Colors.BLUE_400,
        tooltip="刷新存档列表",
    )
    load_archive_btn = ft.Button(
        "📂 加载存档",
        on_click=on_load_selected_archive,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
    )
    clear_chat_btn = ft.TextButton(
        "🧹 清空当前项目",
        on_click=on_clear_chat,
        style=ft.ButtonStyle(color=ft.Colors.RED_300),
    )
    stage_jump_dropdown = ft.Dropdown(
        label="当前阶段快捷跳转",
        value=STAGE_CONCEPT,
        options=[
            ft.dropdown.Option(STAGE_CONCEPT),
            ft.dropdown.Option(STAGE_OUTLINE),
            ft.dropdown.Option(STAGE_CHAPTERS),
        ],
        width=220,
    )
    stage_jump_dropdown.on_change = on_quick_jump_stage
    stage_jump_ref["dropdown"] = stage_jump_dropdown
    global_console_row = ft.Row(
        controls=[
            save_chat_btn,
            stage_jump_dropdown,
            clear_chat_btn,
        ],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    nav_concept_btn = ft.Button("创世沙盒 (Concept)", on_click=on_nav_to_concept)
    nav_outline_btn = ft.Button("全书总纲 (Book Outline)", on_click=on_nav_to_outline)
    nav_chapters_btn = ft.Button("分卷与章节 (Volumes & Chapters)", on_click=on_nav_to_chapters)
    stage_nav_ref["concept"] = nav_concept_btn
    stage_nav_ref["outline"] = nav_outline_btn
    stage_nav_ref["chapters"] = nav_chapters_btn
    free_nav_row = ft.Row(
        controls=[nav_concept_btn, nav_outline_btn, nav_chapters_btn],
        spacing=10,
        wrap=True,
    )
    start_brainstorm_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    finalize_proposal_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    gen_outline_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    run_pipeline_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    save_settings_ring = ft.ProgressRing(visible=False, width=16, height=16, color=ft.Colors.BLUE_400)
    outline_primary_row = ft.Row(
        [start_brainstorm_btn, start_brainstorm_ring, finalize_proposal_btn, finalize_proposal_ring, gen_outline_btn, gen_outline_ring],
        spacing=8,
    )
    pipeline_run_row = ft.Row([run_pipeline_btn, run_pipeline_ring], spacing=8)
    workshop_footer_row = ft.Row(
        controls=[save_chapter_btn, ft.Container(expand=True), workshop_word_count],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )
    settings_save_row = ft.Row([save_settings_btn, save_settings_ring], spacing=8)
    spark_row.controls = [
        ft.Button(
            "帮我补充反派设定",
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_GREY_800, color=ft.Colors.BLUE_300),
            on_click=lambda _: on_spark_click("帮我补充反派设定"),
        ),
        ft.Button(
            "想个反转",
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_GREY_800, color=ft.Colors.BLUE_300),
            on_click=lambda _: on_spark_click("想个反转"),
        ),
        ft.Button(
            "强化爽点节奏",
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_GREY_800, color=ft.Colors.BLUE_300),
            on_click=lambda _: on_spark_click("强化爽点节奏"),
        ),
    ]
    tabs_ref: dict[str, ft.Tabs | None] = {"tabs": None}
    home_tabs_ref: dict[str, ft.Tabs | None] = {"tabs": None}
    home_view_ref: dict[str, ft.Container | None] = {"view": None}
    workspace_view_ref: dict[str, ft.Container | None] = {"view": None}
    workspace_project_title = ft.Text("当前项目：未命名脑洞", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_300)
    project_grid = ft.GridView(
        expand=True,
        runs_count=3,
        max_extent=420,
        child_aspect_ratio=2.6,
        spacing=12,
        run_spacing=12,
    )
    home_empty_hint = ft.Text("暂无项目，点击“➕ 创建新书”开始。", color=ft.Colors.BLUE_GREY_300)

    def on_main_tabs_change(event: ft.ControlEvent) -> None:
        selected_idx = int(getattr(event.control, "selected_index", 0) or 0)
        if selected_idx == 0 and state.current_stage == STAGE_CHAPTERS:
            refresh_all_views(page, state)
            page.update()
        if selected_idx == 1:
            render_character_cards()
            render_entity_detail_panel()
            page.update()

    def on_home_tabs_change(event: ft.ControlEvent) -> None:
        selected_idx = int(getattr(event.control, "selected_index", 0) or 0)
        if selected_idx == 0:
            render_project_shelf()
        if selected_idx == 1:
            reload_local_rules()
            render_rule_result(latest_rule_ref.get("rule"))
        page.update()

    def sync_workspace_project_title() -> None:
        workspace_project_title.value = f"当前项目：{get_project_name()}"

    def show_workspace_view() -> None:
        reload_local_rules()
        sync_workspace_project_title()
        refresh_all_views(page, state)
        set_stage(state.current_stage)
        if home_view_ref["view"] is not None:
            home_view_ref["view"].visible = False
        if workspace_view_ref["view"] is not None:
            workspace_view_ref["view"].visible = True
        page.update()

    def render_project_shelf() -> None:
        cards: list[ft.Control] = []
        for path, project_name, saved_at in list_projects():
            time_text = saved_at[0:19].replace("T", " ")

            def on_open(_: ft.ControlEvent, project_path: Path = path) -> None:
                meta_path = resolve_project_meta_path(project_path)
                if not meta_path.exists():
                    show_info_dialog("项目元数据不存在。")
                    return
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    show_info_dialog(f"项目读取失败：{exc}")
                    return
                hydrate_state_from_payload(payload, fallback_project_name=project_path.name)
                set_status(f"项目《{state.project_name}》已加载。")
                show_workspace_view()

            cards.append(
                ft.Card(
                    content=ft.Container(
                        padding=12,
                        on_click=on_open,
                        content=ft.Column(
                            controls=[
                                ft.Text(project_name, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_300),
                                ft.Text(f"最后修改：{time_text}", size=12, color=ft.Colors.BLUE_GREY_300),
                            ],
                            spacing=6,
                        ),
                    )
                )
            )
        project_grid.controls = cards
        home_empty_hint.visible = not bool(cards)

    def show_home_view() -> None:
        render_project_shelf()
        if home_view_ref["view"] is not None:
            home_view_ref["view"].visible = True
        if workspace_view_ref["view"] is not None:
            workspace_view_ref["view"].visible = False
        page.update()

    def on_back_to_home(_: ft.ControlEvent) -> None:
        save_full_archive()
        show_home_view()

    def on_create_new_project(_: ft.ControlEvent) -> None:
        project_input = ft.TextField(label="书名", autofocus=True)

        def on_confirm_create(_: ft.ControlEvent) -> None:
            raw_name = (project_input.value or "").strip()
            project_name = raw_name or "未命名脑洞"
            state.project_name = project_name
            state.chat_history = []
            state.quick_options = []
            state.concept_proposal = None
            state.book_outline = None
            state.volumes = {}
            state.chapter_outlines = {}
            state.chapter_texts = {}
            state.chapter_files = {}
            state.selected_volume_num = None
            state.volume_outline = None
            state.chapter_outline = None
            state.characters = {}
            state.world_settings = {}
            state.factions = {}
            state.items = {}
            state.timeline_events = {}
            state.mounted_rules = []
            state.active_traits = load_default_traits()
            state.current_stage = STAGE_CONCEPT
            save_full_archive()
            dialog.open = False
            set_status(f"已创建项目《{project_name}》。")
            show_workspace_view()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("创建新书"),
            content=project_input,
            actions=[
                ft.TextButton("取消", on_click=lambda _: setattr(dialog, "open", False)),
                ft.TextButton("创建", on_click=on_confirm_create),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    def open_fullscreen_editor(title: str, target: ft.TextField) -> None:
        editor = ft.TextField(value=target.value, multiline=True, min_lines=25, max_lines=35, expand=True)

        def on_save_dialog(_: ft.ControlEvent) -> None:
            target.value = editor.value or ""
            sync_outline_visibility()
            dialog.open = False
            page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Container(width=1100, height=700, content=editor),
            actions=[
                ft.TextButton("取消", on_click=lambda _: setattr(dialog, "open", False)),
                ft.TextButton("保存", on_click=on_save_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    concept_panel = ft.Container(
        expand=True,
        content=ft.Row(
            controls=[
                ft.Container(
                    expand=5,
                    content=ft.Column(
                        controls=[
                            ft.Text("Inspiration Sandbox", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                            outline_primary_row,
                            spark_row,
                            quick_option_row,
                            chat_list,
                            chat_input,
                            ft.Row([send_sandbox_btn, apply_outline_btn]),
                            outlining_status,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                ),
                ft.VerticalDivider(),
                ft.Container(
                    expand=5,
                    content=ft.Column(
                        controls=[
                            ft.Text("核心策划案看板", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                            proposal_core_hook,
                            proposal_golden_finger,
                            proposal_world_tone,
                        ],
                        spacing=10,
                    ),
                ),
            ],
            expand=True,
        ),
    )
    outline_panel = ft.Container(
        visible=False,
        expand=True,
        content=ft.Column(
            controls=[
                ft.Text("Book Outline", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                outline_loading,
                ft.Row(
                    controls=[
                        ft.Container(
                            expand=True,
                            content=ft.Column(
                                controls=[
                                    ft.Row([ai_book_outline_btn, ai_book_outline_ring], wrap=True),
                                    ft.Row([outline_to_chapters_btn], alignment=ft.MainAxisAlignment.END),
                                    ft.Row(
                                        [
                                            ft.Text("全书总纲", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                                            ft.OutlinedButton("编辑", on_click=lambda _: open_fullscreen_editor("全书总纲", book_outline_box)),
                                        ]
                                    ),
                                    book_outline_box,
                                ],
                                spacing=10,
                                scroll=ft.ScrollMode.AUTO,
                                expand=True,
                            ),
                        ),
                    ],
                    expand=True,
                ),
            ],
            spacing=10,
            expand=True,
        ),
    )
    chapters_panel = ft.Container(
        visible=False,
        expand=True,
        content=ft.Row(
            controls=[
                ft.Container(
                    expand=3,
                    content=ft.Column(
                        controls=[
                            ft.Text("分卷管理台", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                            ft.Row([volume_num_input, ai_volume_outline_btn, ai_volume_outline_ring], wrap=True),
                            volume_cards_view,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                ),
                ft.VerticalDivider(),
                ft.Container(
                    expand=7,
                    content=ft.Column(
                        controls=[
                            ft.Text("章节规划", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                            book_outline_mount_status,
                            book_outline_context_box,
                            selected_volume_title,
                            ft.Row(
                                [
                                    ft.Text("分卷大纲", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                                    ft.OutlinedButton("编辑", on_click=lambda _: open_fullscreen_editor("分卷大纲", volume_outline_box)),
                                ]
                            ),
                            volume_outline_box,
                            ft.Row([chapter_count_input, ai_chapters_btn, ai_chapters_ring], wrap=True),
                            chapter_cards_view,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                ),
            ],
            expand=True,
        ),
    )

    outlining_content = ft.Container(
        padding=12,
        content=ft.Column(
            controls=[
                global_console_row,
                free_nav_row,
                concept_panel,
                outline_panel,
                chapters_panel,
            ],
            expand=True,
            spacing=10,
        ),
    )

    entities_content = ft.Container(
            padding=12,
            content=ft.Row(
                controls=[
                    ft.Container(
                        width=300,
                        content=ft.Column(
                            controls=[
                                entity_category_picker,
                                ft.Row([add_character_btn, ai_character_btn], wrap=True),
                                entity_list_view,
                            ],
                            spacing=10,
                            expand=True,
                        ),
                    ),
                    ft.VerticalDivider(),
                    ft.Container(
                        expand=True,
                        padding=30,
                        content=entity_detail_box,
                    ),
                ],
                expand=True,
            ),
        )

    pipeline_content = ft.Container(
            padding=12,
            content=ft.Column(
                controls=[
                    ft.Text("Writing Workshop", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                    ft.Row([workshop_volume_picker, chapter_picker], spacing=10),
                    chapter_num_input,
                    chapter_idea_input,
                    pipeline_run_row,
                    pipeline_progress,
                    pipeline_status,
                    ft.Container(content=pipeline_output, expand=True, bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.BLUE_GREY_900)),
                    workshop_footer_row,
                ],
                expand=True,
                spacing=10,
            ),
        )

    settings_content = ft.Container(
            padding=12,
            content=ft.Column(
                controls=[
                    ft.Text("Config & API", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                    api_key_input,
                    ft.Text("Model Manager", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                    add_model_btn,
                    model_manager_column,
                    ft.Text("Agent Mapping", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                    ft.Row([supervisor_mapping_dropdown, planner_mapping_dropdown], expand=True),
                    ft.Row([drafter_mapping_dropdown, checker_mapping_dropdown], expand=True),
                    settings_save_row,
                    settings_status,
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )

    rule_forge_content = ft.Container(
        padding=12,
        content=ft.Column(
            controls=[
                ft.Text("⚖️ 法则炼金炉", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                rule_source_input,
                ft.Row([rule_category_dropdown, element_target_input], wrap=True),
                rule_group_input,
                ft.Row(
                    controls=[
                        ft.Text("建议分组：", color=ft.Colors.BLUE_GREY_300),
                        *[
                            ft.OutlinedButton(
                                option,
                                on_click=lambda event, name=option: on_select_rule_group(event, name),
                            )
                            for option in rule_group_default_options
                        ],
                    ],
                    wrap=True,
                ),
                ft.Row(
                    [
                        extract_rule_btn,
                        save_rule_btn,
                        ft.OutlinedButton("刷新本地法则", on_click=lambda _: (reload_local_rules(), page.update())),
                    ],
                    wrap=True,
                ),
                rule_name_preview,
                ft.Text("Positive Inclusions", weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_300),
                rule_positive_box,
                ft.Text("Negative Constraints", weight=ft.FontWeight.BOLD, color=ft.Colors.RED_300),
                rule_negative_box,
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
    )

    tabs = ft.Tabs(
        length=3,
        selected_index=0,
        animation_duration=200,
        expand=True,
        content=ft.Column(
            controls=[
                ft.TabBar(
                    indicator_color=ft.Colors.BLUE_400,
                    label_color=ft.Colors.BLUE_400,
                    tabs=[
                        ft.Tab(label="创世灵感"),
                        ft.Tab(label="赛博档案"),
                        ft.Tab(label="写作工坊"),
                    ],
                ),
                ft.TabBarView(
                    expand=True,
                    controls=[
                        outlining_content,
                        entities_content,
                        pipeline_content,
                    ],
                ),
            ],
            expand=True,
            spacing=8,
        ),
    )
    tabs.on_change = on_main_tabs_change
    tabs_ref["tabs"] = tabs
    workspace_toolbar = ft.Row(
        controls=[
            ft.Button("⬅️ 返回书架", on_click=on_back_to_home),
            ft.Button("🎒 配置战术背包", on_click=on_open_rule_mount_panel),
            workspace_project_title,
            theme_toggle_btn,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )
    workspace_container = ft.Container(
        visible=False,
        expand=True,
        content=ft.Column(
            controls=[workspace_toolbar, tabs, status_label],
            spacing=8,
            expand=True,
        ),
    )
    home_projects_content = ft.Container(
        expand=True,
        padding=12,
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("NovelCraft 书架", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
                        ft.Button(
                            "➕ 创建新书",
                            on_click=on_create_new_project,
                            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                home_empty_hint,
                project_grid,
            ],
            spacing=12,
            expand=True,
        ),
    )
    home_tabs = ft.Tabs(
        length=3,
        selected_index=0,
        animation_duration=200,
        expand=True,
        on_change=on_home_tabs_change,
        content=ft.Column(
            controls=[
                ft.TabBar(
                    indicator_color=ft.Colors.BLUE_400,
                    label_color=ft.Colors.BLUE_400,
                    tabs=[
                        ft.Tab(label="📚 我的书架"),
                        ft.Tab(label="⚖️ 法则炼金炉"),
                        ft.Tab(label="⚙️ 系统设置"),
                    ],
                ),
                ft.TabBarView(
                    expand=True,
                    controls=[
                        home_projects_content,
                        rule_forge_content,
                        settings_content,
                    ],
                ),
            ],
            expand=True,
            spacing=8,
        ),
    )
    home_tabs_ref["tabs"] = home_tabs
    home_container = ft.Container(visible=True, expand=True, content=home_tabs)
    home_view_ref["view"] = home_container
    workspace_view_ref["view"] = workspace_container

    refresh_entity_view()
    refresh_chat_view(persist=False)
    render_quick_options()
    sync_proposal_board()
    sync_active_traits_inputs()
    render_volume_cards()
    sync_outline_visibility()
    render_chapter_cards()
    render_model_manager()
    sync_mapping_dropdowns()
    refresh_archive_options()
    set_stage(STAGE_CONCEPT)
    sync_theme_toggle()
    reload_local_rules()
    render_rule_result(None)
    render_project_shelf()
    page.add(ft.Column(controls=[home_container, workspace_container], spacing=8, expand=True))


if __name__ == "__main__":
    ft.run(main)
