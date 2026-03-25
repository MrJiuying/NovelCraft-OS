import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def projects_dir() -> Path:
    path = project_root() / "data" / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_project_name(name: str) -> str:
    cleaned = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in str(name or "").strip())
    return cleaned or "未命名作品"


def project_dir(project_name: str) -> Path:
    path = projects_dir() / sanitize_project_name(project_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def meta_path(project_name: str) -> Path:
    return project_dir(project_name) / "meta.json"


def chapters_dir(project_name: str) -> Path:
    path = project_dir(project_name) / "chapters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outlines_dir(project_name: str) -> Path:
    path = project_dir(project_name) / "outlines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def characters_dir(project_name: str) -> Path:
    path = project_dir(project_name) / "characters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def world_dir(project_name: str) -> Path:
    path = project_dir(project_name) / "world"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chapter_relative_path(volume_num: int, chapter_num: int) -> str:
    return f"chapters/vol_{volume_num}_ch_{chapter_num}.md"


def _extract_key_num(key: str) -> int:
    try:
        return int(str(key))
    except Exception:
        return 0


def _minimal_meta(project_name: str, snapshot: dict[str, Any], save_stage: str) -> dict[str, Any]:
    book = snapshot.get("book_outline") if isinstance(snapshot.get("book_outline"), dict) else {}
    concept = snapshot.get("current_concept") if isinstance(snapshot.get("current_concept"), dict) else {}
    return {
        "project_name": str(project_name),
        "saved_at": datetime.now().isoformat(),
        "save_stage": str(save_stage or "CONCEPT"),
        "book_title": str(book.get("book_title", "")),
        "book_logline": str(book.get("logline", "")),
        "target_word_count": str(book.get("target_word_count", "")),
        "core_power_system": str(book.get("core_power_system", "")),
        "main_storyline": str(book.get("main_storyline", "")),
        "core_hook": str(concept.get("core_hook", "")),
        "golden_finger": str(concept.get("golden_finger", "")),
        "world_tone": str(concept.get("world_tone", "")),
        "book_outline": book if isinstance(book, dict) else {},
        "current_concept": concept if isinstance(concept, dict) else {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _normalize_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _collect_chapter_files(snapshot: dict[str, Any]) -> dict[str, str]:
    chapter_files = {}
    raw = snapshot.get("chapter_files")
    if isinstance(raw, dict):
        for key, value in raw.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            chapter_files[key_text] = str(value)
    chapters = snapshot.get("chapter_outlines")
    if isinstance(chapters, dict):
        for volume_key, chapter_list in chapters.items():
            try:
                volume_num = int(str(volume_key))
            except Exception:
                continue
            if not isinstance(chapter_list, dict):
                continue
            rows = chapter_list.get("chapters")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    chapter_num = int(row.get("chapter_number", 0))
                except Exception:
                    continue
                if chapter_num <= 0:
                    continue
                key = f"{volume_num}:{chapter_num}"
                chapter_files.setdefault(key, chapter_relative_path(volume_num, chapter_num))
    return chapter_files


def save_project_snapshot(project_name: str, snapshot: dict[str, Any], save_stage: str) -> Path:
    root = project_dir(project_name)
    chapters_dir(project_name)
    outlines = outlines_dir(project_name)
    chars = characters_dir(project_name)
    world = world_dir(project_name)
    meta = _minimal_meta(project_name, snapshot, save_stage)
    _write_json(root / "meta.json", meta)
    volumes = _normalize_dict(snapshot.get("volumes"))
    chapter_outlines = _normalize_dict(snapshot.get("chapter_outlines"))
    expected_outline_names: set[str] = set()
    volume_keys = sorted({_extract_key_num(k) for k in list(volumes.keys()) + list(chapter_outlines.keys()) if _extract_key_num(k) > 0})
    for volume_num in volume_keys:
        key = str(volume_num)
        payload = {
            "volume_number": volume_num,
            "volume_outline": volumes.get(key) if isinstance(volumes.get(key), dict) else {},
            "chapter_outline": chapter_outlines.get(key) if isinstance(chapter_outlines.get(key), dict) else {},
        }
        filename = f"vol_{volume_num}.json"
        expected_outline_names.add(filename)
        _write_json(outlines / filename, payload)
    for file in outlines.glob("vol_*.json"):
        if file.name not in expected_outline_names:
            file.unlink(missing_ok=True)
    characters = _normalize_dict(snapshot.get("characters"))
    expected_character_names: set[str] = set()
    for entity_id, payload in characters.items():
        name = f"{entity_id}.json"
        expected_character_names.add(name)
        _write_json(chars / name, {"entity_id": str(entity_id), "payload": payload if isinstance(payload, dict) else {}})
    for file in chars.glob("*.json"):
        if file.name not in expected_character_names:
            file.unlink(missing_ok=True)
    world_groups = {
        "world_settings": _normalize_dict(snapshot.get("world_settings")),
        "factions": _normalize_dict(snapshot.get("factions")),
        "items": _normalize_dict(snapshot.get("items")),
        "timeline_events": _normalize_dict(snapshot.get("timeline_events")),
    }
    expected_world_names: set[str] = set()
    for group, bucket in world_groups.items():
        for entity_id, payload in bucket.items():
            filename = f"{group}__{entity_id}.json"
            expected_world_names.add(filename)
            _write_json(world / filename, {"group": group, "entity_id": str(entity_id), "payload": payload if isinstance(payload, dict) else {}})
    for file in world.glob("*.json"):
        if file.name not in expected_world_names:
            file.unlink(missing_ok=True)
    return root / "meta.json"


def _is_legacy_meta(payload: dict[str, Any]) -> bool:
    if "app_state" in payload:
        return True
    legacy_keys = {"volumes", "chapter_outlines", "characters", "world_settings", "factions", "items", "timeline_events"}
    return any(key in payload for key in legacy_keys)


def migrate_legacy_meta(project_name: str, payload: dict[str, Any]) -> None:
    snapshot = payload.get("app_state") if isinstance(payload.get("app_state"), dict) else payload
    app_snapshot = {
        "book_outline": snapshot.get("book_outline") if isinstance(snapshot.get("book_outline"), dict) else {},
        "current_concept": snapshot.get("current_concept") if isinstance(snapshot.get("current_concept"), dict) else {},
        "volumes": snapshot.get("volumes") if isinstance(snapshot.get("volumes"), dict) else {},
        "chapter_outlines": snapshot.get("chapter_outlines") if isinstance(snapshot.get("chapter_outlines"), dict) else {},
        "characters": snapshot.get("characters") if isinstance(snapshot.get("characters"), dict) else {},
        "world_settings": snapshot.get("world_settings") if isinstance(snapshot.get("world_settings"), dict) else {},
        "factions": snapshot.get("factions") if isinstance(snapshot.get("factions"), dict) else {},
        "items": snapshot.get("items") if isinstance(snapshot.get("items"), dict) else {},
        "timeline_events": snapshot.get("timeline_events") if isinstance(snapshot.get("timeline_events"), dict) else {},
        "chapter_files": _collect_chapter_files(snapshot),
    }
    save_project_snapshot(
        project_name=str(payload.get("project_name") or project_name),
        snapshot=app_snapshot,
        save_stage=str(payload.get("save_stage") or snapshot.get("save_stage") or "CONCEPT"),
    )


def load_project_snapshot(project_name: str) -> dict[str, Any]:
    root = project_dir(project_name)
    meta_file = root / "meta.json"
    if not meta_file.exists():
        return {"project_name": project_name, "save_stage": "CONCEPT", "app_state": {}}
    meta = _read_json(meta_file) or {}
    if _is_legacy_meta(meta):
        migrate_legacy_meta(project_name, meta)
        meta = _read_json(meta_file) or {}
    outlines = {}
    chapter_outlines = {}
    for file in sorted((root / "outlines").glob("vol_*.json"), key=lambda p: p.name):
        payload = _read_json(file)
        if not payload:
            continue
        volume_num = payload.get("volume_number")
        try:
            volume_num_int = int(volume_num)
        except Exception:
            matched = re.search(r"vol_(\d+)\.json$", file.name)
            if not matched:
                continue
            volume_num_int = int(matched.group(1))
        vol = payload.get("volume_outline")
        ch = payload.get("chapter_outline")
        if isinstance(vol, dict):
            outlines[str(volume_num_int)] = vol
        if isinstance(ch, dict):
            chapter_outlines[str(volume_num_int)] = ch
    characters = {}
    for file in sorted((root / "characters").glob("*.json"), key=lambda p: p.name):
        payload = _read_json(file)
        if not payload:
            continue
        entity_id = str(payload.get("entity_id") or file.stem)
        body = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        characters[entity_id] = body
    world_settings: dict[str, Any] = {}
    factions: dict[str, Any] = {}
    items: dict[str, Any] = {}
    timeline_events: dict[str, Any] = {}
    for file in sorted((root / "world").glob("*.json"), key=lambda p: p.name):
        payload = _read_json(file)
        if not payload:
            continue
        group = str(payload.get("group") or "")
        entity_id = str(payload.get("entity_id") or file.stem)
        body = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        if group == "world_settings":
            world_settings[entity_id] = body
        elif group == "factions":
            factions[entity_id] = body
        elif group == "items":
            items[entity_id] = body
        elif group == "timeline_events":
            timeline_events[entity_id] = body
    app_state = {
        "chat_history": [],
        "quick_options": [],
        "current_concept": meta.get("current_concept") if isinstance(meta.get("current_concept"), dict) else {},
        "book_outline": meta.get("book_outline") if isinstance(meta.get("book_outline"), dict) else {},
        "volumes": outlines,
        "chapter_outlines": chapter_outlines,
        "chapter_files": _collect_chapter_files({"chapter_outlines": chapter_outlines}),
        "selected_volume_num": max([_extract_key_num(k) for k in outlines.keys()] or [0]) or None,
        "active_traits": None,
        "characters": characters,
        "world_settings": world_settings,
        "factions": factions,
        "items": items,
        "timeline_events": timeline_events,
        "mounted_rule_ids": [],
    }
    return {
        "project_name": str(meta.get("project_name") or project_name),
        "save_stage": str(meta.get("save_stage") or "CONCEPT"),
        "saved_at": str(meta.get("saved_at") or ""),
        "app_state": app_state,
    }


def list_project_refs() -> list[tuple[Path, str, str]]:
    result: list[tuple[Path, str, str]] = []
    for path in sorted(projects_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        meta = _read_json(path / "meta.json")
        if not meta:
            continue
        project_name = str(meta.get("project_name") or path.name)
        saved_at = str(meta.get("saved_at") or datetime.fromtimestamp((path / "meta.json").stat().st_mtime).isoformat())
        result.append((path, project_name, saved_at))
    return result


def delete_chapter_markdown(project_name: str, volume_num: int, chapter_num: int, chapter_files: dict[str, str]) -> None:
    key = f"{volume_num}:{chapter_num}"
    candidates = [chapter_files.get(key, ""), chapter_relative_path(volume_num, chapter_num), f"chapters/v{volume_num}_c{chapter_num}.md"]
    for relative in candidates:
        if not relative:
            continue
        target = project_dir(project_name) / relative
        if target.exists():
            target.unlink(missing_ok=True)


def delete_volume_assets(project_name: str, volume_num: int, chapter_numbers: list[int], chapter_files: dict[str, str]) -> None:
    outline_file = outlines_dir(project_name) / f"vol_{volume_num}.json"
    if outline_file.exists():
        outline_file.unlink(missing_ok=True)
    for chapter_num in chapter_numbers:
        delete_chapter_markdown(project_name, volume_num, chapter_num, chapter_files)
