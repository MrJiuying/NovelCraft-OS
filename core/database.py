import json
import sqlite3
from pathlib import Path

from core.schemas import CharacterCardRecord, NLPBaseTraits


DB_PATH = Path(__file__).resolve().parent / "character_traits.db"
DEFAULT_ENTITY_ID = "protagonist"


def _get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_character_traits_table() -> None:
    with _get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS character_traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                version_chapter INTEGER NOT NULL,
                traits_json TEXT NOT NULL,
                update_reason TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_character_traits_entity_chapter
            ON character_traits (entity_id, version_chapter DESC)
            """
        )
        connection.commit()


def save_character_version(record: CharacterCardRecord) -> None:
    init_character_traits_table()
    with _get_connection() as connection:
        connection.execute(
            """
            INSERT INTO character_traits (entity_id, version_chapter, traits_json, update_reason)
            VALUES (?, ?, ?, ?)
            """,
            (
                record.entity_id,
                record.version_chapter,
                json.dumps(record.traits.model_dump(), ensure_ascii=False),
                record.update_reason or "初始设定",
            ),
        )
        connection.commit()


def get_character_at_chapter(entity_id: str, chapter: int) -> CharacterCardRecord:
    init_character_traits_table()
    with _get_connection() as connection:
        row = connection.execute(
            """
            SELECT entity_id, version_chapter, traits_json, update_reason
            FROM character_traits
            WHERE entity_id = ? AND version_chapter <= ?
            ORDER BY version_chapter DESC, id DESC
            LIMIT 1
            """,
            (entity_id, chapter),
        ).fetchone()

    if row is None:
        raise LookupError(f"未找到角色 {entity_id} 在第 {chapter} 章及之前的设定记录")

    return CharacterCardRecord(
        entity_id=row["entity_id"],
        version_chapter=row["version_chapter"],
        traits=json.loads(row["traits_json"]),
        update_reason=row["update_reason"],
    )


def load_default_traits(entity_id: str = DEFAULT_ENTITY_ID) -> NLPBaseTraits:
    init_character_traits_table()
    with _get_connection() as connection:
        row = connection.execute(
            """
            SELECT traits_json
            FROM character_traits
            WHERE entity_id = ?
            ORDER BY version_chapter DESC, id DESC
            LIMIT 1
            """,
            (entity_id,),
        ).fetchone()
        if row is None:
            row = connection.execute(
                """
                SELECT traits_json
                FROM character_traits
                ORDER BY version_chapter DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
    if row is None:
        return NLPBaseTraits(
            environment="",
            behavior="",
            capability="",
            values="",
            identity="",
            vision="",
        )
    return NLPBaseTraits.model_validate(json.loads(row["traits_json"]))


def save_active_traits(
    traits: NLPBaseTraits,
    entity_id: str = DEFAULT_ENTITY_ID,
    update_reason: str = "UI保存",
) -> CharacterCardRecord:
    init_character_traits_table()
    with _get_connection() as connection:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(version_chapter), 0) AS max_version
            FROM character_traits
            WHERE entity_id = ?
            """,
            (entity_id,),
        ).fetchone()
    next_version = int(row["max_version"]) + 1 if row else 1
    record = CharacterCardRecord(
        entity_id=entity_id,
        version_chapter=next_version,
        traits=traits,
        update_reason=update_reason,
    )
    save_character_version(record)
    return record
