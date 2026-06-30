"""SQLite-backed memory integration for ResearchSwarm.

This adapter reads and writes the local AI-Memory database so routing and
execution events can be persisted alongside the rest of the source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any


ALLOWED_FILE_TYPES = {
    "CONTEXT",
    "DECISION",
    "PROGRESS",
    "PATTERN",
    "BRIEF",
    "RESEARCH_REPORT",
    "PLAN_REPORT",
    "EXECUTION_REPORT",
}


DEFAULT_MEMORY_DB_PATH = Path(__file__).resolve().parent / "AI-Memory" / "memory.db"


@dataclass
class MemoryEntry:
    file_type: str
    timestamp: str
    tag: str
    content: str
    metadata: dict[str, Any]
    phase: str | None = None
    progress_status: str | None = None

    def to_line(self) -> str:
        payload = self.content.strip().replace("\n", " ")
        if len(payload) > 160:
            payload = payload[:157] + "..."
        return f"{self.file_type} | {self.tag} | {payload}"


class ResearchSwarmMemoryStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_type TEXT NOT NULL CHECK(file_type IN ('CONTEXT', 'DECISION', 'PROGRESS', 'PATTERN', 'BRIEF', 'RESEARCH_REPORT', 'PLAN_REPORT', 'EXECUTION_REPORT')),
                    timestamp TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    phase TEXT DEFAULT NULL,
                    progress_status TEXT DEFAULT NULL,
                    embedding BLOB DEFAULT NULL
                )
                """
            )

    def record_entry(
        self,
        file_type: str,
        tag: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        phase: str | None = None,
        progress_status: str | None = None,
    ) -> int:
        if file_type not in ALLOWED_FILE_TYPES:
            raise ValueError(f"Unsupported memory file_type: {file_type}")

        payload = json.dumps(metadata or {}, ensure_ascii=False)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO entries (file_type, timestamp, tag, content, metadata, phase, progress_status, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_type, timestamp, tag, content, payload, phase, progress_status, None),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_entries(self, limit: int = 5) -> list[MemoryEntry]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT file_type, timestamp, tag, content, metadata, phase, progress_status
                FROM entries
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        entries: list[MemoryEntry] = []
        for row in rows:
            metadata_value = row["metadata"] or "{}"
            try:
                metadata = json.loads(metadata_value)
            except json.JSONDecodeError:
                metadata = {"raw": metadata_value}
            entries.append(
                MemoryEntry(
                    file_type=row["file_type"],
                    timestamp=row["timestamp"],
                    tag=row["tag"],
                    content=row["content"],
                    metadata=metadata,
                    phase=row["phase"],
                    progress_status=row["progress_status"],
                )
            )
        return entries

    def recent_context_lines(self, limit: int = 3) -> list[str]:
        return [entry.to_line() for entry in self.recent_entries(limit=limit)]

    def record_route(self, report: dict[str, Any]) -> int:
        return self.record_entry(
            "DECISION",
            tag=f"ROUTE:{report['domain']}:{report['status']}",
            content=report["summary"],
            metadata={
                "instruction": report["instruction"],
                "domain": report["domain"],
                "status": report["status"],
                "next_action": report.get("next_action", ""),
                "digital_work": report.get("digital_work", []),
                "human_handoff": report.get("human_handoff", []),
            },
            phase="route",
            progress_status=report["status"],
        )

    def record_execution(self, report: dict[str, Any]) -> int:
        file_type = "EXECUTION_REPORT" if report.get("executor_name") else "PROGRESS"
        return self.record_entry(
            file_type,
            tag=f"EXEC:{report['domain']}:{report['status']}",
            content=report.get("execution_summary") or report.get("summary") or report["instruction"],
            metadata={
                "instruction": report["instruction"],
                "domain": report["domain"],
                "status": report["status"],
                "executor_name": report.get("executor_name", ""),
                "execution_summary": report.get("execution_summary", ""),
            },
            phase="execute",
            progress_status=report["status"],
        )