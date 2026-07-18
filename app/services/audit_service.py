"""SQLite-backed audit logging service."""

import sqlite3
from contextlib import closing
from pathlib import Path

from app.models import (
    AuditCreate,
    AuditDetailsResponse,
    AuditRetrievalStatus,
    AuditStatus,
    AuditSummaryResponse,
)


class AuditService:
    """Encapsulate persistence for audit records."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the audit service and ensure the database table exists."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create the audit table if it does not already exist."""
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT,
                    model TEXT,
                    retrieval_status TEXT,
                    top_distance REAL,
                    retrieved_chunks INTEGER,
                    response_time_ms INTEGER NOT NULL,
                    verification TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            connection.commit()

    def log(self, record: AuditCreate) -> int:
        """Persist an audit record and return the generated row ID."""
        data = record.model_dump(mode="json")

        with closing(sqlite3.connect(self.db_path)) as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_logs (
                    timestamp,
                    query,
                    answer,
                    model,
                    retrieval_status,
                    top_distance,
                    retrieved_chunks,
                    response_time_ms,
                    verification,
                    status,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["timestamp"],
                    data["query"],
                    data["answer"],
                    data["model"],
                    data["retrieval_status"],
                    data["top_distance"],
                    data["retrieved_chunks"],
                    data["response_time_ms"],
                    data["verification"],
                    data["status"],
                    data["error_message"],
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_recent(self, limit: int = 20, offset: int = 0) -> list[AuditSummaryResponse]:
        """Return recent audit summaries ordered by newest timestamp first."""
        if limit < 1:
            return []

        offset = max(offset, 0)

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    id,
                    timestamp,
                    query,
                    status,
                    retrieval_status,
                    model,
                    response_time_ms
                FROM audit_logs
                ORDER BY timestamp DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [self._row_to_summary(row) for row in rows]

    def search(
        self,
        status: AuditStatus | None = None,
        retrieval_status: AuditRetrievalStatus | None = None,
        model: str | None = None,
        min_response_time_ms: int | None = None,
        max_response_time_ms: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AuditSummaryResponse]:
        """Return audit summaries matching the provided optional filters."""
        if limit < 1:
            return []

        offset = max(offset, 0)
        conditions: list[str] = []
        parameters: list[object] = []

        if status is not None:
            conditions.append("status = ?")
            parameters.append(status.value)

        if retrieval_status is not None:
            conditions.append("retrieval_status = ?")
            parameters.append(retrieval_status.value)

        if model is not None:
            conditions.append("model = ?")
            parameters.append(model)

        if min_response_time_ms is not None:
            conditions.append("response_time_ms >= ?")
            parameters.append(min_response_time_ms)

        if max_response_time_ms is not None:
            conditions.append("response_time_ms <= ?")
            parameters.append(max_response_time_ms)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.extend([limit, offset])

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"""
                SELECT
                    id,
                    timestamp,
                    query,
                    status,
                    retrieval_status,
                    model,
                    response_time_ms
                FROM audit_logs
                {where_clause}
                ORDER BY timestamp DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                tuple(parameters),
            ).fetchall()

        return [self._row_to_summary(row) for row in rows]

    def get_by_id(self, audit_id: int) -> AuditDetailsResponse | None:
        """Return detailed audit data by ID, or None when it does not exist."""
        if audit_id < 1:
            return None

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    id,
                    timestamp,
                    query,
                    answer,
                    model,
                    retrieval_status,
                    top_distance,
                    retrieved_chunks,
                    response_time_ms,
                    verification AS verification_status,
                    status,
                    error_message
                FROM audit_logs
                WHERE id = ?
                """,
                (audit_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_details(row)

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> AuditSummaryResponse:
        """Map a SQLite row into an AuditSummaryResponse model."""
        return AuditSummaryResponse(**dict(row))

    @staticmethod
    def _row_to_details(row: sqlite3.Row) -> AuditDetailsResponse:
        """Map a SQLite row into an AuditDetailsResponse model."""
        return AuditDetailsResponse(**dict(row))
