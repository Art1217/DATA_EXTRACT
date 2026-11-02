"""Modelo de un trabajo asíncrono."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


class JobType(StrEnum):
    """Cada tipo se corresponde con una etapa expuesta como endpoint."""

    CLASSIFY_GROUP = "classify_group"
    CLASSIFY_ANALYSIS = "classify_analysis"
    PLAN = "plan"
    IDENTITY = "identity"
    SEARCH = "search"
    CUT = "cut"
    EXTRACTION = "extraction"
    QUALITY = "quality"
    PIPELINE = "pipeline"
    AUTO = "auto"
    BUNDLE = "bundle"
    BUNDLE_BATCH = "bundle_batch"


class Job(BaseModel):
    """Estado observable de un trabajo, desde que se encola hasta que termina."""

    job_id: str
    type: JobType
    status: JobStatus = JobStatus.QUEUED
    document_id: str | None = None
    analysis_type: str | None = None

    progress: float = 0.0
    message: str = "En cola."

    payload: dict[str, Any] = Field(default_factory=dict)
    """Parámetros con los que se lanzó. Permite reproducir la ejecución."""

    result: Any | None = None
    error: str | None = None
    parent_job_id: str | None = None

    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or _now()
        return (end - self.started_at).total_seconds()


class JobEvent(BaseModel):
    """Notificación de cambio de estado, para el flujo SSE."""

    job_id: str
    status: JobStatus
    progress: float
    message: str
    timestamp: datetime = Field(default_factory=_now)
