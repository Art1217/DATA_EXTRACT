"""Utilidades compartidas por los routers."""

from __future__ import annotations

from app.api.schemas import JobAccepted
from app.domain.analysis import get_analysis_type
from app.domain.errors import DocumentNotFoundError
from app.jobs.models import Job
from app.storage import keys
from app.storage.base import Storage

JOBS_PREFIX = "/v1/jobs"


def accepted(job: Job) -> JobAccepted:
    """Respuesta estándar de toda operación asíncrona."""
    return JobAccepted(
        job_id=job.job_id,
        status=job.status,
        type=job.type,
        document_id=job.document_id,
        analysis_type=job.analysis_type,
        status_url=f"{JOBS_PREFIX}/{job.job_id}",
        events_url=f"{JOBS_PREFIX}/{job.job_id}/events",
    )


def ensure_document(storage: Storage, document_id: str) -> str:
    """Falla con 404 antes de encolar trabajo sobre un documento inexistente."""
    if not storage.exists(keys.document_meta(document_id)):
        raise DocumentNotFoundError(document_id)
    return document_id


def ensure_analysis(analysis_type: str) -> str:
    """Valida la clave contra el catálogo antes de encolar."""
    return get_analysis_type(analysis_type).key
