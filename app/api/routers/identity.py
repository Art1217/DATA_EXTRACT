"""Etapa de identidad: metadata general del documento.

Se calcula una vez y la reutilizan todos los análisis. El `PATCH` existe
porque estos campos se copian a cada fila del resultado final: poder
corregirlos a mano evita repetir extracciones enteras por un dato mal leído
en la portada.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import ReadContextDep, RunnerDep, StorageDep
from app.api.routers.common import accepted, ensure_document
from app.api.schemas import IdentityRequest, JobAccepted
from app.domain.errors import StageNotReadyError
from app.jobs.models import JobType
from app.services import identity as identity_service

router = APIRouter(prefix="/v1/documents/{document_id}/identity", tags=["identidad"])


@router.post("", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def extract_identity(
    document_id: str, request: IdentityRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Extrae la identidad a partir de las primeras páginas del documento."""
    ensure_document(storage, document_id)
    job = runner.submit(
        JobType.IDENTITY,
        document_id=document_id,
        payload=request.model_dump(exclude_none=True),
    )
    return accepted(job)


@router.get("")
def get_identity(document_id: str, ctx: ReadContextDep) -> dict:
    identity = identity_service.get_identity(ctx, document_id)
    if identity is None:
        raise StageNotReadyError(
            "identity",
            requires="identity",
            hint="Lanza POST /identity para calcularla.",
        )
    return identity


@router.patch("")
def update_identity(
    document_id: str, identity: dict[str, Any], ctx: ReadContextDep, storage: StorageDep
) -> dict:
    """Sustituye la identidad por una corregida a mano."""
    ensure_document(storage, document_id)
    return identity_service.save_identity(ctx, document_id, identity)
