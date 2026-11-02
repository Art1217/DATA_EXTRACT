"""Etapa de control de calidad.

Separada de la extracción a propósito: permite corregir datos a mano y
revalidarlos sin pagar otra extracción.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import ReadContextDep, RunnerDep, StorageDep
from app.api.routers.common import accepted, ensure_analysis, ensure_document
from app.api.schemas import JobAccepted, QualityCheckRequest, QualityRequest
from app.domain.analysis import get_analysis_type
from app.domain.errors import StageNotReadyError
from app.jobs.models import JobType
from app.services import quality as quality_service
from app.storage import keys

router = APIRouter(tags=["calidad"])


@router.post(
    "/v1/documents/{document_id}/quality",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def check_document(
    document_id: str, request: QualityRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Valida la extracción guardada de un análisis."""
    ensure_document(storage, document_id)
    analysis_type = ensure_analysis(request.analysis_type)
    job = runner.submit(JobType.QUALITY, document_id=document_id, analysis_type=analysis_type)
    return accepted(job)


@router.get("/v1/documents/{document_id}/quality/{analysis_type}")
def get_quality(document_id: str, analysis_type: str, storage: StorageDep) -> dict:
    payload = storage.read_json_or_none(keys.quality_result(document_id, analysis_type))
    if payload is None:
        raise StageNotReadyError(
            "quality",
            requires="quality",
            hint=f"Lanza POST /quality con analysis_type='{analysis_type}'.",
        )
    return payload


@router.post("/v1/quality/check")
def check_rows(request: QualityCheckRequest, ctx: ReadContextDep) -> dict:
    """Valida filas enviadas por el cliente, sin documento asociado.

    Es síncrona porque es una sola llamada al modelo sobre datos ya
    estructurados: no hay que recorrer el documento.
    """
    analysis = get_analysis_type(request.analysis_type)
    result = quality_service.check_rows(ctx, analysis, request.rows)
    return result.model_dump(mode="json")
