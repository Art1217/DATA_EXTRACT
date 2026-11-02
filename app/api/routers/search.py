"""Etapa de búsqueda y recorte del PDF.

El recorte se expone además como operación propia: permite corregir a mano la
selección de páginas y volver a extraer sin repetir la búsqueda.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.api.deps import RunnerDep, StorageDep
from app.api.routers.common import accepted, ensure_analysis, ensure_document
from app.api.schemas import CutRequest, JobAccepted, SearchRequest
from app.domain.errors import StageNotReadyError
from app.jobs.models import JobType
from app.storage import keys

router = APIRouter(prefix="/v1/documents/{document_id}", tags=["búsqueda"])


@router.post("/search", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def search_pages(
    document_id: str, request: SearchRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Localiza las páginas relevantes para un análisis y genera el recorte."""
    ensure_document(storage, document_id)
    analysis_type = ensure_analysis(request.analysis_type)
    payload = request.as_payload() | {
        "use_identity": request.use_identity,
        "cut": request.cut,
    }
    payload.pop("analysis_type", None)
    job = runner.submit(
        JobType.SEARCH,
        document_id=document_id,
        analysis_type=analysis_type,
        payload=payload,
    )
    return accepted(job)


@router.get("/search/{analysis_type}")
def get_search_result(document_id: str, analysis_type: str, storage: StorageDep) -> dict:
    payload = storage.read_json_or_none(keys.search_result(document_id, analysis_type))
    if payload is None:
        raise StageNotReadyError(
            "search",
            requires="search",
            hint=f"Lanza POST /search con analysis_type='{analysis_type}'.",
        )
    return payload


@router.get("/search/{analysis_type}/pdf")
def download_cut_pdf(
    document_id: str, analysis_type: str, storage: StorageDep, ocr: bool = False
) -> StreamingResponse:
    """Descarga el PDF recortado. Con `ocr=true`, la versión con capa de texto."""
    key = (
        keys.search_pdf_searchable(document_id, analysis_type)
        if ocr
        else keys.search_pdf(document_id, analysis_type)
    )
    if not storage.exists(key):
        raise StageNotReadyError(
            "search",
            requires="search",
            hint="El recorte todavía no existe para este análisis.",
        )
    return StreamingResponse(
        storage.open(key),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="corte_{analysis_type}.pdf"'},
    )


@router.post("/cut", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def cut_pages(
    document_id: str, request: CutRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Recorta el documento a una lista de páginas dada por el cliente."""
    ensure_document(storage, document_id)
    analysis_type = ensure_analysis(request.analysis_type)
    job = runner.submit(
        JobType.CUT,
        document_id=document_id,
        analysis_type=analysis_type,
        payload={"pages": request.pages},
    )
    return accepted(job)
