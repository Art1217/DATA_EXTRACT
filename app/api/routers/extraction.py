"""Etapa de extracción: del recorte a filas estructuradas."""

from __future__ import annotations

import io

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.api.deps import RunnerDep, StorageDep
from app.api.routers.common import accepted, ensure_analysis, ensure_document
from app.api.schemas import ExtractionRequest, JobAccepted
from app.domain.errors import StageNotReadyError
from app.domain.models import ExtractionResult
from app.jobs.models import JobType
from app.services import export
from app.storage import keys

router = APIRouter(prefix="/v1/documents/{document_id}", tags=["extracción"])

_XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.post("/extraction", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def extract_data(
    document_id: str, request: ExtractionRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Extrae las filas del recorte de un análisis.

    Es la etapa cara: usa el modelo grande sobre cada página del recorte.
    """
    ensure_document(storage, document_id)
    analysis_type = ensure_analysis(request.analysis_type)
    job = runner.submit(
        JobType.EXTRACTION,
        document_id=document_id,
        analysis_type=analysis_type,
        payload={
            "include_identity": request.include_identity,
            "source_key": request.source_key,
        },
    )
    return accepted(job)


@router.get("/extraction/{analysis_type}")
def get_extraction(document_id: str, analysis_type: str, storage: StorageDep) -> dict:
    return _require(storage, document_id, analysis_type)


@router.get("/extraction/{analysis_type}/xlsx")
def download_excel(document_id: str, analysis_type: str, storage: StorageDep) -> StreamingResponse:
    """Excel con las columnas de identidad ya incorporadas a cada fila.

    Si la publicación ya lo generó y guardó (`extraction/<análisis>/result.xlsx`)
    se sirve ese mismo archivo: la descarga y el objeto del bucket son
    entonces byte a byte el mismo entregable. Si no está, se genera al vuelo.
    """
    stored = keys.extraction_xlsx(document_id, analysis_type)
    if storage.exists(stored):
        workbook: bytes | None = storage.read_bytes(stored)
    else:
        result = ExtractionResult.model_validate(_require(storage, document_id, analysis_type))
        workbook = export.to_excel_bytes(result.rows, result.identity)
    if workbook is None:
        raise StageNotReadyError(
            "extraction",
            requires="extraction",
            hint="La extracción no produjo filas que exportar.",
        )
    return StreamingResponse(
        io.BytesIO(workbook),
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{analysis_type}.xlsx"'},
    )


@router.get("/extraction/{analysis_type}/docai")
def get_docai_audit(document_id: str, analysis_type: str, storage: StorageDep) -> dict:
    """Salida cruda de Document AI, para auditar de dónde salió cada dato."""
    payload = storage.read_json_or_none(keys.extraction_docai(document_id, analysis_type))
    if payload is None:
        raise StageNotReadyError(
            "extraction",
            requires="extraction",
            hint="Este análisis no generó salida de Document AI.",
        )
    return payload


def _require(storage, document_id: str, analysis_type: str) -> dict:
    payload = storage.read_json_or_none(keys.extraction_result(document_id, analysis_type))
    if payload is None:
        raise StageNotReadyError(
            "extraction",
            requires="extraction",
            hint=f"Lanza POST /extraction con analysis_type='{analysis_type}'.",
        )
    return payload
