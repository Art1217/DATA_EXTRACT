"""Etapa de extracción: convertir el recorte en filas estructuradas.

Es la etapa cara del pipeline —usa el modelo grande y procesa cada página—,
así que su resultado se persiste con clave propia y las etapas siguientes lo
reutilizan en vez de volver a pedirlo.

La extracción trabaja siempre sobre una copia del recorte en el workspace, no
sobre el artefacto guardado: cuando el PDF es escaneado, la rutina de
extracción reescribe el archivo de entrada con la capa de texto del OCR. Esa
versión sí interesa conservarla, pero como artefacto aparte
(`corte_searchable.pdf`), no mutando el original que otras etapas pueden estar
leyendo.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.adapters.ai_extractor import extract_structured_data_chunked
from app.domain.analysis import AnalysisType
from app.domain.errors import StageNotReadyError, StageNotSupportedError
from app.domain.models import ExtractionResult
from app.services.context import StageContext
from app.services.limits import llm_slot
from app.storage import keys

logger = logging.getLogger(__name__)


def extract_data(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    *,
    identity: dict | None = None,
    source_key: str | None = None,
) -> ExtractionResult:
    """Extrae las filas del recorte asociado a `analysis`."""
    if not analysis.supports_extraction:
        raise StageNotSupportedError("extraction", analysis.key)

    source_key = source_key or keys.search_pdf(document_id, analysis.key)
    if not ctx.storage.exists(source_key):
        raise StageNotReadyError(
            "extraction",
            requires="search",
            hint=f"Ejecuta la búsqueda de '{analysis.key}' o envía las páginas a recortar.",
        )

    prompt = str(ctx.settings.prompt_path(analysis.extract_prompt))
    working_copy = _stage_source(ctx, source_key)

    ctx.progress.report(f"Extrayendo datos de {analysis.label}...", 0.1)
    ctx.cancellation.raise_if_cancelled()
    with llm_slot():
        raw_rows, docai_document = extract_structured_data_chunked(
            str(working_copy),
            prompt,
            prefer_docai_json=analysis.prefer_docai_json,
            rows_key=analysis.rows_key,
        )

    rows = _normalize_rows(raw_rows, analysis.key)

    docai_key: str | None = None
    if docai_document:
        docai_key = ctx.storage.write_json(
            keys.extraction_docai(document_id, analysis.key), docai_document
        )
        # El PDF quedó enriquecido con la capa OCR: se conserva aparte, sin
        # tocar el recorte original.
        ctx.storage.import_file(
            working_copy, keys.search_pdf_searchable(document_id, analysis.key), move=True
        )

    result = ExtractionResult(
        analysis_type=analysis.key,
        rows=rows,
        identity=identity,
        row_count=len(rows),
        source_pages=_pages_of(ctx, document_id, analysis.key),
        docai_key=docai_key,
    )
    ctx.storage.write_json(
        keys.extraction_result(document_id, analysis.key), result.model_dump(mode="json")
    )
    ctx.progress.report(f"Extracción terminada: {len(rows)} filas.", 1.0)
    logger.info("Extracción %s en %s -> %d filas", analysis.key, document_id, len(rows))
    return result


def get_extraction(
    ctx: StageContext, document_id: str, analysis_key: str
) -> ExtractionResult | None:
    payload = ctx.storage.read_json_or_none(keys.extraction_result(document_id, analysis_key))
    return ExtractionResult.model_validate(payload) if payload else None


# --------------------------------------------------------------------------
# Internos
# --------------------------------------------------------------------------


def _stage_source(ctx: StageContext, source_key: str) -> Path:
    """Copia el recorte al workspace para que la extracción pueda mutarlo."""
    target = ctx.workspace.path("extraction", Path(source_key).name)
    with ctx.storage.materialize(source_key) as origin:
        shutil.copy2(origin, target)
    return target


def _normalize_rows(raw, analysis_key: str) -> list[dict]:
    """Reduce la salida del extractor a una lista de filas.

    Un `dict` con clave `error` es un fallo de parseo del modelo, no un dato:
    se convierte en excepción para que el job termine en `failed` con la
    causa visible, en vez de guardar un artefacto que aparenta ser válido.
    """
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        if "error" in raw:
            raise RuntimeError(f"La extracción de '{analysis_key}' falló: {raw.get('error')}")
        return [raw]
    return []


def _pages_of(ctx: StageContext, document_id: str, analysis_key: str) -> list[int]:
    """Páginas que originaron la extracción, para poder auditar el resultado."""
    payload = ctx.storage.read_json_or_none(keys.search_result(document_id, analysis_key))
    if not payload:
        return []
    return list(payload.get("relevant_pages") or [])
