"""Etapa de control de calidad.

Dos capas, en este orden:

1. **Reglas contables deterministas** (`app/domain/accounting.py`): el activo
   cuadra con el pasivo más el patrimonio, la utilidad neta con la utilidad
   antes de impuestos, el saldo final de caja con el inicial más la
   variación. No cuestan nada, no alucinan y nombran el descuadre exacto.
2. **Revisión con el modelo**, solo si el análisis declara prompt de calidad:
   mira lo que una suma no ve (signos invertidos, períodos cruzados,
   conceptos mal clasificados).

Un descuadre en las reglas rechaza la extracción aunque el modelo la apruebe.

La etapa es independiente de la extracción a propósito: se puede corregir un
dato a mano y volver a validarlo sin pagar otra extracción, que es la
operación cara. Por eso hay dos entradas: `check_document`, que valida lo que
se extrajo y guarda el veredicto, y `check_rows`, que valida filas que llegan
de fuera sin necesidad de que exista un documento.
"""

from __future__ import annotations

import logging

from app.adapters.ai_extractor import QA_ERRORS_KEY, QA_STATUS_KEY, run_quality_review
from app.domain.accounting import run_checks
from app.domain.analysis import AnalysisType
from app.domain.errors import StageNotReadyError, StageNotSupportedError
from app.domain.models import QualityResult
from app.services.context import StageContext
from app.services.limits import llm_slot
from app.storage import keys

logger = logging.getLogger(__name__)

APPROVED = "Aprobado"
REJECTED = "Rechazado"
ERROR = "Error"


def check_document(ctx: StageContext, document_id: str, analysis: AnalysisType) -> QualityResult:
    """Valida la extracción guardada de `analysis` y persiste el veredicto."""
    _require_support(analysis)

    payload = ctx.storage.read_json_or_none(keys.extraction_result(document_id, analysis.key))
    if payload is None:
        raise StageNotReadyError(
            "quality",
            requires="extraction",
            hint=f"Ejecuta la extracción de '{analysis.key}' primero.",
        )

    rows = payload.get("rows") or []
    result = _run(ctx, analysis, rows, identity=payload.get("identity"))
    ctx.storage.write_json(
        keys.quality_result(document_id, analysis.key), result.model_dump(mode="json")
    )
    logger.info(
        "Calidad %s en %s -> %s (%d filas)",
        analysis.key,
        document_id,
        result.status,
        result.rows_checked,
    )
    return result


def check_rows(ctx: StageContext, analysis: AnalysisType, rows: list[dict]) -> QualityResult:
    """Valida filas provistas por el cliente, sin tocar el storage."""
    _require_support(analysis)
    return _run(ctx, analysis, rows)


def get_quality(ctx: StageContext, document_id: str, analysis_key: str) -> QualityResult | None:
    payload = ctx.storage.read_json_or_none(keys.quality_result(document_id, analysis_key))
    return QualityResult.model_validate(payload) if payload else None


def _require_support(analysis: AnalysisType) -> None:
    if not analysis.supports_quality:
        raise StageNotSupportedError("quality", analysis.key)


def _run(
    ctx: StageContext,
    analysis: AnalysisType,
    rows: list[dict],
    *,
    identity: dict | None = None,
) -> QualityResult:
    ctx.progress.report(f"Reglas contables de {analysis.label}...", 0.1)
    checks = run_checks(analysis.key, rows, tolerance=ctx.settings.quality_tolerance)
    errors: list = [check.detail for check in checks if check.failed]
    status = REJECTED if errors else APPROVED

    if analysis.quality_prompt:
        ctx.progress.report(f"Revisión con el modelo de {analysis.label}...", 0.4)
        ctx.cancellation.raise_if_cancelled()
        prompt = str(ctx.settings.prompt_path(analysis.quality_prompt))
        with llm_slot():
            raw = run_quality_review(rows, prompt, identity=identity)
        if not isinstance(raw, dict):
            raw = {}
        review_status = str(raw.get(QA_STATUS_KEY) or ERROR)
        errors.extend(raw.get(QA_ERRORS_KEY) or [])
        status = _combine(status, review_status)

    ctx.progress.report("Control de calidad terminado.", 1.0)
    return QualityResult(
        analysis_type=analysis.key,
        status=status,
        errors=errors,
        checks=[check.to_dict() for check in checks],
        llm_review=analysis.quality_prompt is not None,
        rows_checked=len(rows),
    )


def _combine(rules_status: str, review_status: str) -> str:
    """El veredicto más severo de los dos: Rechazado > Error > Aprobado."""
    if REJECTED in (rules_status, review_status):
        return REJECTED
    if review_status != APPROVED:
        return ERROR
    return rules_status
