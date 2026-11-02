"""Composición de etapas: el pipeline completo para un análisis.

Aquí no hay lógica propia. Solo decide qué etapas correr, en qué orden, y les reparte
la franja de progreso que le toca a cada una. Todo el trabajo vive en los
servicios de etapa, que siguen siendo invocables por separado.

Por defecto reutiliza lo que ya esté calculado. Repetir una búsqueda o una
extracción cuesta dinero, así que rehacerlas tiene que ser una decisión
explícita (`reuse=False`), nunca el comportamiento por omisión.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.domain.analysis import AnalysisType, Stage, get_analysis_type
from app.domain.models import ExtractionResult, QualityResult, SearchResult
from app.services import extraction, quality, search
from app.services import identity as identity_service
from app.services.context import StageContext
from app.services.params import ScanParams
from app.storage import keys

logger = logging.getLogger(__name__)

DEFAULT_STAGES: frozenset[Stage] = frozenset({Stage.SEARCH, Stage.EXTRACTION, Stage.QUALITY})


@dataclass(slots=True)
class PipelineOutcome:
    """Resultado de correr el pipeline sobre un análisis."""

    analysis_type: str
    identity: dict | None = None
    search: SearchResult | None = None
    extraction: ExtractionResult | None = None
    quality: QualityResult | None = None
    skipped: list[str] = field(default_factory=list)
    """Etapas que no se ejecutaron, con el motivo."""
    published: dict | None = None
    """Qué se subió al bucket y si se registró la fila en BigQuery."""


def run_pipeline(
    ctx: StageContext,
    document_id: str,
    analysis_key: str,
    *,
    stages: frozenset[Stage] = DEFAULT_STAGES,
    params: ScanParams = ScanParams(),
    reuse: bool = True,
) -> PipelineOutcome:
    """Ejecuta búsqueda, extracción y calidad para un análisis."""
    analysis = get_analysis_type(analysis_key)
    outcome = PipelineOutcome(analysis_type=analysis_key)

    if analysis.needs_identity:
        outcome.identity = identity_service.extract_identity(
            ctx.with_progress(0.0, 0.08), document_id, force=not reuse
        )

    if Stage.SEARCH in stages:
        outcome.search = _search(ctx, document_id, analysis, outcome, params, reuse)
    else:
        outcome.search = search.get_search_result(ctx, document_id, analysis.key)

    if Stage.EXTRACTION in stages:
        outcome.extraction = _extract(ctx, document_id, analysis, outcome, reuse)
    else:
        outcome.extraction = extraction.get_extraction(ctx, document_id, analysis.key)

    if Stage.QUALITY in stages:
        outcome.quality = _check_quality(ctx, document_id, analysis, outcome, reuse)

    if Stage.EXTRACTION in stages:
        _publish(ctx, document_id, analysis, outcome)

    ctx.progress.report("Pipeline finalizado.", 1.0)
    return outcome


def _publish(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    outcome: PipelineOutcome,
) -> None:
    """Sube los entregables al bucket y registra la fila de trazabilidad.

    Se publica siempre que se haya pedido la etapa de extracción, incluso si
    esta reutilizó un resultado guardado: republicar es justamente lo que hace
    falta cuando la primera publicación falló por permisos. Una corrida que no
    pide extracción (`stages=["search"]`) no publica nada, aunque encuentre una
    extracción vieja en el storage.

    Se publica aunque no se haya extraído nada: un informe revisado en el que
    no había resultados también es un resultado, y omitirlo haría que
    desapareciera del bucket y de BigQuery sin dejar rastro. Quien decide qué
    se sube en ese caso es `publish_deliverables`.

    Un fallo aquí no puede tumbar el job: la extracción ya está calculada,
    pagada y guardada en el storage. El import es local porque `publish`
    necesita el tipo `PipelineOutcome` de este módulo.
    """
    if not ctx.settings.publish_enabled:
        return
    from app.services.publish import publish_deliverables

    try:
        result = publish_deliverables(ctx, document_id, analysis, outcome)
        outcome.published = result.to_dict() if result else None
    except Exception:
        logger.exception("No se pudo publicar %s de %s", analysis.key, document_id)


def _search(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    outcome: PipelineOutcome,
    params: ScanParams,
    reuse: bool,
) -> SearchResult | None:
    if reuse:
        cached = search.get_search_result(ctx, document_id, analysis.key)
        if cached is not None:
            outcome.skipped.append("search: se reutilizó el resultado guardado")
            return cached
    return search.search_pages(
        ctx.with_progress(0.08, 0.55),
        document_id,
        analysis,
        params=params,
        entity_context=identity_service.entity_name(outcome.identity),
    )


def _extract(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    outcome: PipelineOutcome,
    reuse: bool,
) -> ExtractionResult | None:
    if not analysis.supports_extraction:
        outcome.skipped.append(f"extraction: '{analysis.key}' no define extracción")
        return None
    if reuse:
        cached = extraction.get_extraction(ctx, document_id, analysis.key)
        if cached is not None:
            outcome.skipped.append("extraction: se reutilizó el resultado guardado")
            return cached
    if not ctx.storage.exists(keys.search_pdf(document_id, analysis.key)):
        outcome.skipped.append("extraction: la búsqueda no encontró páginas relevantes")
        return None
    return extraction.extract_data(
        ctx.with_progress(0.55, 0.9), document_id, analysis, identity=outcome.identity
    )


def _check_quality(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    outcome: PipelineOutcome,
    reuse: bool,
) -> QualityResult | None:
    if not analysis.supports_quality:
        outcome.skipped.append(f"quality: '{analysis.key}' no define control de calidad")
        return None
    if outcome.extraction is None or not outcome.extraction.rows:
        outcome.skipped.append("quality: no hay filas extraídas que validar")
        return None
    if reuse:
        cached = quality.get_quality(ctx, document_id, analysis.key)
        if cached is not None:
            outcome.skipped.append("quality: se reutilizó el resultado guardado")
            return cached
    return quality.check_document(ctx.with_progress(0.9, 1.0), document_id, analysis)
