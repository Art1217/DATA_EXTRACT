"""Etapa de búsqueda: localizar las páginas relevantes para un análisis.

Recorre el documento por bloques preguntándole al modelo qué páginas contienen
las tablas del análisis pedido, y recorta un PDF con solo esas páginas. Ese
recorte es lo que consume la extracción, que es la etapa cara: cuanto mejor
sea la búsqueda, menos páginas paga la extracción.

El recorte se expone además como operación propia (`cut_pages`), para poder
corregir a mano la lista de páginas y volver a extraer sin repetir la
búsqueda.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from app.adapters.ai_searcher import search_pages_block
from app.adapters.pdf_utils import extract_pages
from app.domain.analysis import AnalysisType
from app.domain.models import SearchResult
from app.services.block_runner import BlockOutcome, run_in_blocks
from app.services.context import StageContext
from app.services.identity import UNKNOWN_ENTITY
from app.services.params import ScanParams
from app.storage import keys

logger = logging.getLogger(__name__)


def search_pages(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    *,
    params: ScanParams = ScanParams(),
    entity_context: str = UNKNOWN_ENTITY,
    cut: bool = True,
) -> SearchResult:
    """Localiza las páginas relevantes y, opcionalmente, genera el recorte."""
    resolved = params.resolve(ctx.settings, analysis)
    prompt = str(ctx.settings.prompt_path(analysis.search_prompt))

    ctx.progress.report(f"Buscando páginas de {analysis.label}...", 0.0)
    search_progress = ctx.progress.slice(0.0, 0.9 if cut else 1.0)
    with ctx.storage.materialize(keys.source_pdf(document_id)) as pdf_path:
        outcomes = run_in_blocks(
            pdf_path,
            lambda images: search_pages_block(
                images, prompt_path=prompt, entity_context=entity_context
            ),
            workspace_dir=ctx.workspace.subdir("search"),
            block_size=resolved.block_size,
            workers=resolved.workers,
            dpi=resolved.dpi,
            progress=search_progress,
            cancellation=ctx.cancellation,
            max_pages=resolved.max_pages,
            label="Bloque",
        )

    sections, alerts = _collect_sections(outcomes)
    relevant_pages = _unique_pages(sections)

    result = SearchResult(
        analysis_type=analysis.key,
        relevant_pages=relevant_pages,
        sections=sections,
        alerts=alerts,
        blocks_total=len(outcomes),
        blocks_failed=sum(1 for outcome in outcomes if not outcome.ok),
    )

    if cut and relevant_pages:
        ctx.progress.report("Generando el recorte del PDF...", 0.92)
        result.cut_pdf_key = cut_pages(ctx, document_id, analysis, relevant_pages)

    ctx.storage.write_json(
        keys.search_result(document_id, analysis.key), result.model_dump(mode="json")
    )
    ctx.progress.report(f"Búsqueda terminada: {len(relevant_pages)} páginas relevantes.", 1.0)
    logger.info(
        "Búsqueda %s en %s -> %d páginas (%d bloques fallidos)",
        analysis.key,
        document_id,
        len(relevant_pages),
        result.blocks_failed,
    )
    return result


def cut_pages(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    pages: Sequence[int],
) -> str:
    """Recorta el documento a `pages` y guarda el PDF. Devuelve su clave."""
    if not pages:
        raise ValueError("No hay páginas que recortar.")

    output_dir = ctx.workspace.subdir("cut")
    with ctx.storage.materialize(keys.source_pdf(document_id)) as pdf_path:
        produced = extract_pages(
            str(pdf_path),
            list(pages),
            output_filename=f"corte_{analysis.key}.pdf",
            output_dir=str(output_dir),
        )
    if not produced:
        raise RuntimeError(f"No se pudo recortar el PDF de {document_id}.")

    key = keys.search_pdf(document_id, analysis.key)
    ctx.storage.import_file(Path(produced), key, move=True)
    return key


def get_search_result(
    ctx: StageContext, document_id: str, analysis_key: str
) -> SearchResult | None:
    payload = ctx.storage.read_json_or_none(keys.search_result(document_id, analysis_key))
    return SearchResult.model_validate(payload) if payload else None


# --------------------------------------------------------------------------
# Normalización de la respuesta del modelo
# --------------------------------------------------------------------------


def _collect_sections(outcomes: Iterable[BlockOutcome]) -> tuple[list[dict], list[str]]:
    """Aplana las respuestas de los bloques a una lista de secciones.

    El modelo devuelve el mismo contenido en varias formas según el bloque:
    un objeto con `relevant_sections`, una lista suelta de secciones, o
    respuestas completas anidadas dentro de `relevant_sections`. Las tres se
    han observado en producción, así que se aceptan las tres.
    """
    sections: list[dict] = []
    alerts: list[str] = []

    for outcome in outcomes:
        if not outcome.ok:
            continue
        payload = outcome.value

        if isinstance(payload, list):
            payload = {"relevant_sections": _flatten_list(payload)}
        if not isinstance(payload, dict):
            continue

        for item in payload.get("relevant_sections") or []:
            if not isinstance(item, dict):
                continue
            if "page_number" in item:
                sections.append(item)
            elif "relevant_sections" in item:
                sections.extend(
                    nested
                    for nested in item.get("relevant_sections") or []
                    if isinstance(nested, dict)
                )

        block_alerts = payload.get("alerts")
        if isinstance(block_alerts, list):
            alerts.extend(str(alert) for alert in block_alerts)

    return sections, alerts


def _flatten_list(payload: list) -> list:
    flattened: list = []
    for item in payload:
        if isinstance(item, dict) and "relevant_sections" in item:
            flattened.extend(item.get("relevant_sections") or [])
        else:
            flattened.append(item)
    return flattened


def _unique_pages(sections: Sequence[dict]) -> list[int]:
    """Páginas distintas, en el orden en que aparecen en el documento."""
    pages: list[int] = []
    seen: set[int] = set()
    for section in sections:
        page = section.get("page_number")
        if page is None:
            continue
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number not in seen:
            seen.add(page_number)
            pages.append(page_number)
    return sorted(pages)
