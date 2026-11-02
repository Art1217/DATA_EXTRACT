"""Etapa de clasificación.

Dos operaciones independientes, cada una expuesta como su propio endpoint:

* **Fase 1 — grupo**: recorre el documento entero y determina qué grupos de
  contenido tiene (estados principales, notas).
* **Fase 2 — análisis**: recorre el documento otra vez, con el prompt de un
  grupo concreto, y determina qué análisis hay dentro de él.

Fase 2 recibe el grupo como parámetro y no exige que Fase 1 se haya
ejecutado. Quien ya sabe que su documento trae los estados principales puede
ir directo y ahorrarse una pasada completa, que es la parte cara.
`build_plan()` encadena ambas para quien no lo sabe.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from app.adapters.ai_classifier import classify_pages_block
from app.domain.analysis import (
    CLASSIFICATION_ANALYSIS_PROMPTS,
    CLASSIFICATION_GROUP_PROMPT,
    Group,
    analyses_for_group,
    find_by_group_and_code,
    valid_codes,
)
from app.domain.models import (
    AnalysisClassification,
    GroupClassification,
    PageFinding,
    Plan,
)
from app.services.block_runner import BlockOutcome, run_in_blocks
from app.services.context import StageContext
from app.services.params import ScanParams
from app.storage import keys

logger = logging.getLogger(__name__)

_GROUP_VALUES = frozenset(g.value for g in Group)
_GROUP_FIELD = "grupo"
_CODE_FIELD = "codigo"


def classify_groups(
    ctx: StageContext, document_id: str, *, params: ScanParams = ScanParams()
) -> GroupClassification:
    """Fase 1: qué grupos de contenido tiene el documento."""
    resolved = params.resolve(ctx.settings, classification=True)
    prompt = str(ctx.settings.prompt_path(CLASSIFICATION_GROUP_PROMPT))

    ctx.progress.report("Fase 1: clasificando grupos de contenido...", 0.0)
    with ctx.storage.materialize(keys.source_pdf(document_id)) as pdf_path:
        outcomes = run_in_blocks(
            pdf_path,
            lambda images: classify_pages_block(images, prompt_path=prompt),
            workspace_dir=ctx.workspace.subdir("grupo"),
            block_size=resolved.block_size,
            workers=resolved.workers,
            dpi=resolved.dpi,
            progress=ctx.progress,
            cancellation=ctx.cancellation,
            max_pages=resolved.max_pages,
            label="Bloque de grupos",
        )

    findings = _collect_findings(outcomes, field=_GROUP_FIELD, valid=_GROUP_VALUES)
    result = GroupClassification(
        groups=sorted({f.value for f in findings}),
        detail=findings,
        pages_analyzed=_pages_analyzed(outcomes),
        blocks_failed=sum(1 for o in outcomes if not o.ok),
    )
    ctx.storage.write_json(keys.classification_group(document_id), result.model_dump(mode="json"))
    logger.info("Fase 1 en %s -> %s", document_id, result.groups or "ninguno")
    return result


def classify_analysis(
    ctx: StageContext,
    document_id: str,
    group: Group,
    *,
    params: ScanParams = ScanParams(),
) -> AnalysisClassification:
    """Fase 2: qué análisis contiene el documento dentro de `group`."""
    resolved = params.resolve(ctx.settings, classification=True)
    prompt = str(ctx.settings.prompt_path(CLASSIFICATION_ANALYSIS_PROMPTS[group]))
    allowed = valid_codes(group)

    ctx.progress.report(f"Fase 2: clasificando análisis de {group.value}...", 0.0)
    with ctx.storage.materialize(keys.source_pdf(document_id)) as pdf_path:
        outcomes = run_in_blocks(
            pdf_path,
            lambda images: classify_pages_block(images, prompt_path=prompt),
            workspace_dir=ctx.workspace.subdir(f"analisis_{group.value}"),
            block_size=resolved.block_size,
            workers=resolved.workers,
            dpi=resolved.dpi,
            progress=ctx.progress,
            cancellation=ctx.cancellation,
            max_pages=resolved.max_pages,
            label=f"Bloque de {group.value}",
        )

    findings = _collect_findings(outcomes, field=_CODE_FIELD, valid=allowed)
    # En orden de catálogo: el balance antes que sus notas, como en el informe.
    found = {f.value for f in findings}
    codes = [code for code in available_codes(group) if code in found]
    result = AnalysisClassification(
        group=group.value,
        analyses=codes,
        analysis_types=_codes_to_keys(group, codes),
        detail=findings,
        blocks_failed=sum(1 for o in outcomes if not o.ok),
    )
    ctx.storage.write_json(
        keys.classification_analysis(document_id, group.value),
        result.model_dump(mode="json"),
    )
    logger.info("Fase 2 en %s (%s) -> %s", document_id, group.value, codes or "ninguno")
    return result


def build_plan(
    ctx: StageContext,
    document_id: str,
    *,
    params: ScanParams = ScanParams(),
    force: bool = False,
) -> Plan:
    """Encadena Fase 1 y una Fase 2 por cada grupo encontrado.

    Reutiliza los resultados ya guardados salvo que se pida `force`: repetir
    una clasificación cuesta una pasada completa por el documento.
    """
    group_result = _cached_or_run(
        ctx.storage,
        keys.classification_group(document_id),
        GroupClassification,
        force,
        lambda: classify_groups(ctx.with_progress(0.0, 0.4), document_id, params=params),
    )

    groups = [Group(value) for value in group_result.groups]
    plan: dict[str, list[str]] = {}
    analysis_types: list[str] = []

    for index, group in enumerate(groups):
        start = 0.4 + 0.6 * (index / len(groups))
        end = 0.4 + 0.6 * ((index + 1) / len(groups))
        analysis_result = _cached_or_run(
            ctx.storage,
            keys.classification_analysis(document_id, group.value),
            AnalysisClassification,
            force,
            lambda g=group, s=start, e=end: classify_analysis(
                ctx.with_progress(s, e), document_id, g, params=params
            ),
        )
        plan[group.value] = analysis_result.analyses
        analysis_types.extend(analysis_result.analysis_types)

    result = Plan(groups=plan, analysis_types=analysis_types)
    ctx.storage.write_json(keys.plan(document_id), result.model_dump(mode="json"))
    ctx.progress.report("Plan de procesamiento listo.", 1.0)
    return result


# --------------------------------------------------------------------------
# Internos
# --------------------------------------------------------------------------


def _collect_findings(
    outcomes: Iterable[BlockOutcome], *, field: str, valid: Iterable[str]
) -> list[PageFinding]:
    """Aplana las respuestas de los bloques y descarta lo que no es del catálogo.

    El filtro contra `valid` no es defensa contra datos raros sino contra
    alucinaciones: el modelo devuelve de vez en cuando etiquetas que no
    existen, y sin filtrar acabarían generando análisis inexistentes en el
    plan.
    """
    allowed = frozenset(valid)
    findings: list[PageFinding] = []
    for outcome in outcomes:
        if not outcome.ok or not isinstance(outcome.value, dict):
            continue
        for entry in outcome.value.get("clasificaciones") or []:
            if not isinstance(entry, dict):
                continue
            page_number = entry.get("page_number")
            value = entry.get(field)
            if page_number is None or value not in allowed:
                continue
            findings.append(
                PageFinding(
                    page_number=int(page_number),
                    value=str(value),
                    confidence=entry.get("confianza"),
                    evidence=entry.get("evidencia"),
                )
            )
    findings.sort(key=lambda f: (f.page_number, f.value))
    return findings


def _codes_to_keys(group: Group, codes: Sequence[str]) -> list[str]:
    """Traduce etiquetas de la Fase 2 a claves del catálogo."""
    resolved = []
    for code in codes:
        analysis = find_by_group_and_code(group, code)
        if analysis is None:
            logger.warning("Código '%s' de %s sin equivalente en el catálogo", code, group)
            continue
        resolved.append(analysis.key)
    return resolved


def _pages_analyzed(outcomes: Sequence[BlockOutcome]) -> int:
    return max((o.last_page for o in outcomes), default=0)


def _cached_or_run(storage, key: str, model, force: bool, run):
    """Devuelve el artefacto guardado, o ejecuta la etapa si falta o se fuerza."""
    if not force:
        payload = storage.read_json_or_none(key)
        if payload is not None:
            return model.model_validate(payload)
    return run()


def available_codes(group: Group) -> list[str]:
    """Códigos que el catálogo reconoce para un grupo."""
    return [a.code for a in analyses_for_group(group) if a.code]
