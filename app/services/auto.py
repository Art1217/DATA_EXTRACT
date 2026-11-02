"""Flujo automático: del PDF al paquete final sin elegir nada a mano.

Encadena plan (Fase 1 + Fase 2) con un pipeline por cada análisis detectado y
empaqueta el resultado.

Un análisis que falla no cancela los demás: se anota el error y se sigue. En
un informe con siete tablas, que una responda mal no justifica tirar el
trabajo de las otras seis, que además ya está pagado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.analysis import (
    Stage,
    find_by_group_and_code,
    get_analysis_type,
    parse_group,
)
from app.domain.models import Plan
from app.services import bundle as bundle_service
from app.services import classification
from app.services import identity as identity_service
from app.services.context import StageContext
from app.services.params import ScanParams
from app.services.pipeline import DEFAULT_STAGES, PipelineOutcome, run_pipeline
from app.storage import keys

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AutoAnalysisOutcome:
    analysis_type: str
    group: str | None
    code: str | None
    outcome: PipelineOutcome | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class AutoOutcome:
    document_id: str
    plan: Plan
    results: list[AutoAnalysisOutcome] = field(default_factory=list)
    bundle_key: str | None = None
    plan_only: bool = False
    published: dict | None = None
    """URIs del ZIP y del `plan.txt` en el bucket, si la publicación está activa."""


def run_auto(
    ctx: StageContext,
    document_id: str,
    *,
    params: ScanParams = ScanParams(),
    stages: frozenset[Stage] = DEFAULT_STAGES,
    plan_only: bool = False,
    reuse: bool = True,
) -> AutoOutcome:
    """Clasifica el documento y procesa todo lo que encuentre."""
    plan = classification.build_plan(
        ctx.with_progress(0.0, 0.25), document_id, params=params, force=not reuse
    )

    _write_plan_text(ctx, document_id, plan)

    if plan_only or not plan.analysis_types:
        ctx.progress.report("Plan generado.", 1.0)
        return AutoOutcome(document_id=document_id, plan=plan, plan_only=True)

    # La identidad es la misma para todos los análisis del documento. Se
    # calcula una vez aquí y las etapas siguientes la leen del storage, en
    # lugar de repetir la misma llamada una vez por análisis.
    identity_service.extract_identity(ctx.with_progress(0.25, 0.30), document_id, force=not reuse)

    results: list[AutoAnalysisOutcome] = []
    total = len(plan.analysis_types)
    for index, analysis_key in enumerate(plan.analysis_types):
        ctx.cancellation.raise_if_cancelled()
        analysis = get_analysis_type(analysis_key)
        start = 0.30 + 0.60 * (index / total)
        end = 0.30 + 0.60 * ((index + 1) / total)
        ctx.progress.report(f"Procesando {analysis.label} ({index + 1}/{total})...", 0.0)

        try:
            outcome = run_pipeline(
                ctx.with_progress(start, end),
                document_id,
                analysis_key,
                stages=stages,
                params=params,
                reuse=reuse,
            )
            results.append(
                AutoAnalysisOutcome(
                    analysis_type=analysis_key,
                    group=analysis.group.value if analysis.group else None,
                    code=analysis.code,
                    outcome=outcome,
                )
            )
        except Exception as exc:
            logger.exception("Falló el análisis %s de %s", analysis_key, document_id)
            results.append(
                AutoAnalysisOutcome(
                    analysis_type=analysis_key,
                    group=analysis.group.value if analysis.group else None,
                    code=analysis.code,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    # El `plan.txt` se reescribe con el resultado de cada análisis antes de
    # empaquetar, para que el ZIP lo lleve ya completo.
    _write_plan_text(ctx, document_id, plan, results)

    # El ZIP es lo que se descarga. Al bucket no va —allí viven los archivos
    # sueltos, ya subidos por cada análisis— pero sí el plan en texto, que es
    # la portada de la corrida.
    bundle_key = bundle_service.build_bundle(
        ctx.with_progress(0.90, 1.0), document_id, analysis_types=plan.analysis_types
    )
    published = _publish_plan(ctx, document_id)
    ctx.progress.report("Flujo automático finalizado.", 1.0)
    return AutoOutcome(
        document_id=document_id,
        plan=plan,
        results=results,
        bundle_key=bundle_key,
        published=published,
    )


def _publish_plan(ctx: StageContext, document_id: str) -> dict | None:
    """Sube el `plan.txt` al bucket. Nunca hace fallar el job."""
    if not ctx.settings.publish_enabled:
        return None
    from app.services.publish import publish_plan

    try:
        return publish_plan(ctx, document_id) or None
    except Exception:
        logger.exception("No se pudo publicar el plan de %s", document_id)
        return None


# --------------------------------------------------------------------------
# Plan legible
# --------------------------------------------------------------------------


def _write_plan_text(
    ctx: StageContext,
    document_id: str,
    plan: Plan,
    results: list[AutoAnalysisOutcome] | None = None,
) -> None:
    """Guarda el plan en texto plano: qué grupos y qué análisis salieron.

    El `plan.json` es para máquinas; este `.txt` acompaña al ZIP y al bucket
    para que quien reciba el paquete sepa de un vistazo qué contiene sin abrir
    ningún archivo.
    """
    try:
        text = _plan_text(ctx, document_id, plan, results)
        ctx.storage.write_bytes(keys.plan_text(document_id), text.encode("utf-8"))
    except Exception:
        logger.exception("No se pudo escribir el plan en texto de %s", document_id)


def _plan_text(
    ctx: StageContext,
    document_id: str,
    plan: Plan,
    results: list[AutoAnalysisOutcome] | None,
) -> str:
    meta = ctx.storage.read_json_or_none(keys.document_meta(document_id)) or {}
    by_key = {item.analysis_type: item for item in (results or [])}

    lines = [
        "PLAN DE PROCESAMIENTO AUTOMÁTICO",
        "=" * 60,
        f"Documento : {meta.get('filename', '(sin nombre)')}",
        f"ID        : {document_id}",
        f"Páginas   : {meta.get('pages', '?')}",
        f"Generado  : {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
    ]

    if not plan.groups:
        lines.append("No se detectó ningún estado financiero ni nota en el documento.")
        return "\n".join(lines) + "\n"

    for group, codes in plan.groups.items():
        lines.append(f"GRUPO: {group}")
        if not codes:
            lines.append("  (sin análisis detectados)")
        for code in codes:
            lines.append(f"  - {_analysis_line(group, code, by_key)}")
        lines.append("")

    lines.append(f"Análisis detectados: {len(plan.analysis_types)}")
    if results is not None:
        lines.append(f"Procesados sin error: {sum(1 for item in results if item.ok)}")
    return "\n".join(lines) + "\n"


def _analysis_line(group: str, code: str, by_key: dict[str, AutoAnalysisOutcome]) -> str:
    try:
        analysis = find_by_group_and_code(parse_group(group), code)
    except Exception:
        analysis = None
    if analysis is None:
        return f"{code}"

    line = f"{code:<20} [{analysis.key}]"
    item = by_key.get(analysis.key)
    if item is None:
        return line
    if item.error:
        return f"{line}  ERROR: {item.error}"
    extraction = item.outcome.extraction if item.outcome else None
    filas = extraction.row_count if extraction else 0
    quality = item.outcome.quality if item.outcome else None
    estado = f"  {filas} filas"
    if quality:
        estado += f", calidad: {quality.status}"
    return line + estado
